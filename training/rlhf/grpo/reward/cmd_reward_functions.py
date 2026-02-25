"""
치과 명령어 GRPO Reward Functions

CommandRewardFunction과 각 component들을 정의합니다.
"""
import logging
import re
import difflib
import os
from collections import Counter
import pandas as pd
from typing import Dict, Any, List, Set, Optional
from reward.reward_functions import RewardComponent

logger = logging.getLogger(__name__)


class CommandInfoLoader:
    """명령어 정보를 파일에서 동적으로 로드하는 유틸리티 클래스"""
    
    def __init__(self, csv_file_path: Optional[str] = None, data_csv_path: Optional[str] = None):
        """
        Args:
            csv_file_path: cmd_bot.csv 파일 경로
            data_csv_path: data.csv 파일 경로 (ground truth 패턴 학습용)
        """
        self.csv_file_path = csv_file_path
        self.data_csv_path = data_csv_path
        self.command_info = {}  # cmd_bot.csv에서 로드한 명령어 정보
        self.gt_patterns = {}  # data.csv에서 학습한 ground truth 패턴
        
        self._load_command_info()
        self._learn_gt_patterns()
    
    def _find_file(self, filename: str) -> Optional[str]:
        """파일을 여러 경로에서 찾기"""
        possible_paths = [
            filename,
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), filename),
            os.path.join(os.getcwd(), filename),
            os.path.join(os.path.dirname(__file__), filename)
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _load_command_info(self):
        """cmd_bot.csv에서 명령어 정보 로드"""
        if not self.csv_file_path:
            self.csv_file_path = self._find_file("cmd_bot.csv")
        
        if not self.csv_file_path or not os.path.exists(self.csv_file_path):
            logger.warning(f"⚠️ cmd_bot.csv 파일을 찾을 수 없습니다. 기본 설정으로 진행합니다.")
            return
        
        try:
            df = pd.read_csv(self.csv_file_path)
            for _, row in df.iterrows():
                cmd = row['cmd'].lower().strip()
                need_num = str(row.get('need_num', 'FALSE')).upper() == 'TRUE'
                need_surface = str(row.get('need_surface', 'FALSE')).upper() == 'TRUE'
                need_bridge = str(row.get('need_bridge', 'FALSE')).upper() == 'TRUE'
                is_status = str(row.get('is_status', 'FALSE')).upper() == 'TRUE'
                is_control = str(row.get('is_control', 'FALSE')).upper() == 'TRUE'
                category = str(row.get('category', '')).lower()
                
                self.command_info[cmd] = {
                    'need_num': need_num,
                    'need_surface': need_surface,
                    'need_bridge': need_bridge,
                    'is_status': is_status,
                    'is_control': is_control,
                    'category': category
                }
            
            logger.info(f"✅ {len(self.command_info)}개 명령어 정보 로드 완료")
        except Exception as e:
            logger.warning(f"⚠️ cmd_bot.csv 로드 실패: {e}. 기본 설정으로 진행합니다.")
    
    def _learn_gt_patterns(self):
        """data.csv에서 ground truth 패턴 학습"""
        if not self.data_csv_path:
            self.data_csv_path = self._find_file("data.csv")
            # data.csv가 없으면 llm_training/rlhf/grpo/data.csv 시도
            if not self.data_csv_path:
                possible_data_paths = [
                    "llm_training/rlhf/grpo/data.csv",
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data.csv")
                ]
                for path in possible_data_paths:
                    if os.path.exists(path):
                        self.data_csv_path = path
                        break
        
        if not self.data_csv_path or not os.path.exists(self.data_csv_path):
            logger.debug("⚠️ data.csv 파일을 찾을 수 없습니다. 패턴 학습을 건너뜁니다.")
            return
        
        try:
            df = pd.read_csv(self.data_csv_path)
            if 'ground_truth' not in df.columns:
                logger.debug("⚠️ data.csv에 ground_truth 컬럼이 없습니다.")
                return
            
            # 각 ground truth에서 명령어 패턴 추출
            command_patterns = {}  # 명령어별 패턴 통계
            
            for _, row in df.iterrows():
                gt = str(row['ground_truth']).lower()
                if not gt or gt == 'nan':
                    continue
                
                # 명령어들을 세미콜론으로 분리
                commands = [c.strip() for c in gt.split(';') if c.strip()]
                
                for cmd_str in commands:
                    # 명령어 추출
                    cmd_match = re.search(r'number\s+\d+.*?,\s*([^,]+?)(?:,|$)', cmd_str)
                    if not cmd_match:
                        continue
                    
                    command = cmd_match.group(1).strip().lower()
                    
                    # 패턴 분석
                    has_surface = any(s in cmd_str for s in ['buccal', 'lingual', 'palatal'])
                    has_position = any(p in cmd_str for p in ['mesial', 'middle', 'distal'])
                    has_value = bool(re.search(r'\d+\s+\d+\s+\d+', cmd_str))
                    has_single_value = bool(re.search(r',\s*\d+\s*;', cmd_str) or re.search(r',\s*\d+\s*$', cmd_str))
                    
                    if command not in command_patterns:
                        command_patterns[command] = {
                            'total': 0,
                            'with_surface': 0,
                            'with_position': 0,
                            'with_value': 0,
                            'with_single_value': 0
                        }
                    
                    command_patterns[command]['total'] += 1
                    if has_surface:
                        command_patterns[command]['with_surface'] += 1
                    if has_position:
                        command_patterns[command]['with_position'] += 1
                    if has_value:
                        command_patterns[command]['with_value'] += 1
                    if has_single_value:
                        command_patterns[command]['with_single_value'] += 1
            
            # 패턴을 확률로 변환
            for cmd, stats in command_patterns.items():
                total = stats['total']
                if total > 0:
                    self.gt_patterns[cmd] = {
                        'surface_prob': stats['with_surface'] / total,
                        'position_prob': stats['with_position'] / total,
                        'value_prob': stats['with_value'] / total,
                        'single_value_prob': stats['with_single_value'] / total
                    }
            
            logger.info(f"✅ {len(self.gt_patterns)}개 명령어 패턴 학습 완료")
        except Exception as e:
            logger.warning(f"⚠️ data.csv 패턴 학습 실패: {e}")
    
    def get_command_info(self, command: str) -> Dict[str, Any]:
        """명령어 정보 가져오기"""
        cmd_lower = command.lower().strip()
        
        # 직접 매칭
        if cmd_lower in self.command_info:
            info = self.command_info[cmd_lower].copy()
            # GT 패턴 추가
            if cmd_lower in self.gt_patterns:
                info['gt_pattern'] = self.gt_patterns[cmd_lower]
            return info
        
        # 부분 매칭
        for cmd_key, info in self.command_info.items():
            if cmd_key in cmd_lower or cmd_lower in cmd_key:
                info = info.copy()
                if cmd_key in self.gt_patterns:
                    info['gt_pattern'] = self.gt_patterns[cmd_key]
                return info
        
        # 기본값
        return {
            'need_num': True,
            'need_surface': False,
            'need_bridge': False,
            'is_status': False,
            'is_control': False,
            'category': 'unknown'
        }
    
    def get_all_commands(self) -> Set[str]:
        """모든 명령어 목록 반환"""
        return set(self.command_info.keys())
    
    def get_status_commands(self) -> Set[str]:
        """STATUS 명령어 목록 반환"""
        return {cmd for cmd, info in self.command_info.items() if info.get('is_status', False)}
    
    def get_value_required_commands(self) -> Set[str]:
        """값이 필요한 명령어 목록 반환 (GT 패턴 기반)"""
        value_commands = set()
        for cmd, pattern in self.gt_patterns.items():
            if pattern.get('value_prob', 0) > 0.5 or pattern.get('single_value_prob', 0) > 0.5:
                value_commands.add(cmd)
        return value_commands
    
    def get_surface_keywords(self) -> Set[str]:
        """표면 키워드 반환"""
        return {'buccal', 'lingual', 'palatal'}
    
    def get_position_keywords(self) -> Set[str]:
        """위치 키워드 반환"""
        return {'mesial', 'middle', 'distal'}


class ToothNumberComponent(RewardComponent):
    """
    치아 번호 매칭 (F1 + 엄격한 페널티)
    가중치: 0.40
    """
    
    def __init__(self, weight: float = 0.40):
        super().__init__(name="tooth_number", weight=weight)
        
        # 엄격한 설정
        self.perfect_match_bonus = 0.2  # 완벽 매칭 시 보너스
        self.hallucination_penalty = 0.4  # 없는 치아 추가 시 감점
        self.missing_penalty = 0.3  # 치아 누락 시 감점
        self.wrong_tooth_penalty = 0.5  # 완전히 다른 치아 시 감점
    
    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        """
        엄격한 치아 번호 평가
        
        Args:
            completion: 생성된 텍스트 (str)
            ground_truth: 정답 라벨 (str)
            **kwargs: 추가 인자 (labels 등)
        """
        # 형식 안전 처리
        if ground_truth is None and 'labels' in kwargs:
            ground_truth = kwargs.get('labels', [])
            if isinstance(ground_truth, list) and len(ground_truth) > 0:
                ground_truth = ground_truth[0]
        
        if ground_truth is None:
            ground_truth = ""
        
        # str로 변환
        completion = str(completion) if not isinstance(completion, str) else completion
        ground_truth = str(ground_truth) if not isinstance(ground_truth, str) else ground_truth
        
        gen_teeth = self._extract_teeth_strict(completion)
        gt_teeth = self._extract_teeth_strict(ground_truth)
        
        if not gt_teeth:
            # Ground truth에 치아 번호가 없는 경우
            return 1.0 if not gen_teeth else 0.3
        
        if not gen_teeth:
            # 치아 번호를 아예 생성하지 않음 → 심각한 오류
            return 0.0
        
        gen_set = set(gen_teeth)
        gt_set = set(gt_teeth)
        
        # 정확한 매칭 계산
        correct = gen_set & gt_set  # True Positives
        hallucinated = gen_set - gt_set  # False Positives
        missing = gt_set - gen_set  # False Negatives
        
        # F1 Score
        if len(gen_set) == 0:
            precision = 0.0
        else:
            precision = len(correct) / len(gen_set)
        
        if len(gt_set) == 0:
            recall = 0.0
        else:
            recall = len(correct) / len(gt_set)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        # 기본 점수
        score = f1
        
        # 완벽 매칭 보너스
        if gen_set == gt_set:
            score += self.perfect_match_bonus
        
        # Hallucination 페널티 (없는 치아 추가)
        if hallucinated:
            hallucination_ratio = len(hallucinated) / len(gt_set)
            score -= self.hallucination_penalty * hallucination_ratio
            
            # 매우 심각한 hallucination (2배 이상 추가)
            if len(hallucinated) >= len(gt_set):
                score -= 0.2
        
        # 누락 페널티
        if missing:
            missing_ratio = len(missing) / len(gt_set)
            score -= self.missing_penalty * missing_ratio
        
        # 완전히 틀린 경우 강력한 페널티
        if len(correct) == 0:
            score -= self.wrong_tooth_penalty
        
        # 0.0 ~ 1.0 범위로 클리핑
        return max(0.0, min(1.0, score))
    
    def _extract_teeth_strict(self, text: str) -> List[int]:
        """
        엄격한 치아 번호 추출
        - number X 패턴만 인정
        """
        teeth = []
        
        # 1) "number X" 패턴 (단일)
        single_pattern = r'\bnumber\s+(\d+)\b'
        for match in re.finditer(single_pattern, text, flags=re.IGNORECASE):
            tooth_num = int(match.group(1))
            if 1 <= tooth_num <= 32 or 11 <= tooth_num <= 48:  # UNS or FDI 범위
                teeth.append(tooth_num)
        
        # 2) "number X-Y" 패턴 (범위)
        range_pattern = r'\bnumber\s+(\d+)\s*-\s*(\d+)\b'
        for match in re.finditer(range_pattern, text, flags=re.IGNORECASE):
            start, end = int(match.group(1)), int(match.group(2))
            if start <= end and start >= 1:
                teeth.extend(range(start, end + 1))
        
        # 3) "number X, Y, Z" 패턴 (콤마로 구분된 복수)
        multi_pattern = r'\bnumber\s+(\d+(?:\s*,\s*\d+)+)'
        for match in re.finditer(multi_pattern, text, flags=re.IGNORECASE):
            numbers_str = match.group(1)
            numbers = re.findall(r'\d+', numbers_str)
            for n in numbers:
                tooth_num = int(n)
                if 1 <= tooth_num <= 32 or 11 <= tooth_num <= 48:
                    teeth.append(tooth_num)
        
        return teeth


class CommandKeywordComponent(RewardComponent):
    """
    명령어 키워드 정확 매칭 (Label 기반)
    가중치: 0.20
    """
    
    def __init__(self, weight: float = 0.20, csv_file_path: Optional[str] = None, data_csv_path: Optional[str] = None):
        super().__init__(name="command_keyword", weight=weight)
        
        # 명령어 정보 로더
        self.loader = CommandInfoLoader(csv_file_path=csv_file_path, data_csv_path=data_csv_path)
        
        # 허용된 정확한 명령어 목록 (파일에서 동적 로드)
        self.valid_commands = self.loader.get_all_commands()
        
        # 핵심 명령어 정의 (파일에서 로드한 명령어 중 주요 명령어)
        self.critical_commands = {
            'probing', 'pocket', 'depth',
            'mobility', 'furcation', 'bleeding',
            'implant', 'fixture', 'crown', 'bridge', 'pontic',
            'suppuration', 'calculus', 'plaque'
        }
        # 파일에 있는 명령어와 교집합
        self.critical_commands = self.critical_commands & self.valid_commands
        
        # 잘못된 명령어 매핑 (자동 수정) - 기본 매핑 유지하되 확장 가능
        self.invalid_command_mapping = {
            'probe': 'probing',
            'tooth': 'number',
            'pd': 'probing',
            'pocketdepth': 'pocket depth',
            'gingivalmargin': 'gingival margin',
            'mucogingivaljunction': 'mucogingival junction',
            'rootrest': 'root rest',
            'primaryteeth': 'primary teeth'
        }
    
    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        """
        Label의 각 명령어 단위(;로 구분)를 정확히 매칭
        """
        # kwargs에서 labels 추출
        if ground_truth is None and 'labels' in kwargs:
            ground_truth = kwargs.get('labels', [])
            if isinstance(ground_truth, list) and len(ground_truth) > 0:
                ground_truth = ground_truth[0]
        
        # ground_truth 형식 변환
        if isinstance(ground_truth, list):
            if len(ground_truth) > 0:
                ground_truth = ground_truth[0] if isinstance(ground_truth[0], str) else str(ground_truth[0])
            else:
                ground_truth = ""
        elif isinstance(ground_truth, dict):
            ground_truth = ground_truth.get("content", "")
        elif ground_truth is None:
            ground_truth = ""
        
        # completion 형식 변환
        if not isinstance(completion, str):
            if isinstance(completion, list):
                completion = " ".join(str(item) for item in completion)
            else:
                completion = str(completion)
        
        # Label을 명령어 단위로 파싱
        gen_commands = self._parse_commands(completion)
        gt_commands = self._parse_commands(ground_truth)
        
        if not gt_commands:
            return 1.0 if not gen_commands else 0.5
        
        if not gen_commands:
            return 0.0
        
        # 각 GT 명령어에 대해 가장 유사한 Gen 명령어 찾기
        match_scores = []
        for gt_cmd in gt_commands:
            best_score = 0.0
            for gen_cmd in gen_commands:
                score = self._command_similarity(gen_cmd, gt_cmd)
                best_score = max(best_score, score)
            match_scores.append(best_score)
        
        # 평균 매칭 점수
        avg_score = sum(match_scores) / len(match_scores)
        
        # 핵심 명령어 누락 페널티
        gt_critical = self._extract_critical_commands(ground_truth)
        gen_critical = self._extract_critical_commands(completion)
        
        if gt_critical:
            critical_recall = len(gen_critical & gt_critical) / len(gt_critical)
            if critical_recall < 0.5:
                avg_score *= 0.7  # 핵심 명령어 절반 이상 누락 시 감점
        
        return avg_score
    
    def _parse_commands(self, text: str) -> List[Dict[str, str]]:
        """
        명령어를 구조화된 딕셔너리로 파싱
        
        Example:
            "number 7, pocket depth 5 3 2" 
            → {"tooth": "7", "command": "pocket depth", "values": "5 3 2"}
        """
        commands = []
        for cmd_str in text.split(';'):
            cmd_str = cmd_str.strip()
            if not cmd_str:
                continue
            
            # "number X, command values" 패턴
            match = re.match(
                r'number\s+([\d\-,\s]+),\s*([^,]+?)(?:,\s*(.+))?$',
                cmd_str,
                re.IGNORECASE
            )
            
            if match:
                tooth = match.group(1).strip()
                command = match.group(2).strip().lower()
                values = match.group(3).strip() if match.group(3) else ""
                
                commands.append({
                    "tooth": tooth,
                    "command": command,
                    "values": values
                })
        
        return commands
    
    def _command_similarity(self, cmd1: Dict, cmd2: Dict) -> float:
        """두 명령어의 유사도"""
        # 치아 번호 일치 필수
        if cmd1["tooth"] != cmd2["tooth"]:
            return 0.0
        
        # 명령어 키워드 비교 (동의어 고려)
        cmd_score = self._keyword_match(cmd1["command"], cmd2["command"])
        
        # 수치값 비교
        if cmd1["values"] and cmd2["values"]:
            value_score = 1.0 if cmd1["values"] == cmd2["values"] else 0.7
        else:
            value_score = 1.0 if not cmd1["values"] and not cmd2["values"] else 0.8
        
        return (cmd_score * 0.7 + value_score * 0.3)
    
    def _keyword_match(self, kw1: str, kw2: str) -> float:
        """키워드 매칭 (동의어 고려 및 정확도 검증 강화)"""
        # 잘못된 명령어 체크 (매우 강한 penalty)
        original_kw1 = kw1
        original_kw2 = kw2
        
        # 잘못된 명령어 사용 시 매우 강한 penalty
        invalid_keywords = {'probe', 'tooth'}
        if original_kw1 in invalid_keywords or original_kw2 in invalid_keywords:
            # "probe"나 "tooth" 같은 잘못된 명령어 사용 시 즉시 0점
            return 0.0  # 잘못된 명령어가 포함되면 0점
        
        # 잘못된 명령어 자동 수정
        kw1 = self.invalid_command_mapping.get(kw1, kw1)
        kw2 = self.invalid_command_mapping.get(kw2, kw2)
        
        # 정확한 일치 (최우선)
        if kw1 == kw2:
            return 1.0
        
        # 동의어 매핑
        synonyms = {
            'probing': 'pocket_depth',
            'pocket depth': 'pocket_depth',
            'pocket': 'pocket_depth',
            'depth': 'pocket_depth',
            'fixture': 'implant',  # 일부 컨텍스트에서
        }
        
        kw1_normalized = synonyms.get(kw1, kw1)
        kw2_normalized = synonyms.get(kw2, kw2)
        
        if kw1_normalized == kw2_normalized:
            return 0.95  # 동의어 매칭은 약간 낮은 점수
        
        # 허용된 명령어 목록과 비교
        kw1_valid = kw1 in self.valid_commands or any(kw1 in cmd for cmd in self.valid_commands)
        kw2_valid = kw2 in self.valid_commands or any(kw2 in cmd for cmd in self.valid_commands)
        
        # 둘 다 허용된 명령어가 아니면 낮은 점수
        if not kw1_valid and not kw2_valid:
            # 부분 매칭 시도
            words1 = set(kw1.split())
            words2 = set(kw2.split())
            if words1 & words2:
                return 0.3  # 부분 매칭은 더 낮은 점수 (0.5 -> 0.3)
            return 0.0
        
        # 하나만 허용된 명령어면 매우 낮은 점수
        if kw1_valid != kw2_valid:
            return 0.1  # 0.3 -> 0.1
        
        # 부분 매칭 (둘 다 허용된 명령어인 경우)
        words1 = set(kw1.split())
        words2 = set(kw2.split())
        
        if words1 & words2:
            return 0.7
        
        return 0.0
    
    def _extract_critical_commands(self, text: str) -> Set[str]:
        """핵심 명령어 추출"""
        text_lower = text.lower()
        found = set()
        for cmd in self.critical_commands:
            if cmd in text_lower:
                found.add(cmd)
        return found


class InstructionComplianceComponent(RewardComponent):
    """
    Instruction 준수 여부 및 환각 체크 (Reflection & Penalty)
    가중치: 0.12
    """
    def __init__(self, weight: float = 0.12, csv_file_path: Optional[str] = None, data_csv_path: Optional[str] = None):
        super().__init__(name="instruction_compliance", weight=weight)
        
        # 명령어 정보 로더
        self.loader = CommandInfoLoader(csv_file_path=csv_file_path, data_csv_path=data_csv_path)
        
        # 금지된 메타 명령어 패턴 (확장되지 않은 형태)
        self.forbidden_patterns = [
            r'\brepeat\b', r'\bothers\b', r'\ball\b', r'\bagain\b', r'\bexcept\b'
        ]
        
        # 기본 허용 키워드 (파일에서 동적 생성)
        surface_keywords = self.loader.get_surface_keywords()
        position_keywords = self.loader.get_position_keywords()
        all_commands = self.loader.get_all_commands()
        
        # 명령어에서 단어 추출
        command_words = set()
        for cmd in all_commands:
            command_words.update(cmd.split())
        
        self.base_keywords = {
            'number', 'jump', 'back', 'clear', 'delete'
        } | surface_keywords | position_keywords | command_words
        
        # 허용된 명령어 목록 (파일에서 동적 로드)
        self.valid_commands = all_commands
        
        # 잘못된 명령어 패턴 (강한 페널티) - 기본 패턴 유지하되 확장 가능
        self.invalid_command_patterns = {
            r'\bprobe\b': 'probing',  # probe -> probing
            r'\btooth\s+\d+': 'number',  # tooth N -> number N
            r'\btooth\s+number': 'number',  # tooth number -> number
        }

    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        score = 1.0
        completion_lower = str(completion).lower()
        ground_truth_lower = str(ground_truth).lower() if ground_truth else ""

        # 1. 금지된 명령어 사용 체크 (Penalty)
        for pattern in self.forbidden_patterns:
            if re.search(pattern, completion_lower):
                score -= 0.3  # 금지어 사용 시 큰 감점

        # 2. 잘못된 명령어 사용 체크 (매우 강한 페널티) - 강화
        invalid_command_penalty = 0.0
        invalid_command_count = 0
        
        # 명령어 셋에 없는 명령어 체크 (probe 등)
        commands = completion_lower.split(';')
        for cmd in commands:
            cmd = cmd.strip()
            if not cmd:
                continue
            
            # 각 명령어에서 실제 명령어 추출 (number 이후 부분)
            cmd_parts = re.split(r'number\s+\d+', cmd, maxsplit=1)
            if len(cmd_parts) > 1:
                remaining = cmd_parts[1].strip()
                # 쉼표로 분리하여 각 부분 확인
                parts = [p.strip() for p in remaining.split(',') if p.strip()]
                for part in parts:
                    part_lower = part.lower()
                    # 숫자만 있는 경우는 제외
                    if re.match(r'^\d+(\s+\d+)*$', part_lower):
                        continue
                    # surface나 position 키워드는 제외
                    if part_lower in self.loader.get_surface_keywords() or part_lower in self.loader.get_position_keywords():
                        continue
                    # 유효한 명령어인지 확인
                    is_valid = False
                    for valid_cmd in self.valid_commands:
                        if part_lower == valid_cmd.lower() or part_lower in valid_cmd.lower() or valid_cmd.lower() in part_lower:
                            is_valid = True
                            break
                    if not is_valid:
                        invalid_command_count += 1
                        # probe 같은 심각한 오류는 매우 강한 페널티
                        if 'probe' in part_lower:
                            invalid_command_penalty += 2.0  # probe는 매우 강한 페널티
                        else:
                            invalid_command_penalty += 1.0  # 다른 잘못된 명령어도 강한 페널티
        
        # 기존 패턴 기반 체크도 유지
        for pattern, correct_cmd in self.invalid_command_patterns.items():
            if re.search(pattern, completion_lower):
                matches = len(re.findall(pattern, completion_lower))
                # "tooth"나 "probe" 같은 심각한 오류는 더 강한 페널티
                if 'tooth' in pattern or 'probe' in pattern:
                    invalid_command_penalty += matches * 2.0  # 1.0 -> 2.0으로 강화
                else:
                    invalid_command_penalty += matches * 1.0  # 0.3 -> 1.0으로 강화
        
        score -= invalid_command_penalty
        if invalid_command_penalty > 0 or invalid_command_count > 0:
            # 잘못된 명령어가 하나라도 있으면 사실상 0점 처리 (더 강화)
            score *= 0.001  # 0.01 -> 0.001로 더 강화
            logger.debug(f"⚠️ Invalid command detected: {invalid_command_count} invalid commands, penalty: {invalid_command_penalty}")

        # 2.5. 명령어 완성 이전에 세미콜론이나 쉼표를 찍는 경우 강력한 페널티
        premature_separator_penalty = 0.0
        
        # 전체 completion에서 number 이전에 세미콜론이나 쉼표가 있는지 체크
        # 패턴 1: "command.\nnumber 1;" 같은 경우 - number 이전에 세미콜론
        if re.search(r'[^n][;,]+\s*number\s+\d+', completion_lower):
            premature_separator_penalty += 2.0
            logger.debug("⚠️ Premature separator before 'number' detected")
        
        # 패턴 2: ",number 1;" 같은 경우 - 쉼표로 시작
        if re.match(r'^\s*[,;]+\s*number\s+\d+', completion_lower):
            premature_separator_penalty += 2.0
            logger.debug("⚠️ Completion starts with separator")
        
        # 패턴 3: "command.\n" 같은 메타 텍스트 후 세미콜론
        if re.search(r'(command|output|result|answer)[.\s]*[;,]+\s*number', completion_lower):
            premature_separator_penalty += 2.0
            logger.debug("⚠️ Meta text followed by separator before 'number'")
        
        # 패턴 4: 각 명령어 내부에서 number 이전에 세미콜론/쉼표
        lines = completion_lower.split('\n')
        for line in lines:
            line = line.strip()
            # number 이전에 세미콜론이나 쉼표가 있는지 체크
            if re.search(r'[;,]+\s*number\s+\d+', line) and not re.match(r'^\s*number\s+\d+', line):
                premature_separator_penalty += 1.0
                logger.debug(f"⚠️ Premature separator in line: {line[:50]}")
        
        score -= premature_separator_penalty
        if premature_separator_penalty > 0:
            # 명령어 완성 이전에 세미콜론/쉼표를 찍으면 매우 강한 페널티
            score *= 0.01  # 99% 감점
        
        # 3. 포맷 규칙 체크: "number N, command" 형식이어야 함 (매우 강화)
        commands = completion_lower.split(';')
        valid_format_count = 0
        total_commands = 0
        format_errors = []
        comma_missing_count = 0
        semicolon_format_count = 0  # 세미콜론으로 구분된 잘못된 형식
        
        for cmd in commands:
            cmd = cmd.strip()
            if not cmd: continue
            total_commands += 1
            
            # 올바른 형식: "number N, command, ..." (쉼표로 구분)
            # 잘못된 형식: "number N; command" 또는 "number N"만 있음
            if re.match(r'^number\s+\d+', cmd):
                # "number N, command" 형식인지 확인
                if ',' in cmd:
                    # 쉼표가 있으면 형식상 올바름
                    # 추가 검증: "number N," 다음에 공백이나 명령어가 있어야 함
                    parts = cmd.split(',', 1)
                    if len(parts) > 1 and parts[1].strip():
                        valid_format_count += 1
                    else:
                        format_errors.append("missing_command_after_comma")
                else:
                    # "number N"만 있고 명령어가 없음 - 심각한 오류
                    format_errors.append("missing_command")
                    comma_missing_count += 1
            else:
                format_errors.append("no_number_prefix")
                # 세미콜론으로 구분된 잘못된 형식 체크
                if ';' in cmd or not cmd.startswith('number'):
                    semicolon_format_count += 1
        
        if total_commands > 0:
            format_ratio = valid_format_count / total_commands
            # 포맷 오류가 많으면 매우 강한 페널티
            if format_ratio < 0.5:
                score *= 0.05  # 절반 이상 형식 오류면 극심한 감점
            elif format_ratio < 0.8:
                score *= 0.4  # 80% 미만이면 큰 감점 (0.6 -> 0.4)
            else:
                score *= format_ratio
            
            # 쉼표 누락에 대한 추가 페널티 (강화)
            if comma_missing_count > 0:
                comma_penalty = (comma_missing_count / total_commands) * 0.5  # 0.3 -> 0.5
                score -= comma_penalty
            
            # 세미콜론 형식 사용에 대한 강한 페널티
            if semicolon_format_count > 0:
                semicolon_penalty = (semicolon_format_count / total_commands) * 0.4
                score -= semicolon_penalty

        # 강한 형식 실패 시 거의 0점 처리
        if total_commands == 0 or format_ratio < 0.3 or "the following output" in completion_lower:
            score *= 0.01

        # "command"와 같은 메타 텍스트로 시작하면 강한 페널티
        if re.search(r'^\s*command\b', completion_lower, re.MULTILINE):
            score *= 0.01
        if "the following output" in completion_lower or "based on the provided instructions" in completion_lower:
            score *= 0.01

        # 4. 환각 체크 (Reflection with GT) - 매우 강화
        if ground_truth_lower:
            gen_words = set(re.findall(r'[a-z]+', completion_lower))
            gt_words = set(re.findall(r'[a-z]+', ground_truth_lower))
            
            # GT에 없고, 기본 키워드도 아닌 단어들
            hallucinated_words = gen_words - gt_words - self.base_keywords
            if hallucinated_words:
                # 환각 단어가 있으면 매우 강한 감점
                # 특히 "command", "output", "here" 같은 불필요한 텍스트는 더 강한 페널티
                critical_hallucinations = {'command', 'output', 'here', 'is', 'the', 'answer', 'result'}
                critical_count = len(hallucinated_words & critical_hallucinations)
                if critical_count > 0:
                    penalty = min(0.9, critical_count * 0.3 + len(hallucinated_words) * 0.2)
                else:
                    penalty = min(0.7, len(hallucinated_words) * 0.2)  # 0.15 -> 0.2
                score -= penalty
        
        # 5. 불필요한 텍스트 포함 체크 (새로 추가)
        unnecessary_patterns = [
            r'^command\.?\s*$',  # "command." 또는 "command"
            r'^output\.?\s*$',   # "output." 또는 "output"
            r'^here\s+is',        # "here is"
            r'^the\s+result',    # "the result"
        ]
        for pattern in unnecessary_patterns:
            if re.search(pattern, completion_lower):
                score -= 0.5  # 불필요한 텍스트 포함 시 강한 페널티

        return max(0.0, score)


class SequenceSimilarityComponent(RewardComponent):
    """
    정답과의 시퀀스 유사도 (LCS ratio + Token-level matching)
    가중치: 0.12
    """
    def __init__(self, weight: float = 0.12):
        super().__init__(name="sequence_similarity", weight=weight)

    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        if not ground_truth:
            return 0.0
        
        completion = str(completion)
        ground_truth = str(ground_truth)
        
        # 1. 전체 문자열 유사도 (LCS)
        matcher = difflib.SequenceMatcher(None, completion, ground_truth)
        string_similarity = matcher.ratio()
        
        # 2. Token-level 유사도 (세미콜론으로 분리된 명령어 단위)
        gen_commands = [c.strip() for c in completion.split(';') if c.strip()]
        gt_commands = [c.strip() for c in ground_truth.split(';') if c.strip()]
        
        if not gt_commands:
            return 1.0 if not gen_commands else 0.0
        
        if not gen_commands:
            return 0.0
        
        # 각 GT 명령어에 대해 가장 유사한 Gen 명령어 찾기
        matched_indices = set()
        match_scores = []
        
        for gt_cmd in gt_commands:
            best_score = 0.0
            best_idx = -1
            for idx, gen_cmd in enumerate(gen_commands):
                if idx in matched_indices:
                    continue
                # 간단한 유사도 (공통 토큰 비율)
                score = self._command_token_similarity(gen_cmd, gt_cmd)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx >= 0:
                matched_indices.add(best_idx)
            match_scores.append(best_score)
        
        token_similarity = sum(match_scores) / len(gt_commands) if match_scores else 0.0
        
        # 3. 명령어 개수 페널티
        count_penalty = 1.0
        if len(gen_commands) != len(gt_commands):
            count_diff = abs(len(gen_commands) - len(gt_commands))
            count_penalty = 1.0 - (count_diff / max(len(gen_commands), len(gt_commands))) * 0.5
        
        # 종합 점수 (Token-level이 더 중요)
        final_score = (token_similarity * 0.6 + string_similarity * 0.4) * count_penalty
        
        return max(0.0, min(1.0, final_score))
    
    def _command_token_similarity(self, cmd1: str, cmd2: str) -> float:
        """두 명령어의 토큰 유사도"""
        # 토큰화 (공백, 쉼표로 분리)
        tokens1 = set(re.findall(r'[a-z0-9]+', cmd1.lower()))
        tokens2 = set(re.findall(r'[a-z0-9]+', cmd2.lower()))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union) if union else 0.0


class StructuralComponent(RewardComponent):
    """구조 정확도 (가중치: 0.04)"""
    
    def __init__(self, weight: float = 0.04):
        super().__init__(name="command_structural", weight=weight)
    
    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        """명령어 개수 및 세미콜론 구조 매칭"""
        # 형식 안전 처리
        if ground_truth is None:
            return 0.0
        
        completion = str(completion)
        ground_truth = str(ground_truth)
        
        gen_count = completion.count(';') + 1
        gt_count = ground_truth.count(';') + 1
        
        count_diff = abs(gen_count - gt_count)
        max_count = max(gen_count, gt_count)
        
        return 1.0 - (count_diff / max_count) if max_count > 0 else 1.0


class NumericalValueComponent(RewardComponent):
    """수치값 정확도 및 Probing 값 형식 검증 (가중치: 0.08)"""
    
    def __init__(self, weight: float = 0.08, csv_file_path: Optional[str] = None, data_csv_path: Optional[str] = None):
        super().__init__(name="numerical_value", weight=weight)
        
        # 명령어 정보 로더
        self.loader = CommandInfoLoader(csv_file_path=csv_file_path, data_csv_path=data_csv_path)
        
        # 값이 필요한 명령어 (파일에서 동적 로드)
        self.value_required_commands = self.loader.get_value_required_commands()
        # 기본값 (파일에 없을 경우)
        if not self.value_required_commands:
            self.value_required_commands = {'probing', 'pocket depth', 'pocket'}
        
        # Probing 관련 명령어 (하위 호환성)
        self.probing_commands = {cmd for cmd in self.value_required_commands if 'probing' in cmd or 'pocket' in cmd}
        if not self.probing_commands:
            self.probing_commands = {'probing', 'pocket depth', 'pocket'}
    
    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        """Probing depths, grades 등 숫자 시퀀스 매칭 및 형식 검증"""
        # 형식 안전 처리
        if ground_truth is None:
            return 0.0
        
        completion = str(completion).lower()
        ground_truth = str(ground_truth).lower()
        
        # 1. "5 3 2" 같은 연속 숫자 패턴 추출 및 매칭
        try:
            gen_seqs = re.findall(r'\b(\d+(?:\s+\d+){2,})\b', completion)
            gt_seqs = re.findall(r'\b(\d+(?:\s+\d+){2,})\b', ground_truth)
        except Exception as e:
            logger.debug(f"Error in numerical value matching: {e}")
            return 0.0
        
        if not gt_seqs:
            return 1.0
        
        # 기본 매칭 점수
        matches = sum(1 for g in gt_seqs if g in gen_seqs)
        base_score = matches / len(gt_seqs) if gt_seqs else 1.0
        
        # 2. Probing 값 형식 검증 강화
        # Probing 명령어는 세 숫자가 공백으로 구분되어야 함 (쉼표 없이)
        probing_format_score = self._validate_probing_format(completion, ground_truth)
        
        # 종합 점수 (기본 매칭 70%, 형식 검증 30%)
        final_score = base_score * 0.7 + probing_format_score * 0.3
        
        return max(0.0, min(1.0, final_score))
    
    def _validate_probing_format(self, completion: str, ground_truth: str) -> float:
        """Probing 값 형식 검증: 세 숫자가 공백으로 구분되어 있는지 확인"""
        # 값이 필요한 명령어가 있는 명령어들 추출
        gt_commands = [c.strip() for c in ground_truth.split(';') if c.strip()]
        gen_commands = [c.strip() for c in completion.split(';') if c.strip()]
        
        # GT에서 값이 필요한 명령어 찾기
        gt_value_commands = []
        for cmd in gt_commands:
            for value_cmd in self.value_required_commands:
                if value_cmd in cmd.lower():
                    gt_value_commands.append(cmd)
                    break
        
        if not gt_value_commands:
            # GT에 값이 필요한 명령어가 없으면 형식 검증 불필요
            return 1.0
        
        format_scores = []
        for gt_cmd in gt_value_commands:
            # GT에서 값 추출 (세 숫자 패턴)
            # 모든 값이 필요한 명령어에 대해 패턴 매칭
            value_pattern = r',\s*(\d+\s+\d+\s+\d+)'
            gt_values_match = re.search(value_pattern, gt_cmd)
            if not gt_values_match:
                # 쉼표 없이도 시도
                value_pattern = r'\s+(\d+\s+\d+\s+\d+)'
                gt_values_match = re.search(value_pattern, gt_cmd)
            
            if not gt_values_match:
                continue
            
            gt_values = gt_values_match.group(1)
            gt_tooth_match = re.search(r'number\s+(\d+)', gt_cmd)
            if not gt_tooth_match:
                continue
            
            gt_tooth = gt_tooth_match.group(1)
            
            # 생성된 명령어에서 같은 치아의 값이 필요한 명령어 찾기
            matching_gen_cmd = None
            for gen_cmd in gen_commands:
                gen_tooth_match = re.search(r'number\s+(\d+)', gen_cmd)
                if not gen_tooth_match:
                    continue
                
                gen_tooth = gen_tooth_match.group(1)
                has_value_cmd = any(value_cmd in gen_cmd.lower() for value_cmd in self.value_required_commands)
                
                if gen_tooth == gt_tooth and has_value_cmd:
                    matching_gen_cmd = gen_cmd
                    break
            
            if not matching_gen_cmd:
                # 값이 필요한 명령어가 생성되지 않음
                format_scores.append(0.0)
                continue
            
            # 생성된 명령어에서 값 추출
            gen_values_match = re.search(r',\s*(\d+\s+\d+\s+\d+)', matching_gen_cmd)
            if not gen_values_match:
                # 쉼표 없이도 시도
                gen_values_match = re.search(r'\s+(\d+\s+\d+\s+\d+)', matching_gen_cmd)
            
            if not gen_values_match:
                # Probing 값이 올바른 형식이 아님
                format_scores.append(0.0)
                continue
            
            gen_values = gen_values_match.group(1)
            
            # 형식 검증: 세 숫자가 공백으로 구분되어 있는지 확인
            # 올바른 형식: "3 3 3" (공백으로 구분)
            # 잘못된 형식: "3,3,3" (쉼표로 구분) 또는 "3;3;3" (세미콜론으로 구분)
            if re.match(r'^\d+\s+\d+\s+\d+$', gen_values):
                # 올바른 형식
                if gen_values == gt_values:
                    format_scores.append(1.0)
                else:
                    # 값은 다르지만 형식은 올바름
                    format_scores.append(0.7)
            else:
                # 잘못된 형식 (쉼표나 세미콜론으로 구분됨)
                format_scores.append(0.2)
        
        if not format_scores:
            return 1.0
        
        return sum(format_scores) / len(format_scores)


class ComponentRewardWrapper:
    """개별 component를 독립적인 reward function으로 동작하게 만드는 wrapper"""
    
    __name__ = "ComponentRewardWrapper"
    
    def __init__(self, component, component_name: str, weight: float = 1.0):
        self.component = component
        self.component_name = component_name
        self._name = component_name
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Component의 reward 계산"""
        # completions가 리스트가 아닌 경우 리스트로 변환
        if not isinstance(completions, list):
            completions = [completions]
        
        # ground_truth 처리: 리스트가 아니면 각 completion에 대해 동일한 값 사용
        ground_truths = kwargs.get("ground_truth", None)
        if ground_truths is not None:
            if not isinstance(ground_truths, list):
                # 단일 값이면 각 completion에 대해 동일한 값 사용
                ground_truths = [ground_truths] * len(completions)
            elif len(ground_truths) != len(completions):
                # 길이가 다르면 첫 번째 값으로 패딩
                if len(ground_truths) > 0:
                    ground_truths = ground_truths[:len(completions)] + [ground_truths[0]] * (len(completions) - len(ground_truths))
                else:
                    ground_truths = [None] * len(completions)
        else:
            ground_truths = [None] * len(completions)
        
        rewards = []
        for i, completion in enumerate(completions):
            # completion 처리: list, dict, str 모두 처리
            if isinstance(completion, list):
                # 리스트인 경우 첫 번째 항목이나 content 추출
                if len(completion) > 0 and isinstance(completion[0], dict):
                    completion_text = completion[0].get("content", "")
                else:
                    completion_text = " ".join(str(item) for item in completion)
            elif isinstance(completion, dict):
                # dict인 경우 content 추출
                completion_text = completion.get("content", "")
            else:
                # str인 경우 그대로 사용
                completion_text = str(completion)
            
            # 각 completion에 해당하는 ground_truth 전달
            component_kwargs = kwargs.copy()
            component_kwargs['ground_truth'] = ground_truths[i] if i < len(ground_truths) else None
            
            try:
                reward = self.component.calculate(completion_text, **component_kwargs)
            except Exception as e:
                logger.warning(f"Error calculating reward for component {self.component_name}: {e}")
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards
    
    def reward_func(self, completions, **kwargs) -> List[float]:
        """Component의 reward 계산 (별칭)"""
        return self.__call__(completions, **kwargs)
    
    def __repr__(self):
        return f"ComponentRewardWrapper({self.component_name})"
    
    def __str__(self):
        return self.component_name


class OrderPenaltyComponent(RewardComponent):
    """
    명령어 순서 오류 페널티
    가중치: 0.10
    """
    def __init__(self, weight: float = 0.10):
        super().__init__(name="order_penalty", weight=weight)
    
    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        if not ground_truth:
            return 0.0
        
        completion = str(completion)
        ground_truth = str(ground_truth)
        
        # 명령어를 세미콜론으로 분리
        gen_commands = [c.strip() for c in completion.split(';') if c.strip()]
        gt_commands = [c.strip() for c in ground_truth.split(';') if c.strip()]
        
        if not gt_commands:
            return 1.0 if not gen_commands else 0.0
        
        if not gen_commands:
            return 0.0
        
        # 각 명령어를 정규화하여 비교 가능하게 만들기
        gen_normalized = [self._normalize_command(cmd) for cmd in gen_commands]
        gt_normalized = [self._normalize_command(cmd) for cmd in gt_commands]
        
        # LCS를 사용하여 순서가 맞는 명령어 찾기
        lcs_length = self._lcs_length(gen_normalized, gt_normalized)
        
        # 순서 정확도 = LCS 길이 / GT 명령어 개수
        order_accuracy = lcs_length / len(gt_normalized) if gt_normalized else 0.0
        
        # 순서 오류 페널티 계산
        # 순서가 완전히 맞으면 1.0, 순서가 많이 틀렸으면 0.0에 가까움
        score = order_accuracy
        
        # 추가 페널티: 순서가 완전히 뒤바뀐 경우
        if len(gen_normalized) == len(gt_normalized):
            # 역순인지 확인
            if gen_normalized == list(reversed(gt_normalized)):
                score = 0.2  # 완전 역순이면 강한 페널티
        
        # 순서 오류가 심각한 경우 추가 페널티
        if order_accuracy < 0.5:
            # 절반 이상 순서가 틀렸으면 추가 감점
            score *= 0.5
        
        return max(0.0, score)
    
    def _normalize_command(self, cmd: str) -> str:
        """명령어를 정규화하여 비교 가능하게 만들기"""
        # 소문자 변환 및 공백 정규화
        normalized = re.sub(r'\s+', ' ', cmd.lower().strip())
        # 숫자 부분은 유지하되, 공백 제거
        normalized = re.sub(r'(\d+)\s+(\d+)', r'\1 \2', normalized)
        return normalized
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """두 시퀀스의 최장 공통 부분 수열(LCS) 길이 계산"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # 명령어 유사도 계산 (정확 일치 또는 높은 유사도)
                similarity = self._command_similarity(seq1[i-1], seq2[j-1])
                if similarity > 0.8:  # 80% 이상 유사하면 같은 명령어로 간주
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _command_similarity(self, cmd1: str, cmd2: str) -> float:
        """두 명령어의 유사도 계산"""
        if cmd1 == cmd2:
            return 1.0
        
        # 토큰 기반 유사도
        tokens1 = set(re.findall(r'[a-z0-9]+', cmd1))
        tokens2 = set(re.findall(r'[a-z0-9]+', cmd2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union) if union else 0.0


class ExactMatchComponent(RewardComponent):
    """
    정확한 매칭 보너스 (Ground Truth와 완전 일치)
    가중치: 0.04
    """
    def __init__(self, weight: float = 0.04):
        super().__init__(name="exact_match", weight=weight)
    
    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        if not ground_truth:
            return 0.0
        
        completion = str(completion).strip()
        ground_truth = str(ground_truth).strip()
        
        # 완전 일치
        if completion == ground_truth:
            return 1.0
        
        # 공백/대소문자 정규화 후 비교
        completion_norm = re.sub(r'\s+', ' ', completion.lower().strip())
        ground_truth_norm = re.sub(r'\s+', ' ', ground_truth.lower().strip())
        
        if completion_norm == ground_truth_norm:
            return 0.95
        
        # 부분 일치 (GT의 주요 부분이 포함되어 있는지)
        # GT의 80% 이상이 포함되어 있으면 보너스
        if len(ground_truth_norm) > 0:
            overlap_ratio = len(set(ground_truth_norm.split()) & set(completion_norm.split())) / len(set(ground_truth_norm.split()))
            if overlap_ratio >= 0.8:
                return 0.7
        
        return 0.0


class CommandFormatComponent(RewardComponent):
    """
    명령어 형식 검증: number, surface(필요한 경우), command, value(필요한 경우) 형식 검증
    cmd_bot.csv의 명령어 특성(need_surface, need_num 등)을 참고하여 검증
    가중치: 0.10
    """
    def __init__(self, weight: float = 0.10, csv_file_path: Optional[str] = None, data_csv_path: Optional[str] = None):
        super().__init__(name="command_format", weight=weight)
        
        # 명령어 정보 로더
        self.loader = CommandInfoLoader(csv_file_path=csv_file_path, data_csv_path=data_csv_path)
        
        # 표면 키워드 (파일에서 동적 로드)
        self.surface_keywords = self.loader.get_surface_keywords()
        # 위치 키워드 (STATUS 명령어에서 사용)
        self.position_keywords = self.loader.get_position_keywords()
        # 값이 필요한 명령어 (파일에서 동적 로드)
        self.value_required_commands = self.loader.get_value_required_commands()
        # STATUS 명령어 (파일에서 동적 로드)
        self.status_commands = self.loader.get_status_commands()
    
    def _get_command_info(self, command: str) -> Dict[str, Any]:
        """명령어 정보 가져오기"""
        return self.loader.get_command_info(command)
    
    def _parse_command_structure(self, cmd_str: str) -> Dict[str, Any]:
        """명령어 구조 파싱: number, surface(선택), command, value(선택)"""
        cmd_str = cmd_str.strip()
        if not cmd_str:
            return {}
        
        # "number N" 추출
        number_match = re.search(r'number\s+(\d+(?:\s*,\s*\d+)*)', cmd_str, re.IGNORECASE)
        if not number_match:
            return {'error': 'no_number'}
        
        number_part = number_match.group(0)
        remaining = cmd_str[len(number_part):].strip()
        
        # 쉼표로 구분된 부분들 추출
        parts = [p.strip() for p in remaining.split(',') if p.strip()]
        
        structure = {
            'number': number_part,
            'surface': None,
            'positions': [],  # mesial, middle, distal
            'command': None,
            'values': [],
            'has_comma_format': ',' in remaining
        }
        
        # 각 부분 분석
        for i, part in enumerate(parts):
            part_lower = part.lower()
            
            # 표면 확인
            if part_lower in self.surface_keywords:
                structure['surface'] = part
                continue
            
            # 위치 확인 (mesial, middle, distal)
            if part_lower in self.position_keywords:
                structure['positions'].append(part)
                continue
            
            # 명령어 확인
            if not structure['command']:
                # 명령어 후보들 확인
                is_command = False
                for cmd in self.value_required_commands | self.status_commands | {'implant', 'fixture', 'crown', 'bridge', 'pontic', 'missing', 'impacted'}:
                    if cmd in part_lower:
                        structure['command'] = part
                        is_command = True
                        break
                
                if not is_command and i == 0:  # 첫 번째 부분이 명령어일 가능성
                    structure['command'] = part
                continue
            
            # 값 확인 (숫자)
            if re.match(r'^\d+(\s+\d+)*$', part):
                structure['values'].append(part)
        
        return structure
    
    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        if not ground_truth:
            return 0.0
        
        completion = str(completion).lower()
        ground_truth = str(ground_truth).lower()
        
        # 명령어들을 세미콜론으로 분리
        gen_commands = [c.strip() for c in completion.split(';') if c.strip()]
        gt_commands = [c.strip() for c in ground_truth.split(';') if c.strip()]
        
        if not gt_commands:
            return 1.0
        
        if not gen_commands:
            return 0.0
        
        # 각 명령어 형식 검증
        format_scores = []
        
        for gen_cmd in gen_commands:
            gen_structure = self._parse_command_structure(gen_cmd)
            
            if 'error' in gen_structure:
                format_scores.append(0.0)
                continue
            
            # 기본 형식 검증
            score = 1.0
            
            # 1. 쉼표 형식 검증 (매우 강한 penalty)
            if not gen_structure['has_comma_format']:
                score *= 0.05  # 쉼표 형식이 아니면 사실상 0점에 가깝게
            
            # 2. 명령어 추출 및 검증
            if not gen_structure['command']:
                score *= 0.0  # 명령어가 없으면 즉시 0
                format_scores.append(score)
                continue
            
            cmd_info = self._get_command_info(gen_structure['command'])
            
            # 3. 표면 필요 여부 검증
            if cmd_info['need_surface'] or cmd_info['is_status']:
                if not gen_structure['surface']:
                    # STATUS 명령어라도 표면이 없으면 강하게 감점 (testset 패턴 반영)
                    score *= 0.05
            else:
                # 표면이 필요 없는데 있으면 감점하지 않음 (선택적이므로)
                pass
            
            # 4. 값 필요 여부 검증
            command_lower = gen_structure['command'].lower()
            if command_lower in self.value_required_commands:
                if not gen_structure['values']:
                    score *= 0.05  # 값이 필요한데 없으면 거의 0점
                else:
                    # 값 형식 검증 (세 숫자여야 함)
                    for val in gen_structure['values']:
                        if not re.match(r'^\d+\s+\d+\s+\d+$', val):
                            score *= 0.1  # 형식이 올바르지 않으면 강하게 감점
            
            # STATUS 명령어는 확장되어야 함 (별도 Component에서 처리)
            if cmd_info['is_status']:
                # 최소한 mesial, middle, distal 중 하나는 있어야 함
                if not gen_structure['positions']:
                    score *= 0.05  # 위치가 없으면 매우 강한 감점
            
            format_scores.append(score)
        
        if not format_scores:
            return 0.0
        
        # 평균 형식 점수
        avg_score = sum(format_scores) / len(format_scores)
        
        # GT와 비교하여 형식 일치도 확인
        gt_format_scores = []
        for gt_cmd in gt_commands:
            gt_structure = self._parse_command_structure(gt_cmd)
            if 'error' not in gt_structure and gt_structure.get('has_comma_format'):
                gt_format_scores.append(1.0)
            else:
                gt_format_scores.append(0.0)
        
        if gt_format_scores:
            gt_format_ratio = sum(gt_format_scores) / len(gt_format_scores)
            # GT 형식 비율과 비교하여 추가 보정
            if gt_format_ratio > 0.9 and avg_score < 0.5:
                avg_score *= 0.7  # GT는 올바른데 생성 결과가 형식이 틀리면 추가 감점
        
        return max(0.0, min(1.0, avg_score))


class StatusExpansionComponent(RewardComponent):
    """
    STATUS 명령어 확장 검증 (bleeding/suppuration 등이 mesial/middle/distal로 확장되었는지)
    가중치: 0.12
    """
    def __init__(self, weight: float = 0.12, csv_file_path: Optional[str] = None, data_csv_path: Optional[str] = None):
        super().__init__(name="status_expansion", weight=weight)
        
        # 명령어 정보 로더
        self.loader = CommandInfoLoader(csv_file_path=csv_file_path, data_csv_path=data_csv_path)
        
        # STATUS 명령어 목록 (파일에서 동적 로드)
        self.status_commands = self.loader.get_status_commands()
        # 기본값 (파일에 없을 경우)
        if not self.status_commands:
            self.status_commands = {'bleeding', 'suppuration', 'plaque', 'calculus'}
        
        # 필수 확장 위치
        self.required_positions = self.loader.get_position_keywords()
    
    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        if not ground_truth:
            return 0.0
        
        completion = str(completion).lower()
        ground_truth = str(ground_truth).lower()
        
        # GT에서 STATUS 명령어가 있는 명령어들 추출
        gt_commands = [c.strip() for c in ground_truth.split(';') if c.strip()]
        gen_commands = [c.strip() for c in completion.split(';') if c.strip()]
        
        # GT에서 STATUS 명령어가 포함된 명령어 찾기
        gt_status_commands = []
        for cmd in gt_commands:
            for status_cmd in self.status_commands:
                if status_cmd in cmd:
                    gt_status_commands.append(cmd)
                    break
        
        if not gt_status_commands:
            # GT에 STATUS 명령어가 없으면 이 component는 점수에 영향 없음
            return 1.0
        
        # 각 GT STATUS 명령어에 대해 확장 검증
        expansion_scores = []
        for gt_cmd in gt_status_commands:
            # GT 명령어에서 치아 번호와 STATUS 명령어 추출
            gt_tooth_match = re.search(r'number\s+(\d+)', gt_cmd)
            if not gt_tooth_match:
                continue
            
            gt_tooth = gt_tooth_match.group(1)
            gt_status = None
            for status_cmd in self.status_commands:
                if status_cmd in gt_cmd:
                    gt_status = status_cmd
                    break
            
            if not gt_status:
                continue
            
            # GT에서 확장된 위치 확인 (mesial, middle, distal)
            gt_positions = set()
            for pos in self.required_positions:
                if pos in gt_cmd:
                    gt_positions.add(pos)
            
            # 생성된 명령어에서 같은 치아와 STATUS 명령어 찾기
            matching_gen_commands = []
            for gen_cmd in gen_commands:
                gen_tooth_match = re.search(r'number\s+(\d+)', gen_cmd)
                if not gen_tooth_match:
                    continue
                
                gen_tooth = gen_tooth_match.group(1)
                gen_status = None
                for status_cmd in self.status_commands:
                    if status_cmd in gen_cmd:
                        gen_status = status_cmd
                        break
                
                if gen_tooth == gt_tooth and gen_status == gt_status:
                    matching_gen_commands.append(gen_cmd)
            
            if not matching_gen_commands:
                # STATUS 명령어가 생성되지 않음 - 강한 페널티
                expansion_scores.append(0.0)
                continue
            
            # 생성된 명령어에서 확장된 위치 확인
            gen_positions = set()
            for gen_cmd in matching_gen_commands:
                for pos in self.required_positions:
                    if pos in gen_cmd:
                        gen_positions.add(pos)
            
            # 확장 검증: GT에 있는 모든 위치가 생성 결과에도 있어야 함
            if gt_positions:
                if gen_positions >= gt_positions:
                    # 모든 필수 위치가 확장됨
                    expansion_scores.append(1.0)
                else:
                    # 일부 위치가 누락됨
                    missing_positions = gt_positions - gen_positions
                    missing_ratio = len(missing_positions) / len(gt_positions)
                    # 누락되면 거의 0에 가깝게 감점
                    expansion_scores.append(max(0.0, 1.0 - missing_ratio * 1.2))
            else:
                # GT에 위치가 없으면 (이상한 경우) 기본 점수
                expansion_scores.append(0.5)
        
        if not expansion_scores:
            return 1.0
        
        # 평균 확장 점수
        avg_score = sum(expansion_scores) / len(expansion_scores)
        
        # 추가 페널티: STATUS 명령어가 전혀 확장되지 않은 경우
        if avg_score < 0.7:
            avg_score *= 0.2  # 확장이 부족하면 더 강하게 감점
        
        return max(0.0, min(1.0, avg_score))


class SelfAuditComponent(RewardComponent):
    """
    Self-reward: 생성된 command 리스트만으로 스스로 형식/필수 요소를 감사하여 보상.
    - 모든 명령이 number, [surface], command, [value] 형식인지
    - cmd_bot.csv의 valid command인지
    - 필요 surface/value/position이 충족되는지
    - number-only 스팸, 잘못된 probe/tooth 등 즉시 강한 감점
    가중치: 0.16
    """
    def __init__(self, weight: float = 0.16, csv_file_path: Optional[str] = None, data_csv_path: Optional[str] = None):
        super().__init__(name="self_audit", weight=weight)
        self.loader = CommandInfoLoader(csv_file_path=csv_file_path, data_csv_path=data_csv_path)
        self.valid_commands = self.loader.get_all_commands()
        self.surface_keywords = self.loader.get_surface_keywords()
        self.position_keywords = self.loader.get_position_keywords()
        self.value_required_commands = self.loader.get_value_required_commands()
        self.status_commands = self.loader.get_status_commands()

    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        completion_original = str(completion)
        completion = completion_original.lower()
        if not completion.strip():
            return 0.0

        # 명령어 완성 이전에 세미콜론/쉼표를 찍는 경우 강력한 페널티
        premature_separator_penalty = 0.0
        
        # 패턴 1: "command.\nnumber 1;" 같은 경우 - number 이전에 세미콜론/쉼표
        if re.search(r'[^n][;,]+\s*number\s+\d+', completion):
            premature_separator_penalty += 2.0
            logger.debug("⚠️ Premature separator before 'number' detected in SelfAudit")
        
        # 패턴 2: ",number 1;" 같은 경우 - 쉼표/세미콜론으로 시작
        if re.match(r'^\s*[,;]+\s*number\s+\d+', completion):
            premature_separator_penalty += 2.0
            logger.debug("⚠️ Completion starts with separator in SelfAudit")
        
        # 패턴 3: "command.\n" 같은 메타 텍스트 후 세미콜론/쉼표
        if re.search(r'(command|output|result|answer)[.\s]*[;,]+\s*number', completion):
            premature_separator_penalty += 2.0
            logger.debug("⚠️ Meta text followed by separator before 'number' in SelfAudit")
        
        # 패턴 4: 각 줄에서 number 이전에 세미콜론/쉼표
        lines = completion.split('\n')
        for line in lines:
            line = line.strip()
            if re.search(r'[;,]+\s*number\s+\d+', line) and not re.match(r'^\s*number\s+\d+', line):
                premature_separator_penalty += 1.0
                logger.debug(f"⚠️ Premature separator in line: {line[:50]}")
        
        # premature_separator_penalty가 있으면 매우 강한 페널티
        base_score_multiplier = 1.0
        if premature_separator_penalty > 0:
            base_score_multiplier = 0.01  # 99% 감점

        gen_commands = [c.strip() for c in completion.split(';') if c.strip()]
        if not gen_commands:
            return 0.0

        per_cmd_scores = []
        severe_violation = False
        number_only_count = 0
        invalid_command_count = 0  # 잘못된 명령어 개수 추적

        for cmd in gen_commands:
            score = 1.0
            parsed = self._parse(cmd)

            if parsed.get("number_error"):
                score = 0.0
                severe_violation = True
            else:
                if parsed.get("is_number_only"):
                    number_only_count += 1
                if parsed.get("missing_command"):
                    score = 0.0
                    severe_violation = True

                # invalid command - 매우 강력한 페널티
                if parsed.get("invalid_command"):
                    score *= 0.001  # 0.01 -> 0.001로 더 강화 (probe 같은 잘못된 명령어)
                    severe_violation = True
                    invalid_command_count += 1
                    logger.debug(f"⚠️ Invalid command in: {cmd[:50]}")

                # surface requirement
                if parsed.get("need_surface_missing"):
                    score *= 0.05

                # status position missing
                if parsed.get("status_missing_positions"):
                    score *= 0.05

                # value requirement
                if parsed.get("need_value_missing"):
                    score *= 0.05
                if parsed.get("bad_value_format"):
                    score *= 0.1

            per_cmd_scores.append(score)

        if not per_cmd_scores:
            return 0.0

        avg_score = sum(per_cmd_scores) / len(per_cmd_scores)
        
        # 명령어 완성 이전에 세미콜론/쉼표를 찍는 경우 penalty 적용
        avg_score *= base_score_multiplier

        # number-only spam
        if number_only_count > 0:
            ratio = number_only_count / max(1, len(gen_commands))
            avg_score *= max(0.02, 1.0 - ratio * 2.0)

        # 잘못된 명령어(probe 등)가 있으면 매우 강한 추가 penalty
        if invalid_command_count > 0:
            invalid_ratio = invalid_command_count / max(1, len(gen_commands))
            avg_score *= (1.0 - invalid_ratio * 0.9)  # 잘못된 명령어 비율만큼 강하게 감점
            logger.debug(f"⚠️ {invalid_command_count} invalid commands detected, applying strong penalty")

        # severe violations crush the score
        if severe_violation:
            avg_score *= 0.02

        # meta/template leftovers
        if "the following output" in completion or "based on the provided instructions" in completion:
            avg_score *= 0.05

        return max(0.0, min(1.0, avg_score))

    def _parse(self, cmd: str) -> Dict[str, Any]:
        result = {
            "number_error": False,
            "missing_command": False,
            "invalid_command": False,
            "need_surface_missing": False,
            "status_missing_positions": False,
            "need_value_missing": False,
            "bad_value_format": False,
            "is_number_only": False,
        }

        m_num = re.match(r'number\s+(\d+(?:\s*,\s*\d+)*)\s*(.*)$', cmd)
        if not m_num:
            result["number_error"] = True
            return result

        tail = m_num.group(2).strip()
        if not tail:
            result["is_number_only"] = True
            result["missing_command"] = True
            return result

        parts = [p.strip() for p in tail.split(',') if p.strip()]
        if not parts:
            result["missing_command"] = True
            return result

        command = None
        surface = None
        positions = []
        values = []

        for p in parts:
            pl = p.lower()
            if pl in self.surface_keywords:
                surface = pl
                continue
            if pl in self.position_keywords:
                positions.append(pl)
                continue
            if re.match(r'^\d+(\s+\d+){2,}$', pl):
                values.append(pl)
                continue
            if command is None:
                command = pl
            else:
                if re.match(r'^\d+(\s+\d+){0,2}$', pl):
                    values.append(pl)
                else:
                    # 추가 토큰은 형식 평가에서 큰 가중치 부여하지 않음
                    pass

        if command is None:
            result["missing_command"] = True
            return result

        # command validity - 강화된 체크
        is_valid_command = False
        # 정확한 매칭
        if command in self.valid_commands:
            is_valid_command = True
        else:
            # 부분 매칭 체크 (하지만 probe 같은 잘못된 명령어는 제외)
            for valid_cmd in self.valid_commands:
                if command in valid_cmd or valid_cmd in command:
                    is_valid_command = True
                    break
        
        # probe 같은 명확히 잘못된 명령어는 강력한 penalty
        invalid_command_keywords = ['probe', 'tooth', 'teeth']  # 명령어 셋에 없는 명령어
        if any(keyword in command for keyword in invalid_command_keywords):
            result["invalid_command"] = True
            logger.debug(f"⚠️ Invalid command keyword detected: {command}")
        elif not is_valid_command:
            result["invalid_command"] = True
            logger.debug(f"⚠️ Invalid command detected: {command} (not in valid commands)")

        cmd_info = self.loader.get_command_info(command)

        # surface requirement
        if cmd_info.get("need_surface") or cmd_info.get("is_status"):
            if surface is None:
                result["need_surface_missing"] = True

        # status position check
        if cmd_info.get("is_status"):
            if not positions:
                result["status_missing_positions"] = True

        # value requirement
        if command in self.value_required_commands:
            if not values:
                result["need_value_missing"] = True
            else:
                for v in values:
                    if not re.match(r'^\d+\s+\d+\s+\d+$', v):
                        result["bad_value_format"] = True
                        break

        return result


class RepetitionPenaltyComponent(RewardComponent):
    """
    반복/스팸 패턴 페널티 (number만 반복, 동일 명령어 과다 반복, meta text 등)
    가중치: 0.10
    """
    def __init__(self, weight: float = 0.10):
        super().__init__(name="repetition_penalty", weight=weight)

    def calculate(self, completion: str, ground_truth: str = None, **kwargs) -> float:
        completion = str(completion).lower()
        if not completion.strip():
            return 0.0
        
        gen_commands = [c.strip() for c in completion.split(';') if c.strip()]
        if not gen_commands:
            return 0.0
        
        # 1) 동일 명령어 반복 비율
        normalized = [self._normalize(cmd) for cmd in gen_commands]
        unique_norm = set(normalized)
        unique_ratio = len(unique_norm) / len(normalized) if normalized else 0.0
        
        score = 1.0
        if unique_ratio < 0.5:
            score *= 0.1  # 절반 이상이 중복이면 더 강한 감점
        elif unique_ratio < 0.7:
            score *= 0.5  # 다수 중복이면 감점
        
        # 2) number-only 패턴(행위 없는 숫자 나열) 감점
        number_only_count = sum(1 for cmd in gen_commands if re.fullmatch(r'number\s+\d+\s*', cmd))
        if number_only_count > 0:
            ratio = number_only_count / max(1, len(gen_commands))
            score *= max(0.02, 1.0 - ratio * 2.0)  # number만 반복되면 사실상 0점
        
        # 3) 같은 number를 여러 번 중복 생성 감점
        numbers = []
        for cmd in gen_commands:
            m = re.search(r'number\s+(\d+)', cmd)
            if m:
                numbers.append(m.group(1))
        if numbers:
            from collections import Counter
            counts = Counter(numbers)
            over_duplicates = [n for n, cnt in counts.items() if cnt > 1]
            if over_duplicates:
                dup_ratio = sum(counts[n]-1 for n in over_duplicates) / len(numbers)
                score *= max(0.1, 1.0 - dup_ratio * 1.0)
        
        # 4) meta text / template 남용 시 강한 감점
        meta_patterns = [
            r'\bthe following output\b',
            r'\bbased on the provided instructions\b',
            r'\bcommand\b\s*\.?$',
            r'\boutput\b\s*\.?$'
        ]
        if any(re.search(p, completion) for p in meta_patterns):
            score *= 0.02
        
        return max(0.0, min(1.0, score))
    
    def _normalize(self, cmd: str) -> str:
        # 숫자를 토큰으로만 유지하고 나머지 공백/소문자 정규화
        cmd = re.sub(r'\s+', ' ', cmd.strip().lower())
        # 값 시퀀스는 축약
        cmd = re.sub(r'\d+\s+\d+\s+\d+', '<val>', cmd)
        return cmd


class CommandRewardFunction:
    """치과 명령어 GRPO Reward"""
    
    __name__ = "CommandRewardFunction"
    
    def __init__(self, config: Dict = None):
        # config에서 csv_file_path와 data_csv_path 가져오기
        csv_file_path = None
        data_csv_path = None
        if config:
            csv_file_path = config.get('csv_file_path')
            data_csv_path = config.get('data_csv_path')
        
        self.components = [
            SequenceSimilarityComponent(weight=0.08),   # GT 시퀀스 유사도
            ToothNumberComponent(weight=0.12),          # 치아 번호 정확도
            OrderPenaltyComponent(weight=0.07),         # 명령어 순서 오류 페널티
            CommandKeywordComponent(weight=0.15, csv_file_path=csv_file_path, data_csv_path=data_csv_path),  # 명령어 키워드 매칭
            StatusExpansionComponent(weight=0.14, csv_file_path=csv_file_path, data_csv_path=data_csv_path),  # STATUS 확장 검증
            CommandFormatComponent(weight=0.16, csv_file_path=csv_file_path, data_csv_path=data_csv_path),  # 명령어 형식 검증
            InstructionComplianceComponent(weight=0.10, csv_file_path=csv_file_path, data_csv_path=data_csv_path),  # 지시 준수/환각
            RepetitionPenaltyComponent(weight=0.10),    # 반복/스팸 패턴 페널티
            SelfAuditComponent(weight=0.18, csv_file_path=csv_file_path, data_csv_path=data_csv_path),  # Self-reward 감사
            NumericalValueComponent(weight=0.08, csv_file_path=csv_file_path, data_csv_path=data_csv_path),  # 수치값 정확도
            StructuralComponent(weight=0.02),           # 구조적 정확도
            ExactMatchComponent(weight=0.02),           # 정확한 매칭 보너스
        ]
        self.config = config or {}
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """통합 reward 계산: 모든 component의 가중 평균"""
        # completions가 리스트가 아닌 경우 리스트로 변환
        if not isinstance(completions, list):
            completions = [completions]
        
        # ground_truth 처리
        ground_truths = kwargs.get("ground_truth", None)
        if ground_truths is not None:
            if not isinstance(ground_truths, list):
                ground_truths = [ground_truths] * len(completions)
            elif len(ground_truths) != len(completions):
                if len(ground_truths) > 0:
                    ground_truths = ground_truths[:len(completions)] + [ground_truths[0]] * (len(completions) - len(ground_truths))
                else:
                    ground_truths = [None] * len(completions)
        else:
            ground_truths = [None] * len(completions)
        
        rewards = []
        for i, completion in enumerate(completions):
            # completion 처리
            if isinstance(completion, list):
                if len(completion) > 0 and isinstance(completion[0], dict):
                    completion_text = completion[0].get("content", "")
                else:
                    completion_text = " ".join(str(item) for item in completion)
            elif isinstance(completion, dict):
                completion_text = completion.get("content", "")
            else:
                completion_text = str(completion)
            
            # 각 component의 reward 계산
            component_kwargs = kwargs.copy()
            component_kwargs['ground_truth'] = ground_truths[i] if i < len(ground_truths) else None
            
            total_reward = 0.0
            total_weight = 0.0
            
            for component in self.components:
                try:
                    component_reward = component.calculate(completion_text, **component_kwargs)
                    total_reward += component_reward * component.weight
                    total_weight += component.weight
                except Exception as e:
                    logger.warning(f"Error calculating reward for component {component.name}: {e}")
            
            # 정규화
            if total_weight > 0:
                total_reward /= total_weight
            
            rewards.append(max(0.0, min(1.0, total_reward)))
        
        return rewards
    
    def reward_func(self, completions, **kwargs) -> List[float]:
        """reward_func 별칭"""
        return self.__call__(completions, **kwargs)
    
    def expand_to_individual_rewards(self) -> List[ComponentRewardWrapper]:
        """CommandRewardFunction을 개별 component reward function들로 확장"""
        individual_rewards = []
        for component in self.components:
            wrapper = ComponentRewardWrapper(
                component=component,
                component_name=f"CommandReward_{component.name}",
                weight=component.weight
            )
            individual_rewards.append(wrapper)
        return individual_rewards
