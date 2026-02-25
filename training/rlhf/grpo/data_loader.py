"""
TRL 표준 데이터 로더 for GRPO training
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoProcessor
import trl.trainer
import pandas as pd
import os
import re

logger = logging.getLogger(__name__)

class GRPODataLoader:
    """TRL 표준 데이터 로더 for GRPO training"""

    def __init__(
        self,
        model_name: str = "unsloth/Qwen3-0.6B-bnb-4bit",
        max_length: int = 2048,
        data_mode: str = "instruction",
        csv_file_path: Optional[str] = None  # 명령어 정의 CSV 경로
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.data_mode = data_mode
        
        # CSV 파일 경로 설정
        if csv_file_path is None:
            possible_paths = [
                "cmd_bot.csv",
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cmd_bot.csv"),
                os.path.join(os.getcwd(), "cmd_bot.csv")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    csv_file_path = path
                    break
        
        self.csv_file_path = csv_file_path
        self.command_patterns = {}
        self.available_commands = []
        self.all_commands_info = {}  # 전체 명령어 정보 (명령어 맵)
        
        # 명령어 정보 로드
        if self.csv_file_path and os.path.exists(self.csv_file_path):
            self._load_commands_from_csv()
        else:
            logger.warning(f"⚠️ CSV 파일을 찾을 수 없습니다: {csv_file_path}. 명령어 정보 없이 진행합니다.")

        # Load tokenizer only (TRL handles the rest)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"✅ TRL DataLoader initialized with model: {model_name}")
        if self.command_patterns:
            logger.info(f"📋 {len(self.available_commands)}개 명령어 로드 완료")
    
    def _load_commands_from_csv(self):
        """CSV 파일에서 명령어 정보 로드 (전체 명령어 맵)"""
        try:
            df = pd.read_csv(self.csv_file_path)
            
            self.available_commands = sorted(df['cmd'].unique().tolist())
            
            # 카테고리별로 명령어 그룹화
            self.command_patterns = {}
            for _, row in df.iterrows():
                cmd = row['cmd']
                category = row.get('category', 'other')
                need_num = row.get('need_num', False)
                need_surface = row.get('need_surface', False)
                need_bridge = row.get('need_bridge', False)
                desc = row.get('desc', '')
                is_status = row.get('is_status', False)
                is_control = row.get('is_control', False)
                
                if category not in self.command_patterns:
                    self.command_patterns[category] = []
                
                cmd_info = {
                    'command': cmd,
                    'need_num': need_num,
                    'need_surface': need_surface,
                    'need_bridge': need_bridge,
                    'description': desc,
                    'is_status': is_status,
                    'is_control': is_control,
                }
                
                self.command_patterns[category].append(cmd_info)
                # 전체 명령어 맵에 저장
                self.all_commands_info[cmd] = cmd_info
            
            logger.info(f"✅ {len(self.available_commands)}개 명령어를 CSV에서 로드 완료")
            logger.info(f"📊 카테고리별 명령어 수: {dict((k, len(v)) for k, v in self.command_patterns.items())}")
            
        except Exception as e:
            logger.error(f"❌ CSV 로드 실패: {e}")
            self.command_patterns = {}
            self.available_commands = []
            self.all_commands_info = {}

    def load_dataset(
        self,
        dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
        split: str = "train_prefs",
        max_samples: Optional[int] = None,
        streaming: bool = False
    ) -> Dataset:
        """
        Load dataset from HuggingFace Hub

        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            split: Dataset split to load
            max_samples: Maximum number of samples to load
            streaming: Whether to use streaming mode

        Returns:
            Dataset: Loaded dataset (not DatasetDict)
        """
        logger.info(f"📦 Loading dataset: {dataset_name} (split: {split})")

        try:
            if streaming:
                dataset = load_dataset(dataset_name, split=split, streaming=True)
                if max_samples:
                    dataset = dataset.take(max_samples)
                return dataset
            else:
                dataset = load_dataset(dataset_name, split=split)
                if max_samples:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                return dataset

        except Exception as e:
            logger.error(f"❌ Failed to load dataset {dataset_name}: {e}")
            raise
    
    def load_custom_dataset(
        self,
        data_path: str,
        split: str = "train"
    ) -> Dataset:
        """
        Load custom dataset from local files

        Args:
            data_path: Path to the dataset file (JSON, JSONL, CSV, etc.)
            split: Dataset split to load (default: "train")

        Returns:
            Dataset: Loaded dataset from specified split
        """
        logger.info(f"📁 Loading custom dataset from: {data_path} (split: {split})")

        try:
            if data_path.endswith('.jsonl'):
                dataset_dict = load_dataset('json', data_files=data_path)
            elif data_path.endswith('.json'):
                dataset_dict = load_dataset('json', data_files=data_path)
            elif data_path.endswith('.csv'):
                dataset_dict = load_dataset('csv', data_files=data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")

            # Get the specified split (default to first available split if specified split doesn't exist)
            if split in dataset_dict:
                dataset = dataset_dict[split]
            else:
                # Fallback to first available split
                available_splits = list(dataset_dict.keys())
                if available_splits:
                    dataset = dataset_dict[available_splits[0]]
                    logger.warning(f"⚠️ Split '{split}' not found, using '{available_splits[0]}' instead")
                else:
                    raise ValueError(f"No splits available in dataset: {data_path}")

            return dataset

        except Exception as e:
            logger.error(f"❌ Failed to load custom dataset: {e}")
            raise
    
    def _extract_used_commands(self, ground_truth: str) -> Set[str]:
        """ground_truth에서 실제 사용된 명령어 토큰만 추출"""
        if not ground_truth:
            return set()
        
        used_commands = set()
        
        # 세미콜론으로 명령어 분리
        commands = ground_truth.split(';')
        
        for cmd in commands:
            # 쉼표로 분리하여 토큰 추출
            parts = [p.strip().lower() for p in cmd.split(',')]
            
            # CSV의 명령어 목록과 정확히 매칭
            for part in parts:
                # 정확한 명령어 매칭
                for cmd_name in self.available_commands:
                    if cmd_name.lower() == part:
                        used_commands.add(cmd_name)
                        break
                
                # "pocket depth" 같은 복합 명령어 처리
                if 'pocket depth' in part:
                    if 'pocket depth' in self.available_commands:
                        used_commands.add('pocket depth')
                    if 'probing' in self.available_commands:
                        used_commands.add('probing')
                elif 'probing' in part and 'probing' in self.available_commands:
                    used_commands.add('probing')
        
        return used_commands

    def _format_available_commands_map(self) -> str:
        """전체 명령어 맵 포맷팅 (평가용)"""
        if not self.command_patterns:
            return "명령어 정보를 로드할 수 없습니다."
        
        formatted = []
        formatted.append("AVAILABLE COMMANDS MAP:")
        formatted.append("(You MUST select commands ONLY from this list)")
        formatted.append("")
        
        for category, commands in sorted(self.command_patterns.items()):
            formatted.append(f"=== {category.upper()} ===")
            for cmd_info in sorted(commands, key=lambda x: x['command']):
                cmd = cmd_info['command']
                requirements = []
                if cmd_info.get('need_num'):
                    requirements.append("needs tooth number")
                if cmd_info.get('need_surface'):
                    requirements.append("needs surface")
                if cmd_info.get('need_bridge'):
                    requirements.append("needs bridge")
                
                req_text = f" [{', '.join(requirements)}]" if requirements else ""
                formatted.append(f"- {cmd}{req_text}")
            
            formatted.append("")
        
        return "\n".join(formatted)

    def _analyze_ground_truth_patterns(self, ground_truth: str) -> Dict[str, Any]:
        """ground_truth의 실제 사용 패턴 분석"""
        if not ground_truth:
            return {}
        
        patterns = {
            'commands': set(),
            'has_surface': False,
            'has_positions': False,
            'status_expansion': None,  # 'full', 'mesial_distal', 'single'
            'probing_format': None,  # 'with_surface', 'no_surface'
            'value_format': None,  # 'with_surface', 'no_surface'
        }
        
        # 세미콜론으로 명령어 분리
        commands = ground_truth.split(';')
        
        for cmd in commands:
            cmd_lower = cmd.lower()
            parts = [p.strip() for p in cmd.split(',')]
            
            # 명령어 추출
            for part in parts:
                part_lower = part.strip().lower()
                for cmd_name in self.available_commands:
                    if cmd_name.lower() == part_lower:
                        patterns['commands'].add(cmd_name)
                        break
            
            # Surface 패턴 분석
            if any(s in cmd_lower for s in ['buccal', 'lingual', 'palatal']):
                patterns['has_surface'] = True
            
            # Position 패턴 분석 (mesial, middle, distal)
            has_mesial = 'mesial' in cmd_lower
            has_middle = 'middle' in cmd_lower
            has_distal = 'distal' in cmd_lower
            
            if has_mesial or has_middle or has_distal:
                patterns['has_positions'] = True
                
                # Status 명령어 expansion 패턴 분석
                if any(status in cmd_lower for status in ['bleeding', 'suppuration', 'plaque', 'calculus']):
                    if has_mesial and has_middle and has_distal:
                        patterns['status_expansion'] = 'full'
                    elif has_mesial and has_distal and not has_middle:
                        patterns['status_expansion'] = 'mesial_distal'
                    else:
                        patterns['status_expansion'] = 'single'
            
            # Probing 패턴 분석
            if 'probing' in cmd_lower or 'pocket depth' in cmd_lower:
                if patterns['has_surface']:
                    patterns['probing_format'] = 'with_surface'
                else:
                    patterns['probing_format'] = 'no_surface'
            
            # Value 패턴 분석 (furcation, mobility, recession)
            if any(value in cmd_lower for value in ['furcation', 'mobility', 'recession', 'gingival margin']):
                if patterns['has_surface']:
                    patterns['value_format'] = 'with_surface'
                else:
                    patterns['value_format'] = 'no_surface'
        
        return patterns

    def _build_minimal_rules(self, patterns: Dict[str, Any]) -> str:
        """패턴 분석 결과에 따른 최소한의 규칙 생성"""
        if not patterns or not patterns.get('commands'):
            return ""
        
        rules = []
        
        # Probing 규칙 (실제 사용 패턴 기반)
        probing_cmds = [c for c in patterns['commands'] if c.lower() in ['probing', 'pocket depth']]
        if probing_cmds:
            rule = f"PROBING ({', '.join(probing_cmds)}): number N, "
            if patterns.get('probing_format') == 'with_surface':
                rule += "[surface, ] probing, X Y Z"
            else:
                rule += "probing, X Y Z"
            rule += " (three numbers, no positions)"
            rules.append(rule)
        
        # Status 규칙 (실제 expansion 패턴 기반)
        status_cmds = [c for c in patterns['commands'] 
                    if any(s in c.lower() for s in ['bleeding', 'suppuration', 'plaque', 'calculus'])]
        if status_cmds:
            expansion = patterns.get('status_expansion')
            if expansion == 'full':
                rule = f"STATUS ({', '.join(status_cmds)}): number N, [surface, ] mesial, [cmd], 1, middle, [cmd], 1, distal, [cmd], 1"
            elif expansion == 'mesial_distal':
                rule = f"STATUS ({', '.join(status_cmds)}): number N, [surface, ] mesial, [cmd], 1, distal, [cmd], 1"
            else:
                rule = f"STATUS ({', '.join(status_cmds)}): number N, [surface, ] [cmd], 1"
            
            # Surface 패턴
            if patterns.get('has_surface'):
                rule += " (surface specified)"
            else:
                rule += " (both buccal and lingual if no surface)"
            
            rules.append(rule)
        
        # Value 규칙 (실제 사용 패턴 기반)
        value_cmds = [c for c in patterns['commands'] 
                    if any(v in c.lower() for v in ['furcation', 'mobility', 'recession', 'gingival margin'])]
        if value_cmds:
            rule = f"VALUE ({', '.join(value_cmds)}): number N, "
            if patterns.get('value_format') == 'with_surface':
                rule += "[surface, ] [cmd], value"
            else:
                rule += "[cmd], value"
            rule += " (no positions)"
            rules.append(rule)
        
        # Restoration 규칙
        restoration_cmds = [c for c in patterns['commands'] 
                        if any(r in c.lower() for r in ['crown', 'implant', 'fixture', 'bridge'])]
        if restoration_cmds:
            rules.append(f"RESTORATION ({', '.join(restoration_cmds)}): number N, [number M, ] [cmd]")
        
        # Control 규칙
        control_cmds = [c for c in patterns['commands'] 
                    if any(ctrl in c.lower() for ctrl in ['jump', 'back', 'clear', 'delete'])]
        if control_cmds:
            rules.append(f"CONTROL ({', '.join(control_cmds)}): [cmd] or [cmd] to number N")
        
        return "\n".join(rules) if rules else ""

    def _analyze_transformation_logic(self, utterance: str, ground_truth: str) -> List[str]:
        """입력과 정답을 비교하여 필요한 변환 로직 규칙 추출"""
        rules = []
        utterance_lower = utterance.lower()
        gt_lower = ground_truth.lower()

        # 1. Default Surface Logic (가장 중요)
        # 입력에는 surface가 없는데 정답에는 있는 경우 -> 기본값 규칙 필요
        status_keywords = ['bleeding', 'suppuration', 'plaque', 'calculus']
        has_status_cmd = any(k in utterance_lower for k in status_keywords)
        
        if has_status_cmd:
            input_surface = any(s in utterance_lower for s in ['buccal', 'lingual', 'palatal', 'facial', 'labial'])
            output_has_buccal = 'buccal' in gt_lower
            output_has_lingual = 'lingual' in gt_lower
            
            if not input_surface:
                if output_has_buccal and output_has_lingual:
                    rules.append("LOGIC: No surface mentioned for STATUS → Output BOTH buccal AND lingual commands.")
                elif output_has_buccal:
                    rules.append("LOGIC: No surface mentioned → Default to 'buccal' only.")

        # 2. Probing Logic (값 처리)
        if 'probing' in gt_lower or 'pocket depth' in gt_lower:
            # 입력이 '323' 처럼 붙어있거나 '3, 2, 3' 처럼 떨어져 있어도 정답은 '3 2 3'
            rules.append("LOGIC: Input numbers (e.g., '3 2 3') → Format as 'probing, X Y Z' (three distinct numbers).")

        # 3. Range/Sequence Logic (입력 키워드 기반)
        if any(k in utterance_lower for k in ['to', 'through', '-', 'all']):
            # 정답에 number가 여러 개 등장하면 Range 확장임
            if gt_lower.count('number') > 3:
                rules.append("LOGIC: Range/All detected ('to', 'through', 'all') → Expand to explicit command for EACH tooth in range.")

        # 4. Repeat Logic
        if 'repeat' in utterance_lower:
            rules.append("LOGIC: 'repeat' detected → Apply previous command values to subsequent teeth explicitly.")

        # 5. Exception Logic
        if 'except' in utterance_lower or 'but' in utterance_lower:
            rules.append("LOGIC: 'except/but' detected → Apply general rule first, then override specific teeth.")
            
        # 6. Position Logic (Proximal)
        if 'proximal' in utterance_lower:
            rules.append("LOGIC: 'proximal' detected → Expand to 'mesial' AND 'distal' (skip middle).")

        return rules
    
    def _build_adaptive_cmd_prompt(
        self,
        utterance: str,
        numbering_system: str,
        ground_truth: Optional[str] = None,
        numbering_method: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """답지 기반 적응형 프롬프트 생성 (system prompt + user prompt 분리)"""
        
        # Numbering system 정보
        if numbering_system == "FDI":
            quadrant_mapping = "Q1 → teeth 11–18, Q2 → 21–28, Q3 → 31–38, Q4 → 41–48"
            numbering_info = "[FDI] Q1(11-18), Q2(21-28), Q3(31-38), Q4(41-48)"
        else:  # UNS
            quadrant_mapping = "Q1 → teeth 1–8, Q2 → 9–16, Q3 → 17–24, Q4 → 25–32"
            numbering_info = "[UNS] Q1(1-8), Q2(9-16), Q3(17-24), Q4(25-32)"
        
        if numbering_method:
            numbering_info = numbering_method
        
        # System prompt 구성
        system_prompt = f"""🦷 PERIODONTAL CHARTING ASSISTANT

TASK: Convert natural language into structured command sequences.

CRITICAL: Use ONLY commands from AVAILABLE COMMANDS MAP below.

TOOTH NUMBERING: {numbering_system}
{numbering_info}
Quadrant: {quadrant_mapping}

"""
        
        # 전체 명령어 맵 (평가용)
        if self.available_commands:
            system_prompt += self._format_available_commands_map() + "\n\n"
        
        # ground_truth 패턴 분석 및 규칙 생성
        patterns = self._analyze_ground_truth_patterns(ground_truth) if ground_truth else {}
        format_rules = self._build_minimal_rules(patterns)
        logic_rules = self._analyze_transformation_logic(utterance, ground_truth) if ground_truth else []
        
        # 1. FORMAT RULES (Output Structure)
        if format_rules:
            system_prompt += "FORMAT RULES (Output Structure):\n"
            system_prompt += format_rules + "\n\n"
        
        # 2. TRANSFORMATION LOGIC (How to process Input)
        if logic_rules:
            system_prompt += "TRANSFORMATION LOGIC (How to process Input):\n"
            system_prompt += "\n".join([f"- {r}" for r in logic_rules]) + "\n\n"
        
        # 공통 규칙 (최소화)
        common_rules = """COMMON RULES:
- Single line output, semicolons (;) separate commands
- Always start with "number N"
- Three numbers = probing values (NOT tooth number)
- Never output meta-commands: expand "repeat", "others", "all" to explicit commands
- VALIDATION: Check that all commands in your output exist in AVAILABLE COMMANDS MAP above

"""
        
        system_prompt += common_rules
        
        # User prompt 구성
        user_prompt = f"Convert: {utterance}\n\nOutput (commands only):"
        
        # Messages 형식으로 반환 (chat template 사용)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def prepare_grpo_data(
        self,
        dataset
    ) -> Dataset:
        """
        TRL 표준 데이터 형식으로 변환

        TRL GRPO는 다음 형식의 데이터를 기대합니다:
        - prompt/chosen/rejected 필드
        또는
        - messages 필드 (대화 형식)

        Args:
            dataset: Dataset 또는 DatasetDict 객체

        Returns:
            Dataset: TRL 형식으로 변환된 데이터셋
        """
        logger.info("🔄 Converting to TRL standard format")

        # DatasetDict인 경우 train split 사용
        if isinstance(dataset, DatasetDict):
            if "train" in dataset:
                dataset = dataset["train"]
            else:
                # 첫 번째 사용 가능한 split 사용
                available_splits = list(dataset.keys())
                if available_splits:
                    dataset = dataset[available_splits[0]]
                    logger.warning(f"⚠️ Using split '{available_splits[0]}' from DatasetDict")
                else:
                    raise ValueError("No splits available in DatasetDict")

        if not isinstance(dataset, Dataset):
            raise ValueError(f"Expected Dataset, got {type(dataset)}")

        def convert_to_trl_format(example):
            """Convert to TRL standard format"""
            # 이미 TRL 형식이면 그대로 반환
            if "messages" in example:
                # messages 는 (prompt, chosen, rejected) 조합에서는 사용되지 않음
                del example["messages"]

            if "prompt" in example:
                if not all([prompt for prompt in example.get("prompt") if type(prompt) == str and type(prompt) == list]):
                    example["prompt"] = [{"role": "user", "content": prompt} for prompt in example.get("prompt")]

            if "prompt" in example and not ("chosen" in example and "rejected" in example):
                if self.data_mode == "cmd":
                    # 적응형 프롬프트 생성 (system prompt + user prompt 분리)
                    numbering_system = example.get('numbering_system', 'UNS')
                    numbering_method = example.get('numbering_method', None)
                    ground_truth = example.get('ground_truth') or example.get('label')
                    utterance = example['prompt']  # 원본 utterance
                    
                    # Messages 형식으로 프롬프트 생성
                    messages = self._build_adaptive_cmd_prompt(
                        utterance=utterance,
                        numbering_system=numbering_system,
                        ground_truth=ground_truth,
                        numbering_method=numbering_method
                    )
                    
                    # TRL은 messages 형식 또는 문자열 형식 모두 지원
                    # messages 형식으로 저장 (chat template 사용)
                    example["prompt"] = messages
                return {"prompt": example["prompt"]}

            # UltraFeedback 형식 변환
            if "chosen" in example and "rejected" in example:
                chosen = example["chosen"]
                rejected = example["rejected"]

                if isinstance(chosen, list) and isinstance(rejected, list):
                    # chosen과 rejected가 리스트인 경우 (메시지 형식)
                    chosen_text = chosen[-1]["content"] if chosen else ""
                    rejected_text = rejected[-1]["content"] if rejected else ""
                    prompt = chosen[0]["content"] if chosen else ""

                    return {
                        "prompt": prompt,
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    }

            # 기본적으로 원본 반환 (TRL이 처리)
            return example

        # 데이터 변환
        processed_dataset = dataset.map(
            convert_to_trl_format,
            desc="Converting to TRL format"
        )

        logger.info(f"✅ Converted {len(processed_dataset)} samples to TRL format")
        return processed_dataset
    
    def get_sample_data(
        self,
        dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    ) -> Dict[str, Any]:
        """
        샘플 데이터를 가져와서 TRL 형식 확인

        Args:
            dataset_name: 확인할 데이터셋 이름
        """
        logger.info(f"🔍 Getting sample data from {dataset_name}")

        try:
            # 작은 샘플 로드
            dataset = self.load_dataset(dataset_name, max_samples=5)

            # 첫 번째 샘플 반환
            if len(dataset) > 0:
                sample = dict(dataset[0])
                logger.info("✅ Sample data retrieved successfully")
                logger.info(f"📋 Sample keys: {list(sample.keys())}")
                return sample
            else:
                logger.warning("⚠️ No samples found in dataset")
                return {}

        except Exception as e:
            logger.error(f"❌ Failed to get sample data: {e}")
            return {}


def create_grpo_dataloader(
    model_name: str = "unsloth/Qwen3-0.6B-bnb-4bit",
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
    max_samples: int = 1000,
    max_length: int = 2048,
    split: str = "train_prefs"
) -> tuple[GRPODataLoader, Dataset]:
    """
    TRL 표준 데이터 로더 생성 및 데이터셋 로드

    Args:
        model_name: 모델 이름
        dataset_name: 데이터셋 이름
        max_samples: 최대 샘플 수
        max_length: 최대 시퀀스 길이
        split: 사용할 데이터셋 분할

    Returns:
        (data_loader, dataset) 튜플
    """
    # 데이터 로더 생성
    data_loader = GRPODataLoader(
        model_name=model_name,
        max_length=max_length,
        data_mode="cmd"  # Default to cmd mode for create_grpo_dataloader if intended for cmd training
    )

    # 데이터셋 로드 및 TRL 형식으로 변환
    dataset = data_loader.load_dataset(dataset_name, split=split, max_samples=max_samples)
    processed_dataset = data_loader.prepare_grpo_data(dataset)

    return data_loader, processed_dataset
