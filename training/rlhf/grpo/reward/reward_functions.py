"""
통합 커스텀 Reward Functions for TRL GRPO

단일 또는 다중 보상 함수를 지원하는 통합 보상 시스템입니다.
"""
from functools import wraps
import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RewardComponent(ABC):
    """보상 계산의 개별 컴포넌트"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def calculate(self, completion: str, **kwargs) -> float:
        """개별 보상 계산"""
        pass


class AccuracyComponent(RewardComponent):
    """정확성 보상 컴포넌트"""

    def __init__(self, config: Dict[str, Any] = None, weight: float = 1.0):
        super().__init__("accuracy", weight)
        self.correct_keywords = config.get("correct_keywords", ["correct", "right", "yes"]) if config else ["correct", "right", "yes"]

    def calculate(self, completion: str, **kwargs) -> float:
        solutions = kwargs.get("solutions", [])
        completion_lower = completion.lower()

        # 정확한 키워드가 포함되어 있는지 확인
        if any(keyword in completion_lower for keyword in self.correct_keywords):
            return 1.0
        elif solutions:
            # 정답과 유사한지 확인
            if any(sol.lower() in completion_lower or completion_lower in sol.lower()
                  for sol in solutions):
                return 0.8

        return 0.0


class LengthComponent(RewardComponent):
    """길이 보상 컴포넌트"""

    def __init__(self, config: Dict[str, Any] = None, weight: float = 1.0):
        super().__init__("length", weight)
        self.optimal_length = config.get("optimal_length", 100) if config else 100
        self.length_weight = config.get("length_weight", 0.1) if config else 0.1

    def calculate(self, completion: str, **kwargs) -> float:
        length = len(completion)
        length_diff = abs(length - self.optimal_length)
        return max(0, self.length_weight * (self.optimal_length - length_diff))


class QualityComponent(RewardComponent):
    """품질 보상 컴포넌트"""

    def __init__(self, config: Dict[str, Any] = None, weight: float = 1.0):
        super().__init__("quality", weight)
        self.reward_scale = config.get("reward_scale", 1.0) if config else 1.0
        self.penalty_scale = config.get("penalty_scale", -1.0) if config else -1.0
        self.quality_keywords = config.get("quality_keywords", ["좋아", "완벽", "우수"]) if config else ["좋아", "완벽", "우수"]
        self.negative_keywords = config.get("negative_keywords", ["모르겠", "잘못", "틀렸"]) if config else ["모르겠", "잘못", "틀렸"]

    def calculate(self, completion: str, **kwargs) -> float:
        reward = 0.0
        completion_lower = completion.lower()

        # 충분한 길이 체크
        if len(completion.strip()) > 10:
            reward += self.reward_scale * 0.5

        # 문장 완성도 체크
        if not completion.strip().endswith('.'):
            reward += self.penalty_scale * 0.2

        # 품질 키워드 체크
        if any(keyword in completion for keyword in self.quality_keywords):
            reward += self.reward_scale * 0.3

        # 부정 키워드 체크
        if any(keyword in completion_lower for keyword in self.negative_keywords):
            reward += self.penalty_scale * 0.3

        return max(0, reward)

def _lcs(a: List[str], b: List[str]) -> int:
    """Computes the length of the Longest Common Subsequence."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


class RougeLComponent(RewardComponent):
    """ROUGE-L 기반 정확성 보상 컴포넌트"""

    def __init__(self, weight: float = 1.0):
        super().__init__("rouge_l_accuracy", weight)

    def calculate(self, completion: str, **kwargs) -> float:
        """
        ROUGE-L F1 점수를 계산합니다.
        kwargs에서 'ground_truth' 키를 사용하여 정답 문자열을 가져옵니다.
        """
        ground_truth = kwargs.get("ground_truth")
        if not ground_truth or not completion:
            return 0.0

        # 문자열을 토큰(단어) 리스트로 분리
        completion_tokens = completion.split()
        ground_truth_tokens = ground_truth.split()
        
        if not completion_tokens or not ground_truth_tokens:
            return 0.0

        lcs_length = _lcs(completion_tokens, ground_truth_tokens)

        # Precision, Recall, F1-score 계산
        precision = lcs_length / len(completion_tokens) if len(completion_tokens) > 0 else 0.0
        recall = lcs_length / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0.0
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
            
        return f1_score


class BaseRewardFunction:
    __name__ = "RewardFunction"

    @abstractmethod
    def reward_func(self, *args, **kwargs) -> List[float]:
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self, *args, **kwargs) -> List[float]:
        return self.reward_func(*args, **kwargs)

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__

class MultiRewardFunction(BaseRewardFunction):
    """
    다중 보상 함수를 지원하는 통합 클래스
    
    reward 처리
        completion 예시: [{role: "assistant", content: "어쩌고 저쩌고..."}]
        reward function에서는 위의 completion의 content를 처리하여 보상 부여.
    """

    def __init__(
        self,
        components: Optional[List[RewardComponent]] = None,
        config: Dict[str, Any] = None
    ):
        self.config = config or {}
        self.components = components or []

        # 기본 컴포넌트들 설정
        if not self.components:
            self.components = [
                RougeLComponent(weight=self.config.get("rouge_l_accuracy", {}).get("weight", 0.8)),
                LengthComponent(self.config.get("length", {})),
                QualityComponent(self.config.get("quality", {}))
            ]

    def reward_func(self, completions: List[str], **kwargs) -> List[float]:
        """TRL GRPOTrainer와 호환되는 다중 보상 계산"""
        rewards = []
        # 'ground_truth'가 kwargs에 리스트로 제공될 것으로 예상
        ground_truths = kwargs.get("ground_truth", [None] * len(completions))

        for i, completion in enumerate(completions):
            total_reward = 0.0
            
            component_kwargs = kwargs.copy()
            component_kwargs['ground_truth'] = ground_truths[i] if i < len(ground_truths) else None

            # 각 컴포넌트의 보상을 가중합으로 계산
            for component in self.components:
                component_reward = component.calculate(completion, **component_kwargs)
                total_reward += component_reward * component.weight

            # 정규화 (최대 보상이 1.0이 되도록)
            max_possible_reward = sum(comp.weight for comp in self.components)
            if max_possible_reward > 0:
                total_reward /= max_possible_reward

            rewards.append(max(0, total_reward))  # 음수 보상 방지

        return rewards

    def add_component(self, component: RewardComponent):
        """새로운 보상 컴포넌트 추가"""
        self.components.append(component)
        logger.info(f"✅ Added reward component: {component.name}")

    def remove_component(self, name: str):
        """보상 컴포넌트 제거"""
        self.components = [comp for comp in self.components if comp.name != name]
        logger.info(f"✅ Removed reward component: {name}")


class SingleCustomRewardFunction(BaseRewardFunction):
    """
    단일 커스텀 보상 함수
    
    reward 처리
        completions는 문자열 리스트로 가정합니다.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 통합된 보상 로직 설정
        self.accuracy_weight = self.config.get("accuracy_weight", 0.4)
        self.length_weight = self.config.get("length_weight", 0.2)
        self.quality_weight = self.config.get("quality_weight", 0.4)

        # 각 컴포넌트 설정
        self.accuracy_config = self.config.get("accuracy", {})
        self.length_config = self.config.get("length", {})
        self.quality_config = self.config.get("quality", {})
        
        # RougeL 컴포넌트 인스턴스화
        self.rouge_l_component = RougeLComponent()

    def reward_func(self, completions: List[str], **kwargs) -> List[float]:
        """단일 통합 보상 계산"""
        rewards = []
        ground_truths = kwargs.get("ground_truth", [None] * len(completions))

        for i, completion_content in enumerate(completions):
            reward = 0.0
            
            component_kwargs = kwargs.copy()
            component_kwargs['ground_truth'] = ground_truths[i] if i < len(ground_truths) else None

            # 정확성 보상 (ROUGE-L 사용)
            accuracy_reward = self.rouge_l_component.calculate(completion_content, **component_kwargs)
            reward += self.accuracy_weight * accuracy_reward

            # 길이 보상
            optimal_length = self.length_config.get("optimal_length", 100)
            length_weight = self.length_config.get("length_weight", 0.1) # 이 가중치는 SingleCustomRewardFunction의 self.length_weight와 다름
            length = len(completion_content)
            length_diff = abs(length - optimal_length)
            length_reward = max(0, length_weight * (optimal_length - length_diff))
            reward += self.length_weight * length_reward

            # 품질 보상
            quality_reward = 0.0
            if len(completion_content.strip()) > 10:
                quality_reward += 0.5

            if not completion_content.strip().endswith('.'):
                quality_reward -= 0.2
            
            reward += self.quality_weight * max(0, quality_reward)

            rewards.append(max(0, reward))

        return rewards
