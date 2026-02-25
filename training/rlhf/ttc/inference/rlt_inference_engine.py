import sys
import os
# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.teacher_model import TeacherModel
from models.reward_model import RewardModel
from config.config import ttc_config
import torch
import re
from typing import Dict, List, Tuple, Optional
import numpy as np

class RLTInferenceEngine:
    """
    RLT-style inference engine with budget forcing and test-time scaling.
    Implements the core concepts from Sakana's RLT paper.
    """
    
    def __init__(self, teacher_model: TeacherModel, reward_model: Optional[RewardModel] = None):
        self.teacher_model = teacher_model
        self.reward_model = reward_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def apply_budget_forcing(self, prompt: str, min_tokens: Optional[int] = None, max_tokens: Optional[int] = None) -> str:
        """
        Apply budget forcing to control test-time compute.
        This is inspired by the RLT paper's budget forcing technique.
        """
        if min_tokens is None:
            min_tokens = ttc_config.MIN_THINKING_TOKENS
        if max_tokens is None:
            max_tokens = ttc_config.MAX_THINKING_TOKENS
        
        # Add thinking start tag
        thinking_prompt = prompt + "\n<think>"
        
        # Generate with budget forcing
        inputs = self.teacher_model.tokenizer(thinking_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.teacher_model.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=ttc_config.TEMPERATURE,
                do_sample=True,
                pad_token_id=self.teacher_model.tokenizer.eos_token_id,
                eos_token_id=self.teacher_model.tokenizer.eos_token_id,
                # Custom stopping criteria for budget forcing
                stopping_criteria=self._create_budget_forcing_criteria(min_tokens, max_tokens)
            )
        
        generated_text = self.teacher_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def _create_budget_forcing_criteria(self, min_tokens: int, max_tokens: int):
        """Create custom stopping criteria for budget forcing."""
        from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
        
        class BudgetForcingCriteria(StoppingCriteria):
            def __init__(self, min_tokens: int, max_tokens: int, tokenizer, wait_token: str):
                self.min_tokens = min_tokens
                self.max_tokens = max_tokens
                self.tokenizer = tokenizer
                self.wait_token = wait_token
                self.wait_token_ids = tokenizer.encode(wait_token, add_special_tokens=False)
            
            def __call__(self, input_ids, scores, **kwargs):
                # Check if we've reached max tokens
                if input_ids.shape[1] >= self.max_tokens:
                    return True
                
                # Check if we've reached min tokens and model wants to stop
                if input_ids.shape[1] >= self.min_tokens:
                    # Check if the model is trying to end thinking
                    last_tokens = input_ids[0, -len(self.wait_token_ids):].tolist()
                    if last_tokens == self.wait_token_ids:
                        # Model wants to stop, but we force it to continue
                        return False
                
                return False
        
        criteria = BudgetForcingCriteria(
            min_tokens, 
            max_tokens, 
            self.teacher_model.tokenizer, 
            ttc_config.WAIT_TOKEN
        )
        return StoppingCriteriaList([criteria])
    
    def run_inference_with_scaling(self, question: str, scaling_factor: float = 1.0) -> Dict:
        """
        Run inference with test-time scaling.
        scaling_factor: 1.0 = normal, 2.0 = double compute, 0.5 = half compute
        """
        # Adjust token limits based on scaling factor
        base_min_tokens = ttc_config.MIN_THINKING_TOKENS
        base_max_tokens = ttc_config.MAX_THINKING_TOKENS
        
        scaled_min_tokens = int(base_min_tokens * scaling_factor)
        scaled_max_tokens = int(base_max_tokens * scaling_factor)
        
        # Generate with budget forcing
        formatted_prompt = self.teacher_model.format_reasoning_prompt(question)
        generated_text = self.apply_budget_forcing(
            formatted_prompt, 
            min_tokens=scaled_min_tokens, 
            max_tokens=scaled_max_tokens
        )
        
        # Extract thinking and solution
        thinking_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
        solution_match = re.search(r'<solution>(.*?)</solution>', generated_text, re.DOTALL)
        
        thinking = thinking_match.group(1).strip() if thinking_match else ""
        solution = solution_match.group(1).strip() if solution_match else ""
        
        # Calculate quality score if reward model is available
        quality_score = 0.0
        if self.reward_model:
            full_response = f"Question: {question}\n<think>{thinking}</think>\n<solution>{solution}</solution>"
            quality_score = self.reward_model.score([full_response])[0]
        else:
            quality_score = self.teacher_model.evaluate_reasoning_quality(thinking)
        
        return {
            "question": question,
            "reasoning_trace": thinking,
            "solution": solution,
            "quality_score": quality_score,
            "scaling_factor": scaling_factor,
            "tokens_used": len(self.teacher_model.tokenizer.encode(thinking)),
            "full_response": generated_text
        }
    
    def run_best_of_n_inference(self, question: str, n: int = 5, scaling_factor: float = 1.0) -> Dict:
        """Run best-of-N inference with test-time scaling."""
        results = []
        
        for i in range(n):
            # Vary temperature slightly for diversity
            temp = ttc_config.TEMPERATURE + (i * 0.1)
            
            # Temporarily set temperature
            original_temp = ttc_config.TEMPERATURE
            ttc_config.TEMPERATURE = temp
            
            result = self.run_inference_with_scaling(question, scaling_factor)
            results.append(result)
            
            # Restore original temperature
            ttc_config.TEMPERATURE = original_temp
        
        # Select best result based on quality score
        best_result = max(results, key=lambda x: x['quality_score'])
        best_result['all_candidates'] = results
        
        return best_result
    
    def run_adaptive_scaling(self, question: str, target_quality: float = 0.8) -> Dict:
        """
        Run adaptive scaling - automatically adjust compute until target quality is reached.
        """
        scaling_factors = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        results = []
        
        for scaling_factor in scaling_factors:
            result = self.run_inference_with_scaling(question, scaling_factor)
            results.append(result)
            
            # Check if we've reached target quality
            if result['quality_score'] >= target_quality:
                result['adaptive_scaling_info'] = {
                    'target_quality': target_quality,
                    'scaling_factors_tried': scaling_factors[:scaling_factors.index(scaling_factor) + 1],
                    'quality_achieved': result['quality_score']
                }
                return result
        
        # If target quality not reached, return best result
        best_result = max(results, key=lambda x: x['quality_score'])
        best_result['adaptive_scaling_info'] = {
            'target_quality': target_quality,
            'scaling_factors_tried': scaling_factors,
            'quality_achieved': best_result['quality_score'],
            'target_not_reached': True
        }
        
        return best_result
    
    def run_majority_voting(self, question: str, num_votes: int = 5, scaling_factor: float = 1.0) -> Dict:
        """
        Run majority voting inference (parallel scaling approach).
        """
        results = []
        
        for i in range(num_votes):
            result = self.run_inference_with_scaling(question, scaling_factor)
            results.append(result)
        
        # Extract solutions and find most common
        solutions = [result['solution'] for result in results]
        
        # Simple majority voting (in practice, you'd want more sophisticated matching)
        from collections import Counter
        solution_counter = Counter(solutions)
        most_common_solution = solution_counter.most_common(1)[0][0]
        
        # Find the result with the most common solution and highest quality
        best_result = None
        best_quality = -1
        
        for result in results:
            if result['solution'] == most_common_solution and result['quality_score'] > best_quality:
                best_result = result
                best_quality = result['quality_score']
        
        if best_result is None:
            # Fallback to first result if no match found
            best_result = results[0]
        
        best_result['majority_voting_info'] = {
            'num_votes': num_votes,
            'solution_distribution': dict(solution_counter),
            'most_common_solution': most_common_solution
        }
        
        return best_result
    
    def run_inference(self, question: str, strategy: str = "adaptive", **kwargs) -> Dict:
        """
        Main inference method with different strategies.
        
        Args:
            question: The question to answer
            strategy: One of ["adaptive", "best_of_n", "majority_voting", "fixed_scaling"]
            **kwargs: Additional arguments for specific strategies
        """
        if strategy == "adaptive":
            target_quality = kwargs.get('target_quality', 0.8)
            return self.run_adaptive_scaling(question, target_quality)
        
        elif strategy == "best_of_n":
            n = kwargs.get('n', ttc_config.BEST_OF_N)
            scaling_factor = kwargs.get('scaling_factor', 1.0)
            return self.run_best_of_n_inference(question, n, scaling_factor)
        
        elif strategy == "majority_voting":
            num_votes = kwargs.get('num_votes', 5)
            scaling_factor = kwargs.get('scaling_factor', 1.0)
            return self.run_majority_voting(question, num_votes, scaling_factor)
        
        elif strategy == "fixed_scaling":
            scaling_factor = kwargs.get('scaling_factor', 1.0)
            return self.run_inference_with_scaling(question, scaling_factor)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}") 