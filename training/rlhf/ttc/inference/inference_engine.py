from models.llm_model import LLMModel
from models.reward_model import RewardModel
from inference.mcts_module import MCTSNode, run_mcts
from config.config import ttc_config
import torch

class TTCInferenceEngine:
    def __init__(self, llm_model: LLMModel, reward_model: RewardModel):
        self.llm = llm_model
        self.reward_model = reward_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _generate_and_score_candidates(self, prompt: str, num_candidates: int):
        candidates = self.llm.generate(
            prompt,
            max_new_tokens=ttc_config.MAX_NEW_TOKENS,
            num_return_sequences=num_candidates,
            temperature=ttc_config.TEMPERATURE
        )
        scores = self.reward_model.score(candidates)
        return list(zip(candidates, scores))

    def run_inference(self, prompt: str, strategy: str = "best_of_n"):
        if strategy == "best_of_n":
            candidates_with_scores = self._generate_and_score_candidates(
                prompt, ttc_config.BEST_OF_N
            )
            # Select the candidate with the highest reward score
            best_candidate, best_score = max(candidates_with_scores, key=lambda item: item[1])
            return best_candidate, best_score
        elif strategy == "mcts":
            # For MCTS, the LLM will be used within the MCTS run_mcts function
            # The reward_model will be used to evaluate states/responses within MCTS
            # The run_mcts function needs to be adapted to use self.llm and self.reward_model
            # For now, a simplified call:
            def evaluate_mcts_state(state_text):
                # This needs to be a function that the MCTS can call
                # It should use the LLM to potentially continue generation
                # and then the reward model to score.
                # Simplification: treat the state_text as a complete response for scoring
                # In a real scenario, MCTS nodes would represent partial generations.
                score = self.reward_model.score([state_text])[0]
                return score

            final_result = run_mcts(
                initial_state=prompt,
                evaluate_state_fn=evaluate_mcts_state,
                iterations=ttc_config.MCTS_ITERATIONS,
                llm_model=self.llm, # Pass LLM to MCTS for generation within simulation
                reward_model=self.reward_model # Pass reward model to MCTS for scoring
            )
            return final_result, self.reward_model.score([final_result])[0] # Return final result and its score
        else:
            raise ValueError(f"Unknown strategy: {strategy}") 