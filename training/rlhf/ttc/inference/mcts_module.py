import numpy as np
import math

class MCTSNode:
    def __init__(self, state, parent=None, is_terminal=False):
        self.state = state  # Represents the current text/state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0 # Sum of rewards
        self.is_terminal = is_terminal # True if this state is a complete generation
        self.unexplored_actions = [] # List of possible next actions/tokens

    def is_fully_expanded(self):
        return len(self.unexplored_actions) == 0 and len(self.children) > 0

    def best_child(self, c_param=1.0):
        # UCB1 formula to select the best child node
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, llm_model, tokenizer, max_new_tokens=10):
        # Generate possible next tokens/sequences using the LLM
        # This is a simplified expansion for MCTS in LLM context.
        # In a real setup, it might involve token-level generation.
        # For now, let's assume it generates a few "next steps" or "continuations".
        if not self.unexplored_actions:
            # Generate a few diverse continuations from the current state
            # This is where the LLM is used to explore potential paths
            continuations = llm_model.generate(
                self.state,
                max_new_tokens=max_new_tokens,
                num_return_sequences=3, # Generate 3 diverse continuations for expansion
                temperature=0.8 # Higher temperature for diverse exploration
            )
            self.unexplored_actions = continuations # Each continuation is a potential action/next state

        if self.unexplored_actions:
            next_continuation = self.unexplored_actions.pop(0)
            # Determine if this continuation leads to a terminal state (e.g., ends with EOS token)
            is_new_terminal = tokenizer.eos_token in next_continuation # Simplified check
            new_node = MCTSNode(next_continuation, parent=self, is_terminal=is_new_terminal)
            self.children.append(new_node)
            return new_node
        return None

    def simulate(self, evaluate_state_fn):
        # Rollout / Playout: From the current state, run a simulation to get a reward
        # This could involve further LLM generation until a terminal state is reached,
        # then scoring the final generated text using the reward model.
        # For simplicity, we directly score the current state text.
        return evaluate_state_fn(self.state)

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)

def run_mcts(initial_state: str, evaluate_state_fn, iterations: int, llm_model, reward_model):
    # Pass tokenizer directly to run_mcts or make it available via llm_model
    tokenizer = llm_model.tokenizer
    root = MCTSNode(initial_state)

    for _ in range(iterations):
        node = root
        # Selection
        while node.is_fully_expanded() and not node.is_terminal:
            node = node.best_child()

        # Expansion
        if not node.is_terminal:
            # Expand using the LLM to get new potential states
            new_node = node.expand(llm_model, tokenizer)
            if new_node:
                node = new_node
            else:
                # If no new node was expanded (e.g., no more unexplored actions),
                # continue simulation from the current node.
                pass


        # Simulation (Rollout)
        # The simulate function will use the evaluate_state_fn which internally uses the reward_model
        # For LLM-based MCTS, 'simulate' might involve generating a full response
        # from the current node's state using the LLM and then scoring it.
        # Here, evaluate_state_fn is designed to take text and return a score.
        reward = node.simulate(evaluate_state_fn)

        # Backpropagation
        node.backpropagate(reward)

    # After all iterations, select the best path/result from the root's children
    # This might be the child with the highest average value or highest total value
    if root.children:
        # Choose the child that leads to the highest average reward
        best_final_node = max(root.children, key=lambda c: c.value / (c.visits if c.visits > 0 else 1e-6))
        # In a real scenario, you'd likely want the full generated sequence,
        # possibly by reconstructing the path from root to best_final_node if needed.
        return best_final_node.state
    else:
        return initial_state # Fallback if no children were expanded 