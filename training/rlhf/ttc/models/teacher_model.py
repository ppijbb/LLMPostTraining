from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Optional
import re

class TeacherModel:
    """
    RLT-style Teacher Model for reasoning instruction.
    This model is trained to generate high-quality reasoning traces and solutions.
    """
    
    def __init__(self, model_name: str, quantization_config=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        
        if quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def format_reasoning_prompt(self, question: str, system_prompt: str = None) -> str:
        """Format a question into a reasoning prompt with thinking tags."""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that solves problems step by step."
        
        formatted_prompt = f"{system_prompt}\n\nQuestion: {question}\n\n<think>"
        return formatted_prompt
    
    def generate_reasoning_trace(self, question: str, max_tokens: int = 512, 
                               temperature: float = 0.7, system_prompt: str = None) -> Dict:
        """Generate a reasoning trace for a given question."""
        prompt = self.format_reasoning_prompt(question, system_prompt)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract thinking and solution parts
        thinking_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
        solution_match = re.search(r'<solution>(.*?)</solution>', generated_text, re.DOTALL)
        
        thinking = thinking_match.group(1).strip() if thinking_match else ""
        solution = solution_match.group(1).strip() if solution_match else ""
        
        return {
            "question": question,
            "reasoning_trace": thinking,
            "solution": solution,
            "full_response": generated_text
        }
    
    def generate_multiple_traces(self, question: str, num_traces: int = 3, 
                               max_tokens: int = 512, temperature: float = 0.7) -> List[Dict]:
        """Generate multiple reasoning traces for the same question."""
        traces = []
        for i in range(num_traces):
            # Vary temperature slightly for diversity
            current_temp = temperature + (i * 0.1)
            trace = self.generate_reasoning_trace(question, max_tokens, current_temp)
            traces.append(trace)
        return traces
    
    def evaluate_reasoning_quality(self, reasoning_trace: str) -> float:
        """Simple heuristic to evaluate reasoning quality."""
        # This is a placeholder - in practice, you'd use a trained reward model
        quality_score = 0.0
        
        # Check for step-by-step reasoning
        if "step" in reasoning_trace.lower() or "first" in reasoning_trace.lower():
            quality_score += 0.3
        
        # Check for mathematical expressions
        if re.search(r'\d+[\+\-\*\/\^]\d+', reasoning_trace):
            quality_score += 0.2
        
        # Check for logical connectors
        logical_connectors = ["because", "therefore", "thus", "hence", "so"]
        if any(connector in reasoning_trace.lower() for connector in logical_connectors):
            quality_score += 0.3
        
        # Check for explanation length (not too short, not too long)
        word_count = len(reasoning_trace.split())
        if 20 <= word_count <= 200:
            quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def save_model(self, output_dir: str):
        """Save the teacher model."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    @classmethod
    def from_pretrained(cls, model_path: str, quantization_config=None):
        """Load a pre-trained teacher model."""
        return cls(model_path, quantization_config) 