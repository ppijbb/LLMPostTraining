from typing import List, Dict, Optional
import os
import sys
import json
import torch
from tqdm import tqdm
import re
from datasets import load_dataset

from deepeval.metrics import BaseMetric
from deepeval.evaluate.execute import execute_test_cases
from deepeval.dataset import EvaluationDataset
from deepeval.integrations.hugging_face import DeepEvalHuggingFaceCallback
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.integrations.hugging_face.utils import generate_test_cases
from transformers import (
        Trainer,
        TrainingArguments,
        TrainerState,
        TrainerControl,
        AutoTokenizer,
        AutoProcessor,
        AutoConfig,
        GenerationConfig
    )
from transformers.trainer_callback import TrainerCallback
from transformers.generation.stopping_criteria import StopStringCriteria, StoppingCriteriaList, MaxLengthCriteria
from deepeval.benchmarks import MMLU

from deepeval.benchmarks.hellaswag.hellaswag import HellaSwag
from deepeval.benchmarks import BigBenchHard
from deepeval.benchmarks import HumanEval
from deepeval.benchmarks import SQuAD
from deepeval.benchmarks import GSM8K
from deepeval.benchmarks import MathQA
from deepeval.benchmarks import TruthfulQA
from deepeval.benchmarks import IFEval
# from deepeval.benchmarks import AGIEval
from deepeval.benchmarks import ARC
from deepeval.benchmarks import HellaSwag
# from deepeval.benchmarks import OpenBookQA
# from deepeval.benchmarks import PIQA
# from deepeval.benchmarks import WinoGrande
from deepeval.benchmarks import BoolQ
# from deepeval.benchmarks import CB
# from deepeval.benchmarks import COPA
# from deepeval.benchmarks import MultiRC
# from deepeval.benchmarks import ReCoRD
# from deepeval.benchmarks import RTE
# from deepeval.benchmarks import WiC
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import ArenaTestCase, LLMTestCase
from deepeval.metrics import ArenaGEval
from deepeval.integrations.hugging_face.rich_manager import RichManager
from trl import SFTTrainer
from transformers.trainer_callback import ProgressCallback
from evaluation.local_deepeval import LocalModel
# Add parent directory to path for custom model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Distributed rank helper
def _is_main_process() -> bool:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True

# Disable torch.compile to avoid data-dependent branching issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch.compiler.disable()
torch._dynamo.reset()


# DeepSpeed 지원을 위한 유틸리티 함수
def get_inference_model_for_generate(model):
    """
    DeepSpeed로 감싸진 모델에서 실제 모델을 추출합니다.
    ZeRO Stage 3의 경우 GatheredParameters 컨텍스트를 반환합니다.
    
    Returns:
        tuple: (actual_model, context_manager) 또는 (actual_model, None)
    """
    # DeepSpeed로 감싸진 모델인지 확인
    if hasattr(model, 'module'):
        # DeepSpeed engine이 있는지 확인
        if hasattr(model, 'engine'):
            engine = model.engine
            # ZeRO stage 확인
            try:
                zero_stage = engine.zero_optimization_stage()
            except:
                zero_stage = 0
            
            # ZeRO Stage 3인 경우 GatheredParameters 필요
            if zero_stage == 3:
                try:
                    import deepspeed
                    return model.module, deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=None)
                except ImportError:
                    # deepspeed가 없으면 그냥 module 사용
                    return model.module, None
            else:
                # ZeRO 0, 1, 2는 그냥 module 사용
                return model.module, None
        else:
            # DDP 등 다른 래핑
            return model.module, None
    else:
        # 래핑되지 않은 모델
        return model, None

@torch.inference_mode()
def run_mme_evaluation(model, tokenizer, max_samples_per_task=50):
    """
    MME 벤치마크를 수동으로 실행하여 모델의 비전-언어 능력을 평가합니다.
    """
    print("\n" + "="*50)
    print("MME (Multimodal Model Evaluation) 수동 평가 시작")
    print("="*50)

    try:
        print("MME 데이터셋 로딩 중...")
        mme_dataset = load_dataset("MMMU/MME")
        print("MME 데이터셋 로드 완료.")
    except Exception as e:
        print(f"MME 데이터셋 로딩 실패: {e}")
        return {}

    tasks_to_run = ['color', 'count', 'position', 'posters', 'ocr']
    results = {}
    
    # DeepSpeed 모델 처리: 실제 모델 추출
    inference_model, gather_context = get_inference_model_for_generate(model)
    
    # device 가져오기
    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(inference_model, 'device'):
        device = inference_model.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for task in tasks_to_run:
        if task not in mme_dataset:
            print(f"경고: MME 데이터셋에 '{task}' 태스크가 없습니다. 건너뜁니다.")
            continue
            
        print(f"\n--- '{task}' 태스크 평가 중 ---")
        task_dataset = mme_dataset[task]
        correct_predictions = 0
        total_samples = 0

        # Limit samples for faster evaluation during training
        samples_to_evaluate = task_dataset[:max_samples_per_task]

        for sample in tqdm(samples_to_evaluate, desc=f"Evaluating {task}"):
            image = sample['image']
            question = sample['question']
            ground_truth = sample['answer'].strip().lower()

            prompt = f"<image>\n{question}\nAnswer with Yes or No."
            
            inputs = tokenizer(text=[prompt], images=[image], return_tensors="pt").to(device)

            # ZeRO Stage 3인 경우 GatheredParameters 컨텍스트 사용
            if gather_context is not None:
                with gather_context:
                    generated_ids = inference_model.generate(**inputs, max_new_tokens=10, do_sample=False)
            else:
                generated_ids = inference_model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            response_text = tokenizer.decode(generated_ids[0])
            cleaned_prompt = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            answer_text = response_text.replace(cleaned_prompt, '').strip().lower()

            match = re.search(r'\b(yes|no)\b', answer_text)
            predicted_answer = match.group(0) if match else "n/a"
            
            if predicted_answer == ground_truth:
                correct_predictions += 1
            total_samples += 1
        
        if total_samples > 0:
            accuracy = (correct_predictions / total_samples) * 100
            results[task] = accuracy
            print(f"'{task}' 태스크 정확도: {accuracy:.2f}% ({correct_predictions}/{total_samples})")

    if results:
        overall_accuracy = sum(results.values()) / len(results)
        results['overall'] = overall_accuracy
        print(f"\n  - {'평균':<10}: {overall_accuracy:.2f}%")
    
    return results


def get_model_eval_callback(
    trainer: Trainer,
    evaluation_dataset: Optional[EvaluationDataset] = None,
    metrics: Optional[List[BaseMetric]] = None,
    tokenizer_args: Optional[Dict] = {},
    aggregation_method: str = "avg",
    show_table: bool = False,
    enable_benchmarks: bool = False,
    benchmarks_to_run: List[str] = ['mmlu', 'hellaswag', 'gsm8k', 'truthfulqa', 'arc', 'piqa'],
    benchmark_eval_frequency: int = 1000,  # Changed to 1000 steps default
    eval_mode: str = "step",  # Added eval_mode parameter with step as default
    mme_max_samples: int = 20,  # Limit MME samples for faster evaluation
    # Lightweight benchmark controls
    benchmark_max_samples_per_task: int = 3,
    benchmark_gsm8k_max_samples: int = 3,
    benchmark_max_tasks: Optional[int] = None,
    benchmark_max_new_tokens: int = 64,
    benchmark_disable_cot: bool = True,
    benchmark_ifeval_max_samples: int = 5,  # IFEval n_problems limit
): 
    if evaluation_dataset is None:
        """ ref
        evaluation_dataset = EvaluationDataset(
            goldens=[
                Golden(input="..."),
                Golden(input="...")
            ]
        )
        """
        from deepeval.benchmarks.mmlu.task import MMLUTask
        from deepeval.benchmarks import MMLU

        benchmark = MMLU()
        evaluation_dataset = EvaluationDataset(
            goldens=benchmark.load_benchmark_dataset(
                task=MMLUTask.HIGH_SCHOOL_EUROPEAN_HISTORY
            )
        )
        # print("Golden dataset: ","\n",  evaluation_dataset.goldens, "\n")
        # evaluation_dataset = EvaluationDataset(
        #     test_cases=[
        #         LLMTestCase(
        #             name="GPT-4",
        #             input="What is the capital of France?",
        #             actual_output="Paris",
        #         ),
        #         LLMTestCase(
        #             name="Claude",
        #             input="What is the capital of France?",
        #             actual_output="Paris is the capital of France.",
        #         )
        #     ]
        # )
        evaluation_dataset.alias = evaluation_dataset._alias
    if metrics is None:
        metrics = [
             ArenaGEval(
                name="Friendly",
                criteria="Choose the winter of the more friendly contestant based on the input and actual output",
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
            )
        ]
    tokenizer_args.update({
        "return_tensors": "pt", 
        "padding": "max_length", 
        "truncation": True, 
        "max_length": 8192
        })
    return ModelEvalCallback(
        trainer=trainer,
        evaluation_dataset=evaluation_dataset,
        metrics=metrics,
        tokenizer_args=tokenizer_args,
        aggregation_method=aggregation_method,
        show_table=show_table,
        enable_benchmarks=enable_benchmarks,
        benchmarks_to_run=benchmarks_to_run,
        benchmark_eval_frequency=benchmark_eval_frequency,
        eval_mode=eval_mode,
        mme_max_samples=mme_max_samples,
        benchmark_max_samples_per_task=benchmark_max_samples_per_task,
        benchmark_gsm8k_max_samples=benchmark_gsm8k_max_samples,
        benchmark_max_tasks=benchmark_max_tasks,
        benchmark_max_new_tokens=benchmark_max_new_tokens,
        benchmark_disable_cot=benchmark_disable_cot,
        benchmark_ifeval_max_samples=benchmark_ifeval_max_samples,
    )

class EmptyRichManager(RichManager): 
    def __init__(self, show_table: bool, total_train_epochs: int) -> None:
        """
        Initialize RichManager.

        Args:
            show_table (bool): Flag to show or hide the table.
            total_train_epochs (int): Total number of training epochs.
        """
        pass

    def _initialize_progress_trackers(self) -> None:
        pass

    def change_spinner_text(self,*args, **kwargs) -> None:
        pass

    def stop(self) -> None:
        pass

    def start(self) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        pass

    def create_column(self, *args, **kwargs):
        return None, None

    def advance_progress(self) -> None:
        pass


class ModelEvalCallback(DeepEvalHuggingFaceCallback):
    '''
    This callback is used to evaluate the model during training.
    It is used to evaluate the model on the evaluation dataset and save the results to a file.
    The results are saved to a file in the format of a pandas dataframe.
    The file is saved to the path specified by the `results_file` argument.
    The file is saved to the path specified by the `results_file` argument.
    '''
    train_mode = "transformers"
    
    def __init__(
        self, 
        trainer: Trainer,
        evaluation_dataset: Optional[EvaluationDataset] = None,
        metrics: Optional[List[BaseMetric]] = None,
        tokenizer_args: Optional[Dict] = None,
        aggregation_method: str = "avg",
        show_table: bool = False,
        enable_benchmarks: bool = False,
        benchmarks_to_run: List[str] = ['mmlu', 'hellaswag', 'gsm8k', 'truthfulqa', 'arc', 'piqa'],
        benchmark_eval_frequency: int = 1000,  # Changed to step-based (1000 steps)
        eval_mode: str = "step",  # Changed default to step-based
        mme_max_samples: int = 20,
        benchmark_max_samples_per_task: int = 3,
        benchmark_gsm8k_max_samples: int = 3,
        benchmark_max_tasks: Optional[int] = None,
        benchmark_max_new_tokens: int = 64,
        benchmark_disable_cot: bool = True,
        benchmark_ifeval_max_samples: int = 5,  # IFEval n_problems limit
        *args, 
        **kwargs
    )->None:
        super().__init__(
            trainer=trainer, 
            evaluation_dataset=evaluation_dataset, 
            metrics=metrics,
            tokenizer_args=tokenizer_args,
            show_table=show_table)

        self.trainer.add_callback(ProgressCallback)
        self.rich_manager = EmptyRichManager(show_table=show_table, total_train_epochs=trainer.args.num_train_epochs)
        self.eval_model = LocalModel(model=trainer.model, tokenizer=trainer.tokenizer)
        self.enable_benchmarks = enable_benchmarks
        # Limit number of tasks if requested (to keep step eval light)
        if benchmark_max_tasks is not None and benchmark_max_tasks > 0:
            self.benchmarks_to_run = benchmarks_to_run[:benchmark_max_tasks]
        else:
            self.benchmarks_to_run = benchmarks_to_run
        self.benchmark_eval_frequency = benchmark_eval_frequency
        self.eval_mode = eval_mode  # "step" or "epoch"
        self.mme_max_samples = mme_max_samples
        self.benchmark_results_history = []
        self.last_eval_step = 0  # Track last evaluation step
        self.is_main_process = _is_main_process()
        # Lightweight controls
        self.benchmark_max_samples_per_task = max(1, benchmark_max_samples_per_task)
        self.benchmark_gsm8k_max_samples = max(1, benchmark_gsm8k_max_samples)
        self.benchmark_max_new_tokens = max(8, benchmark_max_new_tokens)
        self.benchmark_disable_cot = benchmark_disable_cot
        self.benchmark_ifeval_max_samples = max(1, benchmark_ifeval_max_samples)
        
        try:
            import deepspeed
        except ImportError:
            print("Deepspeed is not installed, using outlines instead")
        else:
            self.train_mode = "deepspeed"

    
    def _calculate_metric_scores(self) -> Dict[str, List[float]]:
        # return super()._calculate_metric_scores()
        on_log_metrics = IFEval(n_problems=10, verbose_mode=True)
        self.eval_model = LocalModel(model=self.trainer.model, tokenizer=self.trainer.tokenizer)
        try:
            with torch.no_grad():
                test_results = on_log_metrics.evaluate(self.eval_model)
            scores = {
                "ifeval": [test_results["overall_accuracy"]]
            }
            self.trainer.log(scores)
        except Exception as e:
            print(f"Error in _calculate_metric_scores: {e}\nIFEval scores set to 0.0")
            scores = self._aggregate_scores({"ifeval": [0.0]})
        return scores
    
    def _aggregate_scores(
        self, 
        scores: Dict[str, List[float]]
    ) -> Dict[str, float]:
        return super()._aggregate_scores(scores)
        
    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_epoch_begin(args, state, control, **kwargs)
        
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Step-based benchmark evaluation"""
        # Only run heavy benchmarking on the main process
        if not self.is_main_process:
            return control
        # Only run if step-based evaluation is enabled
        if (self.eval_mode == "step" and 
            self.enable_benchmarks and
            state.global_step > 0 and
            state.global_step - self.last_eval_step >= self.benchmark_eval_frequency):
            
            with torch.inference_mode():
                print(f"\n{'='*60}")
                print(f"Step {state.global_step} Benchmark Evaluation")
                print(f"{'='*60}")
                
                benchmark_results = self._run_benchmark_evaluation(mode="step")
                
                if benchmark_results:
                    self.benchmark_results_history.append({
                        'step': state.global_step,
                        'epoch': state.epoch,
                        'results': benchmark_results
                    })
                    
                    # Log benchmark results to trainer
                    for metric_name, score in benchmark_results.items():
                        self.trainer.log({
                            f"benchmark/{metric_name}": score,
                            "step": state.global_step,
                            "epoch": state.epoch
                        })
                    
                    print(f"Benchmark results logged for step {state.global_step}")
                
                self.last_eval_step = state.global_step

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Only run epoch-based evaluation if eval_mode is "epoch"
        if not self.is_main_process:
            return control
        if self.eval_mode == "epoch":
            with torch.inference_mode():
                # with torch.device("cuda"):
                #     control.should_log = True
                #     self.rich_manager.change_spinner_text(
                #         self.task_descriptions["generating"]
                #     )
                #     test_cases = generate_test_cases(
                #         self.eval_model,
                #         self.trainer.tokenizer.tokenizer,
                #         self.tokenizer_args,
                #         self.evaluation_dataset,
                #     )
                #     self.evaluation_dataset.test_cases = test_cases

                # Run benchmark evaluation if enabled and it's the right epoch
                if (self.enable_benchmarks and 
                    state.epoch is not None and 
                    (state.epoch) % self.benchmark_eval_frequency == 0):
                    
                    print(f"\n{'='*60}")
                    print(f"Epoch {state.epoch} Benchmark Evaluation")
                    print(f"{'='*60}")
                    
                    benchmark_results = self._run_benchmark_evaluation(mode="epoch")
                    
                    if benchmark_results:
                        self.benchmark_results_history.append({
                            'epoch': state.epoch,
                            'step': state.global_step,
                            'results': benchmark_results
                        })
                        
                        # Log benchmark results to trainer
                        for metric_name, score in benchmark_results.items():
                            self.trainer.log({
                                f"benchmark/{metric_name}": score,
                                "epoch": state.epoch,
                                "step": state.global_step
                            })
                        
                        print(f"Benchmark results logged for epoch {state.epoch}")
    
    @torch.no_grad()
    def _run_benchmark_evaluation(self, mode:str="epoch") -> Dict[str, float]:
        """
        Run benchmark evaluation on the current model with progress tracking
        CRITICAL: Ensure model is in eval mode and no gradients are computed
        """
        results = {}
        
        # CRITICAL: Save original model state and switch to eval mode
        original_training_state = None
        actual_model = None
        trainer_model = None
        
        try:
            # Get actual model from trainer (handle DeepSpeed wrapping)
            trainer_model = self.trainer.model
            if hasattr(trainer_model, 'module'):
                actual_model = trainer_model.module
            else:
                actual_model = trainer_model
            
            # Save original training state
            if actual_model is not None:
                original_training_state = actual_model.training
                # CRITICAL: Switch to eval mode to disable gradient computation
                actual_model.eval()
                print("   🔧 Model switched to eval mode (gradients disabled)")
            
            # Clear GPU cache before benchmark evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Get current model and tokenizer from trainer
            model = self.eval_model
            tokenizer = self.trainer.tokenizer
            
            if tokenizer is None:
                print("Warning: No tokenizer available for benchmark evaluation")
                return results
            
            # Fast/slow controls
            fast_mode = (mode != "epoch")
            n_per_task = self.benchmark_max_samples_per_task if fast_mode else None
            gsm8k_n = self.benchmark_gsm8k_max_samples if fast_mode else 1319
            
            # Track overall progress
            total_benchmarks = len(self.benchmarks_to_run)
            completed_benchmarks = 0
            
            print(f"\n📊 Running {total_benchmarks} benchmark(s): {', '.join(self.benchmarks_to_run)}")
            print(f"   Mode: {'Fast (step)' if fast_mode else 'Full (epoch)'}")
            if fast_mode:
                print(f"   Samples per task: {n_per_task}")
            print("-" * 60)
            
            # Run text-based benchmarks using self as the model wrapper
            if 'mmlu' in self.benchmarks_to_run:
                try:
                    completed_benchmarks += 1
                    print(f"\n[{completed_benchmarks}/{total_benchmarks}] 🔍 Running MMLU benchmark...")
                    print(f"   Config: n_problems_per_task={n_per_task or 'all'}, n_shots=3")
                    import time
                    start_time = time.time()
                    mmlu_benchmark = MMLU(
                        n_problems_per_task=n_per_task or 5,
                        n_shots=3)
                    mmlu_benchmark.evaluate(model=self.eval_model)
                    elapsed = time.time() - start_time
                    results['mmlu'] = mmlu_benchmark.overall_score
                    print(f"   ✅ MMLU completed in {elapsed:.1f}s - Score: {mmlu_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"   ❌ MMLU evaluation failed: {e}")
                    results['mmlu'] = 0.0
            
            if 'hellaswag' in self.benchmarks_to_run:
                try:
                    completed_benchmarks += 1
                    print(f"\n[{completed_benchmarks}/{total_benchmarks}] 🔍 Running HellaSwag benchmark...")
                    print(f"   Config: n_problems_per_task={n_per_task or 'all'}, n_shots=3")
                    import time
                    start_time = time.time()
                    hellaswag_benchmark = HellaSwag(
                        n_problems_per_task=n_per_task or 5,
                        n_shots=3)
                    hellaswag_benchmark.evaluate(model=self.eval_model)
                    elapsed = time.time() - start_time
                    results['hellaswag'] = hellaswag_benchmark.overall_score
                    print(f"   ✅ HellaSwag completed in {elapsed:.1f}s - Score: {hellaswag_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"   ❌ HellaSwag evaluation failed: {e}")
                    results['hellaswag'] = 0.0
            
            if 'gsm8k' in self.benchmarks_to_run:
                try:
                    completed_benchmarks += 1
                    print(f"\n[{completed_benchmarks}/{total_benchmarks}] 🔍 Running GSM8K benchmark...")
                    enable_cot = False if (fast_mode and self.benchmark_disable_cot) else True
                    print(f"   Config: n_problems={gsm8k_n}, n_shots=3, enable_cot={enable_cot}")
                    import time
                    start_time = time.time()
                    gsm8k_benchmark = GSM8K(
                        n_problems=gsm8k_n, 
                        n_shots=3, 
                        enable_cot=enable_cot,
                    )
                    gsm8k_benchmark.evaluate(model=self.eval_model)
                    elapsed = time.time() - start_time
                    results['gsm8k'] = gsm8k_benchmark.overall_score
                    print(f"   ✅ GSM8K completed in {elapsed:.1f}s - Score: {gsm8k_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"   ❌ GSM8K evaluation failed: {e}")
                    results['gsm8k'] = 0.0
            
            if 'truthfulqa' in self.benchmarks_to_run:
                try:
                    completed_benchmarks += 1
                    print(f"\n[{completed_benchmarks}/{total_benchmarks}] 🔍 Running TruthfulQA benchmark...")
                    print(f"   Config: n_problems_per_task={n_per_task or 'all'}")
                    import time
                    start_time = time.time()
                    truthfulqa_benchmark = TruthfulQA(
                        n_problems_per_task=n_per_task or 5,
                    )
                    truthfulqa_benchmark.evaluate(model=self.eval_model)
                    elapsed = time.time() - start_time
                    results['truthfulqa'] = truthfulqa_benchmark.overall_score
                    print(f"   ✅ TruthfulQA completed in {elapsed:.1f}s - Score: {truthfulqa_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"   ❌ TruthfulQA evaluation failed: {e}")
                    results['truthfulqa'] = 0.0
            
            if 'arc' in self.benchmarks_to_run:
                try:
                    completed_benchmarks += 1
                    print(f"\n[{completed_benchmarks}/{total_benchmarks}] 🔍 Running ARC benchmark...")
                    print(f"   Config: n_problems={n_per_task or 'all'}, n_shots=3")
                    import time
                    start_time = time.time()
                    arc_benchmark = ARC(
                        n_problems=n_per_task or 5,
                        n_shots=3)
                    arc_benchmark.evaluate(model=self.eval_model)
                    elapsed = time.time() - start_time
                    results['arc'] = arc_benchmark.overall_score
                    print(f"   ✅ ARC completed in {elapsed:.1f}s - Score: {arc_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"   ❌ ARC evaluation failed: {e}")
                    results['arc'] = 0.0
            
            # if 'piqa' in self.benchmarks_to_run:
            #     try:
            #         print("Running PIQA benchmark...")
            #         piqa_benchmark = PIQA(n_shots=3)
            #         piqa_benchmark.evaluate(model=self)
            #         results['piqa'] = piqa_benchmark.overall_score
            #         print(f"PIQA Score: {piqa_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"PIQA evaluation failed: {e}")
            #         results['piqa'] = 0.0
            
            if 'bigbenchhard' in self.benchmarks_to_run:
                try:
                    completed_benchmarks += 1
                    print(f"\n[{completed_benchmarks}/{total_benchmarks}] 🔍 Running BigBenchHard benchmark...")
                    print(f"   Config: n_problems_per_task={n_per_task or 'all'}, enable_cot=True")
                    import time
                    start_time = time.time()
                    bigbench_benchmark = BigBenchHard(
                        n_problems_per_task=n_per_task or 5,
                        enable_cot=True)
                    bigbench_benchmark.evaluate(model=self.eval_model)
                    elapsed = time.time() - start_time
                    results['bigbenchhard'] = bigbench_benchmark.overall_score
                    print(f"   ✅ BigBenchHard completed in {elapsed:.1f}s - Score: {bigbench_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"   ❌ BigBenchHard evaluation failed: {e}")
                    results['bigbenchhard'] = 0.0
            
            if 'humaneval' in self.benchmarks_to_run:
                try:
                    completed_benchmarks += 1
                    n_samples = 5 if mode != "epoch" else 200
                    print(f"\n[{completed_benchmarks}/{total_benchmarks}] 🔍 Running HumanEval benchmark...")
                    print(f"   Config: n={n_samples}")
                    import time
                    start_time = time.time()
                    humaneval_benchmark = HumanEval(
                        # tasks=[HumanEvalTask.HAS_CLOSE_ELEMENTS, HumanEvalTask.SORT_NUMBERS],
                        n=n_samples,
                    )
                    humaneval_benchmark.evaluate(model=self.eval_model)
                    elapsed = time.time() - start_time
                    results['humaneval'] = humaneval_benchmark.overall_score
                    print(f"   ✅ HumanEval completed in {elapsed:.1f}s - Score: {humaneval_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"   ❌ HumanEval evaluation failed: {e}")
                    results['humaneval'] = 0.0
            
            if 'squad' in self.benchmarks_to_run:
                try:
                    completed_benchmarks += 1
                    print(f"\n[{completed_benchmarks}/{total_benchmarks}] 🔍 Running SQuAD benchmark...")
                    print(f"   Config: n_problems_per_task={n_per_task or 'all'}, n_shots=3")
                    import time
                    start_time = time.time()
                    squad_benchmark = SQuAD(
                        # tasks=[SQuADTask.PHARMACY, SQuADTask.NORMANS],
                        n_shots=3,
                        n_problems_per_task=n_per_task or 5,
                    )
                    squad_benchmark.evaluate(model=self.eval_model)
                    elapsed = time.time() - start_time
                    results['squad'] = squad_benchmark.overall_score
                    print(f"   ✅ SQuAD completed in {elapsed:.1f}s - Score: {squad_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"   ❌ SQuAD evaluation failed: {e}")
                    results['squad'] = 0.0
            
            if 'mathqa' in self.benchmarks_to_run:
                try:
                    completed_benchmarks += 1
                    print(f"\n[{completed_benchmarks}/{total_benchmarks}] 🔍 Running MathQA benchmark...")
                    print(f"   Config: n_problems_per_task={n_per_task or 'all'}, n_shots=3")
                    import time
                    start_time = time.time()
                    mathqa_benchmark = MathQA(
                        # tasks=[MathQATask.PROBABILITY, MathQATask.GEOMETRY],
                        n_shots=3,
                        n_problems_per_task=n_per_task or 5,
                    )
                    mathqa_benchmark.evaluate(model=self.eval_model)
                    elapsed = time.time() - start_time
                    results['mathqa'] = mathqa_benchmark.overall_score
                    print(f"   ✅ MathQA completed in {elapsed:.1f}s - Score: {mathqa_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"   ❌ MathQA evaluation failed: {e}")
                    results['mathqa'] = 0.0
            
            # if 'agieval' in self.benchmarks_to_run:
            #     try:
            #         print("Running AGIEval benchmark...")
            #         agieval_benchmark = AGIEval(n_shots=3)
            #         agieval_benchmark.evaluate(model=self.eval_model)
            #         results['agieval'] = agieval_benchmark.overall_score
            #         print(f"AGIEval Score: {agieval_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"AGIEval evaluation failed: {e}")
            #         results['agieval'] = 0.0
            
            # if 'openbookqa' in self.benchmarks_to_run:
            #     try:
            #         print("Running OpenBookQA benchmark...")
            #         openbookqa_benchmark = OpenBookQA(n_shots=3)
            #         openbookqa_benchmark.evaluate(model=self.eval_model)
            #         results['openbookqa'] = openbookqa_benchmark.overall_score
            #         print(f"OpenBookQA Score: {openbookqa_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"OpenBookQA evaluation failed: {e}")
            #         results['openbookqa'] = 0.0
            
            # if 'winogrande' in self.benchmarks_to_run:
            #     try:
            #         print("Running WinoGrande benchmark...")
            #         winogrande_benchmark = WinoGrande(n_shots=3)
            #         winogrande_benchmark.evaluate(model=self.eval_model)
            #         results['winogrande'] = winogrande_benchmark.overall_score
            #         print(f"WinoGrande Score: {winogrande_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"WinoGrande evaluation failed: {e}")
            #         results['winogrande'] = 0.0
            
            if 'boolq' in self.benchmarks_to_run:
                try:
                    completed_benchmarks += 1
                    print(f"\n[{completed_benchmarks}/{total_benchmarks}] 🔍 Running BoolQ benchmark...")
                    print(f"   Config: n_problems={n_per_task or 'all'}, n_shots=3")
                    import time
                    start_time = time.time()
                    boolq_benchmark = BoolQ(
                        n_problems=n_per_task or 5,
                        n_shots=3)
                    boolq_benchmark.evaluate(model=self.eval_model)
                    elapsed = time.time() - start_time
                    results['boolq'] = boolq_benchmark.overall_score
                    print(f"   ✅ BoolQ completed in {elapsed:.1f}s - Score: {boolq_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"   ❌ BoolQ evaluation failed: {e}")
                    results['boolq'] = 0.0
            
            if 'ifeval' in self.benchmarks_to_run:
                try:
                    completed_benchmarks += 1
                    n_problems = self.benchmark_ifeval_max_samples if fast_mode else None
                    print(f"\n[{completed_benchmarks}/{total_benchmarks}] 🔍 Running IFEval benchmark...")
                    print(f"   Config: n_problems={n_problems or 'all'}")
                    import time
                    start_time = time.time()
                    ifeval_benchmark = IFEval(
                        n_problems=n_problems,
                    )
                    ifeval_benchmark.evaluate(model=self.eval_model)
                    elapsed = time.time() - start_time
                    results['ifeval'] = ifeval_benchmark.overall_score
                    print(f"   ✅ IFEval completed in {elapsed:.1f}s - Score: {ifeval_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"   ❌ IFEval evaluation failed: {e}")
                    results['ifeval'] = 0.0
            
            # if 'cb' in self.benchmarks_to_run:
            #     try:
            #         print("Running CB benchmark...")
            #         cb_benchmark = CB(n_shots=3)
            #         cb_benchmark.evaluate(model=self.eval_model)
            #         results['cb'] = cb_benchmark.overall_score
            #         print(f"CB Score: {cb_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"CB evaluation failed: {e}")
            #         results['cb'] = 0.0
            
            # if 'copa' in self.benchmarks_to_run:
            #     try:
            #         print("Running COPA benchmark...")
            #         copa_benchmark = COPA(n_shots=3)
            #         copa_benchmark.evaluate(model=self.eval_model)
            #         results['copa'] = copa_benchmark.overall_score
            #         print(f"COPA Score: {copa_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"COPA evaluation failed: {e}")
            #         results['copa'] = 0.0
            
            # if 'multirc' in self.benchmarks_to_run:
            #     try:
            #         print("Running MultiRC benchmark...")
            #         multirc_benchmark = MultiRC(n_shots=3)
            #         multirc_benchmark.evaluate(model=self.eval_model)
            #         results['multirc'] = multirc_benchmark.overall_score
            #         print(f"MultiRC Score: {multirc_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"MultiRC evaluation failed: {e}")
            #         results['multirc'] = 0.0
            
            # if 'record' in self.benchmarks_to_run:
            #     try:
            #         print("Running ReCoRD benchmark...")
            #         record_benchmark = ReCoRD(n_shots=3)
            #         record_benchmark.evaluate(model=self.eval_model)
            #         results['record'] = record_benchmark.overall_score
            #         print(f"ReCoRD Score: {record_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"ReCoRD evaluation failed: {e}")
            #         results['record'] = 0.0
            
            # if 'rte' in self.benchmarks_to_run:
            #     try:
            #         print("Running RTE benchmark...")
            #         rte_benchmark = RTE(n_shots=3)
            #         rte_benchmark.evaluate(model=self.eval_model)
            #         results['rte'] = rte_benchmark.overall_score
            #         print(f"RTE Score: {rte_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"RTE evaluation failed: {e}")
            #         results['rte'] = 0.0
            
            # if 'wic' in self.benchmarks_to_run:
            #     try:
            #         print("Running WiC benchmark...")
            #         wic_benchmark = WiC(n_shots=3)
            #         wic_benchmark.evaluate(model=self.eval_model)
            #         results['wic'] = wic_benchmark.overall_score
            #         print(f"WiC Score: {wic_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"WiC evaluation failed: {e}")
            #         results['wic'] = 0.0
            
            # Run vision-based benchmark (MME)
            if 'mme' in self.benchmarks_to_run:
                try:
                    completed_benchmarks += 1
                    print(f"\n[{completed_benchmarks}/{total_benchmarks}] 🔍 Running MME benchmark...")
                    print(f"   Config: max_samples_per_task={self.mme_max_samples}")
                    import time
                    start_time = time.time()
                    mme_results = run_mme_evaluation(model, tokenizer, self.mme_max_samples)
                    elapsed = time.time() - start_time
                    if mme_results:
                        results.update(mme_results)
                        overall_score = mme_results.get('overall', 0)
                        print(f"   ✅ MME completed in {elapsed:.1f}s - Overall Score: {overall_score:.2f}%")
                except Exception as e:
                    print(f"   ❌ MME evaluation failed: {e}")
                    results['mme_overall'] = 0.0
            
            # Summary
            print("\n" + "-" * 60)
            print(f"📊 Benchmark Evaluation Summary ({completed_benchmarks}/{total_benchmarks} completed)")
            print("-" * 60)
            for metric_name, score in results.items():
                print(f"   {metric_name.upper()}: {score:.4f}")
            print("-" * 60)
            
            # Clear GPU cache after evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            print(f"\n❌ Benchmark evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # CRITICAL: Always restore original training state, even if evaluation failed
            try:
                if actual_model is not None and original_training_state is not None:
                    actual_model.train(original_training_state)
                    if original_training_state:
                        print("   🔧 Model restored to train mode (ready for training)")
                    else:
                        print("   🔧 Model kept in eval mode")
                elif trainer_model is not None:
                    # Fallback: try to restore trainer_model directly if actual_model is None
                    if hasattr(trainer_model, 'train'):
                        trainer_model.train()
                        print("   🔧 Trainer model restored to train mode (fallback)")
            except Exception as restore_error:
                print(f"   ⚠️ Warning: Failed to restore model training state: {restore_error}")
                # Try one more time with trainer.model directly
                try:
                    if hasattr(self.trainer, 'model') and hasattr(self.trainer.model, 'train'):
                        self.trainer.model.train()
                        print("   🔧 Trainer model restored to train mode (final fallback)")
                except:
                    pass
        
        return results
        
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # super().on_log(args, state, control, **kwargs)
        pass
        
    def _generate_table(self):
        super()._generate_table()
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_train_end(args, state, control, **kwargs)
        
        # Save benchmark results history
        if self.benchmark_results_history:
            results_file = os.path.join(args.output_dir, "benchmark_results.json")
            try:
                with open(results_file, 'w') as f:
                    json.dump(self.benchmark_results_history, f, indent=2)
                print(f"Benchmark results saved to {results_file}")
            except Exception as e:
                print(f"Failed to save benchmark results: {e}")
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_train_begin(args, state, control, **kwargs)


class SpecHornScheduler(TrainerCallback):
    """
    SpecHorn Scheduler: Dynamically adjusts hyperparameters based on real-time CV monitoring.
    
    - Monitors Coefficient of Variation (CV) at each training step
    - Adjusts cap_penalty_scale when CV deviates from target range
    - Gradually increases bias_scale and ortho_scale with training progress
    - Optional Wandb logging
    """
    
    def __init__(
        self,
        target_cv_min: float = 0.03,
        target_cv_max: float = 0.08,
        cap_penalty_min: float = 5.0,
        cap_penalty_max: float = 30.0,
        cap_penalty_step: float = 1.0,
        bias_scale_min: float = 4.0,
        bias_scale_max: float = 12.0,
        ortho_scale_min: float = 0.1,
        ortho_scale_max: float = 0.6,
        use_wandb: bool = False,
    ):
        super().__init__()
        self.target_cv_min = target_cv_min
        self.target_cv_max = target_cv_max
        self.cap_penalty_min = cap_penalty_min
        self.cap_penalty_max = cap_penalty_max
        self.cap_penalty_step = cap_penalty_step
        self.bias_scale_min = bias_scale_min
        self.bias_scale_max = bias_scale_max
        self.ortho_scale_min = ortho_scale_min
        self.ortho_scale_max = ortho_scale_max
        self.use_wandb = use_wandb
        
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Adjust hyperparameters based on real-time CV monitoring"""
        if model is None:
            return control
        
        # Only run on main process
        if not _is_main_process():
            return control
        
        # Get the router from the model
        try:
            # Navigate to the router (model structure may vary)
            if hasattr(model, 'module'):
                actual_model = model.module
            else:
                actual_model = model
            
            # Find router in the model
            router = None
            for name, module in actual_model.named_modules():
                if hasattr(module, '_is_seqorth_router'):
                    router = module
                    break
            
            if router is None:
                return control
            
            # Get current CV from router's load_ema
            with torch.no_grad():
                load_sum = router.load_ema.sum() + 1e-8
                normalized_load = router.load_ema / load_sum
                mean_load = normalized_load.mean()
                std_load = normalized_load.std()
                current_cv = (std_load / (mean_load + 1e-8)).item()
            
            # Adjust cap_penalty_scale based on CV
            if current_cv > self.target_cv_max:
                # CV too high, increase penalty
                new_penalty = min(
                    router.cap_penalty_scale + self.cap_penalty_step,
                    self.cap_penalty_max
                )
                router.cap_penalty_scale = new_penalty
            elif current_cv < self.target_cv_min:
                # CV too low, decrease penalty
                new_penalty = max(
                    router.cap_penalty_scale - self.cap_penalty_step,
                    self.cap_penalty_min
                )
                router.cap_penalty_scale = new_penalty
            
            # Progressive scaling of bias_scale and ortho_scale
            max_steps = state.max_steps if state.max_steps > 0 else 10000
            progress = min(1.0, state.global_step / max_steps)
            
            # bias_scale: gradually increase from min to max
            router.bias_scale = self.bias_scale_min + (self.bias_scale_max - self.bias_scale_min) * progress
            
            # ortho_scale: gradually increase from min to max
            router.ortho_scale = self.ortho_scale_min + (self.ortho_scale_max - self.ortho_scale_min) * progress
            
            # Log to Wandb if enabled
            if self.use_wandb and state.global_step % args.logging_steps == 0:
                try:
                    import wandb
                    wandb.log({
                        "spechorn/cv": current_cv,
                        "spechorn/cap_penalty_scale": router.cap_penalty_scale,
                        "spechorn/bias_scale": router.bias_scale,
                        "spechorn/ortho_scale": router.ortho_scale,
                        "spechorn/progress": progress,
                        "step": state.global_step,
                    })
                except ImportError:
                    pass
            
            # Log to console periodically
            if state.global_step % (args.logging_steps * 10) == 0:
                print(f"\n[SpecHorn Scheduler] Step {state.global_step}:")
                print(f"  CV: {current_cv:.4f} (target: {self.target_cv_min:.2f}-{self.target_cv_max:.2f})")
                print(f"  cap_penalty_scale: {router.cap_penalty_scale:.2f}")
                print(f"  bias_scale: {router.bias_scale:.2f}")
                print(f"  ortho_scale: {router.ortho_scale:.4f}")
        
        except Exception as e:
            print(f"[SpecHornScheduler] Warning: Failed to adjust hyperparameters: {e}")
        
        return control
