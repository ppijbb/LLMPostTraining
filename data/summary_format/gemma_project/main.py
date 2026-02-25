import os
from model import load_model_and_tokenizer
from data_processor import load_and_preprocess_data
from trainer import setup_trainer
from datetime import datetime
import numpy
import wandb 

def main():
    # 모델과 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer()

    # 모델 이름 생성 (한 번만 생성)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"model_{current_time}"
    
    # 데이터셋 로드 및 전처리
    dataset = load_and_preprocess_data("/user/glory/summary/gemma_project/fine_tome_all.jsonl", tokenizer)
    
    # 학습 설정 - model_name 전달
    trainer = setup_trainer(model, tokenizer, dataset, model_name)
    
    # 학습 실행
    trainer_stats = trainer.train()
    
    # 모델 저장 경로 구성 (model_name 재사용)
    model_save_dir = "/user/glory/summary/gemma_project/saved_models" 
    model_save_path = os.path.join(model_save_dir, model_name)
    
    # 디렉토리가 없으면 생성
    os.makedirs(model_save_path, exist_ok=True)
    # 모델 저장
    print(f"\n모델을 {model_save_path}에 저장합니다...")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # wandb에 모델 아티팩트 저장
    wandb.log({"model_path": model_save_path})
    
    # 최종 메트릭 로깅
    final_metrics = {
        "train_loss": trainer_stats.training_loss,
        "train_runtime": trainer_stats.train_runtime,
        "train_samples_per_second": trainer_stats.train_samples_per_second
    }
    wandb.log(final_metrics)
    
    # wandb 종료
    wandb.finish()
    
    print("모델 저장 완료!")
    print(f"인퍼런스 시 사용할 모델 경로: {model_save_path}")

if __name__ == "__main__":
    main()
 
