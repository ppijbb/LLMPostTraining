from datasets import Dataset
from unsloth.chat_templates import standardize_data_formats

def load_and_preprocess_data(jsonl_path, tokenizer):
    """
    JSONL 파일을 로드하고 전처리하는 함수
    """
    # 데이터셋 로드
    dataset = Dataset.from_json(jsonl_path)
    
    # 데이터 형식 표준화
    dataset = standardize_data_formats(dataset)
    
    def apply_chat_template(examples):
        texts = tokenizer.apply_chat_template(examples["conversations"])
        return {"text": texts}
    
    # 채팅 템플릿 적용
    dataset = dataset.map(apply_chat_template, batched=True)
    
    return dataset 


