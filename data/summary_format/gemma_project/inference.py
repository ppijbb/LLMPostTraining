from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import torch

def load_inference_model(model_path):
    """
    인퍼런스용 모델 로드 함수
    Args:
        model_path: 학습된 모델이 저장된 경로
    """
    # 모델 및 토크나이저 로드
    model, tokenizer = FastModel.from_pretrained(
        model_path,
        max_seq_length = 4096,
        load_in_4bit = True,
        load_in_8bit = False,
        full_finetuning = False,
    )
    
    # 채팅 템플릿 설정
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    """
    주어진 프롬프트에 대한 응답을 생성하는 함수
    """
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": prompt
        }]
    }]
    
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
    )
    
    outputs = model.generate(
        **tokenizer([text], return_tensors = "pt").to("cuda"),
        max_new_tokens = 1024,
        temperature = 1.0,
        top_p = 0.95,
        top_k = 64,
    )
    
    return tokenizer.batch_decode(outputs)

if __name__ == "__main__":
    # 사용할 모델 경로 지정
    model_path = "saved_models/model_20250421_131823"
    
    # 모델 로드
    model, tokenizer = load_inference_model(model_path)
    
#     # 테스트 프롬프트
    test_prompt = '''
    다음은 의사와 환자의 진료 대화입니다. markdown table 구조의 포맷 형식으로 정리하고 정리해야할 항목 종류는 진단명, 총비용입니다. 지정된 항목명에 대한 것만 생성하고 대화내용에 항목명의 내용이 없으면 언급없음이라고 생성하세요. 반드시 지정한 포맷으로 생성해주세요.
    진료 대화:
    의사: 유재석 환자분 맞으신가요? 생년월일은 몇 년 몇 월 몇 일이신가요?
    환자: 1972년 8월 14일이에요.
    의사: 네 감사합니다. 어디가 불편하신가요?
    환자: 선생님, 오른쪽 위 어금니가 계속 욱신거리고, 잇몸에서도 피가 나요. 한 2주 전부터 점점 심해졌어요.
    의사: 통증은 하루 종일 지속되나요, 아니면 특정한 상황에서만 느껴지시나요?
    환자: 딱딱한 걸 씹을 때랑, 밤에 잘 때 특히 아파요. 낮에는 조금 괜찮은데요.
    의사: 잇몸 출혈은 양치할 때만 나시나요, 아니면 평소에도 피가 나나요?
    환자: 양치할 때 피가 나고, 가끔 침에 피가 섞여 있을 때도 있어요.
    의사: 이전에 같은 부위에 치료받으신 적 있으세요? 충치 치료나 신경치료 같은 거요.
    환자: 작년에 충치 치료는 했었고, 신경치료는 안 했어요. 그때는 괜찮았는데, 요즘 갑자기 이러네요.
    의사: 방금 구강 내 검사해보니, 해당 부위 잇몸이 많이 부어 있고, 치석도 꽤 쌓여 있어요. 그리고 어금니 안쪽에 크랙(치아 균열) 가능성도 보여요.
    환자: 크랙이면 어떻게 해야 하나요?
    의사: 정확한 판단을 위해 엑스레이랑 CT 촬영을 먼저 해볼게요. 크랙이 깊다면 발치 후 임플란트까지 고려할 수 있고, 그렇지 않다면 보철로 마무리할 수 있어요. 그리고 잇몸도 치주치료가 필요한 상태입니다. 스케일링과 함께 잇몸 염증 관리도 같이 들어갈 예정입니다.
    환자: 네, 그렇게 진행해주세요.
    의사: 비용은 안내 받으시겠지만 엑스레이랑 CT 촬영은 보험이 어렵고 치주치료는 보험에 포함되어 있어요. 대략 30만원정도 생각하시면 됩니다.
    환자: 네, 그렇게 진행해주세요.
    '''
    
    # 응답 생성
    response = generate_response(model, tokenizer, test_prompt)
    print("Generated Response:", response) 