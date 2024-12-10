from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TextClassificationPipeline,
)
import torch

# 레이블 정의
labels = [
        "성차별",
        "연령차별",
        "혐오욕설",
        "clean",
        "기타혐오"
    ]

def load_model(model_path, tokenizer_name="beomi/kcbert-base"):
    """
    # 모델과 토크나이저를 로드하는 함수
    """


    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 모델 설정
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,  # 사이즈 불일치 무시
    )

    # 레이블 매핑 설정
    model.config.id2label = {i: label for i, label in enumerate(labels)}
    model.config.label2id = {label: i for i, label in enumerate(labels)}

    return model, tokenizer


def create_pipeline(model, tokenizer):
    """
    # 추론 파이프라인 생성 함수
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=True,
        function_to_apply="sigmoid",
    )
    return pipe


def predict_text(pipe, text, threshold=0.5):
    """
    # 텍스트 예측 및 결과 출력 함수
    """
    results = pipe(text)[0]

    print("\n=== 분석 결과 ===")
    print(f"입력 문장: {text}\n")
    print("카테고리별 결과:")

    labels = []
    for result in results:
        label = result["label"]
        score = result["score"]
        print(f"{label}: {score:.3f}")

        if score >= threshold:
            labels.append(label)

    print("\n탐지된 카테고리:")
    if labels:
        for category in labels:
            print(f"- {category}")
    else:
        print("- 혐오 표현이 탐지되지 않았습니다.")
    print("================\n")
    print(results)


def main():
    # 모델 경로 설정
    MODEL_PATH = "./new_model_trainer"  # 학습된 모델이 저장된 경로



    print("모델을 로딩중입니다...")
    model, tokenizer = load_model(MODEL_PATH)
    pipe = create_pipeline(model, tokenizer)
    print("모델 로딩이 완료되었습니다!")

    print("\n혐오 표현 탐지 시스템에 오신 것을 환영합니다!")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")

    while True:
        text = input("\n분석할 문장을 입력하세요: ")

        if text.lower() in ["quit", "exit"]:
            print("프로그램을 종료합니다.")
            break

        if not text.strip():
            print("문장을 입력해주세요!")
            continue

        try:
            predict_text(pipe, text)
        except Exception as e:
            print(f"에러가 발생했습니다: {e}")
            continue


if __name__ == "__main__":
    main()
