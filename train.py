import pandas as pd
from transformers import (
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorWithPadding,
    TextClassificationPipeline,
    EarlyStoppingCallback
)
import optuna
from transformers import TrainerCallback
from sklearn.utils import shuffle
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import label_ranking_average_precision_score, classification_report
import tqdm, random, ast
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

labels = [
    "성차별",
    "연령차별",
    "혐오욕설",
    "clean",
    "기타혐오"
]

# df = pd.read_csv("processed_data.csv")
df = pd.read_csv("reclassified_data.csv")
df = df[["sentence", "one_hot_label_reclassified"]]

model_name = "beomi/kcbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_function(examples):
    # 문장을 토큰화 (패딩과 트런케이션 추가)
    tokenized_examples = tokenizer(
        examples["sentence"].tolist(),  # DataFrame의 컬럼을 리스트로 변환
        padding=True,  # 패딩 적용
        return_tensors="pt",  # PyTorch 텐서로 반환
    )

    # 라벨을 텐서로 변환
    labels = torch.tensor(
        [
            eval(label) if isinstance(label, str) else label
            for label in examples["one_hot_label_reclassified"].tolist()
        ],
        dtype=torch.float,
    )

    return {
        "input_ids": tokenized_examples["input_ids"],
        "attention_mask": tokenized_examples["attention_mask"],
        "labels": labels,
    }

def count_labels_per_dataset(df, label_column, label_names=[
    "성차별",
    "연령차별",
    "혐오욕설",
    "clean",
    "기타혐오"
]):
    # 각 라벨의 카운트를 저장할 딕셔너리 초기화
    label_counts = {label: 0 for label in label_names}
    
    # 각 행에서 라벨값을 확인하고 카운트
    for labels in df[label_column]:
        # 문자열로 저장된 경우 리스트로 변환
        if isinstance(labels, str):
            labels = ast.literal_eval(labels)  # 문자열을 리스트로 변환
        for idx, value in enumerate(labels):
            if value == 1:  # 라벨이 활성화된 경우
                label_counts[label_names[idx]] += 1
    
    return label_counts

# 1. 먼저 train과 나머지(temp)로 분할
# train_df, test_df = split_by_label_indices(df, "one_hot_label_reclassified", 0.2, 42)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=20
)

# 3. 데이터 크기 확인
logging.info(f"전체 데이터 크기: {len(df)}")
logging.info(f"학습 데이터 크기: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
logging.info(f"테스트 데이터 크기: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")


# logging.info(f"훈련셋 조금만\n{train_df.head()}")
train_label_counts = count_labels_per_dataset(train_df, "one_hot_label_reclassified")
logging.info(f"학습 데이터 라벨별 갯수:")
logging.info(f"{train_label_counts}\n")

# logging.info(f"테스트셋 조금만\n{test_df.head()}")
test_label_counts = count_labels_per_dataset(test_df, "one_hot_label_reclassified")
logging.info(f"테스트 데이터 라벨별 갯수:")
logging.info(f"{test_label_counts}\n")

# 4. 각 데이터셋에 대해 전처리 함수 적용
train_tokenized = preprocess_function(train_df)
test_tokenized = preprocess_function(test_df)


# 1. 먼저 토크나이징된 데이터를 Dataset 객체로 변환
def convert_to_dataset(tokenized_dict):
    return Dataset.from_dict(
        {
            "input_ids": tokenized_dict["input_ids"],
            "attention_mask": tokenized_dict["attention_mask"],
            "labels": tokenized_dict["labels"],
        }
    )


# 2. 각 분할된 데이터에 대해 Dataset 객체 생성
train_dataset = convert_to_dataset(train_tokenized)
test_dataset = convert_to_dataset(test_tokenized)

train_dataset.set_format(
    type="torch", columns=["input_ids", "labels", "attention_mask"]
)
test_dataset.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

num_labels = len(labels)  # Label 갯수

model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, problem_type="multi_label_classification"
)
model.config.id2label = {i: label for i, label in zip(range(num_labels), labels)}
model.config.label2id = {label: i for i, label in zip(range(num_labels), labels)}

logging.info(model.config.label2id)


def compute_metrics(x):
    return {
        "lrap": label_ranking_average_precision_score(x.label_ids, x.predictions),
    }


batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

model = model.to(device)


class CustomCallback(TrainerCallback):
    """Optuna와 Trainer를 연결하기 위한 콜백 클래스"""
    def __init__(self, trial):
        self.trial = trial
        self.best_score = 0

    def on_evaluate(self, args, state, control, **kwargs):
        # `Trainer`의 평가 결과를 사용하여 Optuna의 trial 업데이트
        metrics = kwargs["metrics"]
        score = metrics.get("eval_lrap", 0)  # LRAP 점수 사용
        if score > self.best_score:
            self.best_score = score
            self.trial.set_user_attr("best_score", self.best_score)

def objective(trial):
    # 튜닝할 하이퍼파라미터 정의
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-4)
    per_device_train_batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 8)
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)

    # TrainingArguments 업데이트
    args = TrainingArguments(
        output_dir="model_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="lrap",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=10,
        fp16=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=4,
    )

    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda x: {
            "lrap": label_ranking_average_precision_score(x.label_ids, x.predictions)
        },
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[CustomCallback(trial)],
    )

    # 학습 수행
    trainer.train()

    # 최고 점수 반환
    return trial.user_attrs.get("best_score", 0)

# Optuna 튜닝 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # 20회 시도

# 최적 하이퍼파라미터 출력
best_params = study.best_params
print("Best hyperparameters:", best_params)

# 최적의 하이퍼파라미터로 모델 재학습
args = TrainingArguments(
    output_dir="model_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=best_params["learning_rate"],
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"],
    num_train_epochs=best_params["num_train_epochs"],
    load_best_model_at_end=True,
    metric_for_best_model="lrap",
    greater_is_better=True,
    logging_strategy="steps",
    logging_steps=10,
    fp16=True,
    gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
    dataloader_num_workers=4,
)

# 최적의 하이퍼파라미터로 모델 재학습
# args = TrainingArguments(
#     output_dir="model_output",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=best_params["learning_rate"],
#     per_device_train_batch_size=best_params["batch_size"],
#     per_device_eval_batch_size=best_params["batch_size"],
#     num_train_epochs=best_params["num_train_epochs"],
#     load_best_model_at_end=True,
#     metric_for_best_model="lrap",
#     greater_is_better=True,
#     logging_strategy="steps",
#     logging_steps=10,
#     fp16=True,
#     gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
#     dataloader_num_workers=4,
# )

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda x: {
        "lrap": label_ranking_average_precision_score(x.label_ids, x.predictions)
    },
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("model_output_trainer")

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     compute_metrics=compute_metrics,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     callbacks=[
#         EarlyStoppingCallback(early_stopping_patience=5),  # Early Stopping 적용, 3번 연속 개선되지 않으면 중지
#     ],
# )

# trainer.train()
# trainer.save_model()

# PyTorch 형식으로 추가 저장
model_save_path = "model_output/pytorch_model"
torch.save(
    {"model_state_dict": model.state_dict(), "config": model.config},
    model_save_path + ".pt",
)

# 평가
pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0,
    return_all_scores=True,
    function_to_apply="sigmoid",
)


def get_predicated_label(output_labels, min_score):
    labels = []
    for label in output_labels:
        if label["score"] > min_score:
            labels.append(1)
        else:
            labels.append(0)
    return labels


# 2. 예측 임계값을 설정하는 함수
def get_predicted_labels(scores, threshold=0.5):
    # scores는 각 레이블에 대한 확률값
    return [1 if score >= threshold else 0 for score in scores]


# 3. 예측 수행
predicted_labels = []
true_labels = []

# tqdm으로 진행률 표시하면서 예측
for i in tqdm.tqdm(range(len(test_dataset))):
    # 문장 가져오기
    text = test_df.iloc[i]["sentence"]

    # 예측
    out = pipe(text)[0]  # 첫 번째 결과만 사용

    # 각 레이블의 점수 추출
    scores = [score["score"] for score in out]

    # 예측 레이블 생성
    pred_label = get_predicted_labels(scores)
    predicted_labels.append(pred_label)

    # 실제 레이블 저장
    true_label = test_dataset[i]["labels"].numpy()
    true_labels.append(true_label)

# 4. numpy 배열로 변환
predicted_labels = np.array(predicted_labels)
true_labels = np.array(true_labels)

# 6. 성능 평가 리포트 출력
logging.info(f"\n분류 성능 평가 리포트:\n{classification_report(true_labels, predicted_labels, target_names=labels, zero_division=0)}")

model.save_pretrained("model_output")
tokenizer.save_pretrained("model_output")