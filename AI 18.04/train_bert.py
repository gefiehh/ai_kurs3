import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

MODEL_NAME = "DeepPavlov/rubert-base-cased"
DATASET_PATH = "dataset.csv"
OUTPUT_DIR = "./intent_bert_model"
NUM_LABELS = 8                    
MAX_LENGTH = 64
EPOCHS = 10
BATCH_SIZE = 16

print("Загружаем датасет...")
df = pd.read_csv(DATASET_PATH, encoding="windows-1251")

intent_list = ["greeting", "farewell", "howareyou", "time", "date", "set_name", "addition", "weather", "help", "smalltalk", "thanks", "unknown"]
label2id = {label: idx for idx, label in enumerate(intent_list)}
id2label = {idx: label for label, idx in label2id.items()}

df["label"] = df["intent"].map(label2id)

train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=50,
    fp16=True,
    logging_steps=10,
    report_to="none",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Начинаем fine-tuning RuBERT...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Обучение завершено! Модель сохранена в: {OUTPUT_DIR}")