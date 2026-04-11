import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Загружаем модель с векторами
print("Загружаю spaCy модель ru_core_news_md...")
nlp = spacy.load("ru_core_news_md")

def sentence_vector(text):
    doc = nlp(text.lower())
    return doc.vector

print("Загружаю датасет...")
df = pd.read_csv("dataset.csv", encoding="windows-1251")

print("Вычисляю эмбеддинги...")
X = np.array([sentence_vector(text) for text in df['text']])
y = df['intent'].values

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Обучаю LogisticRegression на word embeddings...")
model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs',     
    n_jobs=-1           
)

model.fit(X_train, y_train)

# Оценка качества
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Сохраняем модель
joblib.dump(model, "embedding_model.pkl")
print("\nМодель успешно обучена и сохранена: embedding_model.pkl")