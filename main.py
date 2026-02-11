import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. تجهيز بيانات تجريبية (يمكنك استبدالها ببيانات من Kaggle لاحقاً)
data = {
    'question': [
        'What are the side effects of Aspirin?',
        'How can I prevent the flu?',
        'What is the dosage for Paracetamol?',
        'How to reduce high blood pressure?',
        'Is dizziness a symptom of COVID-19?'
    ],
    'category': ['Medication', 'Prevention', 'Medication', 'Prevention', 'Symptoms']
}

df = pd.DataFrame(data)

# 2. تقسيم البيانات لتدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(df['question'], df['category'], test_size=0.2, random_state=42)

# 3. بناء Pipeline (تحويل النص إلى أرقام + خوارزمية التصنيف)
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# 4. تدريب النموذج
model.fit(X_train, y_train)

# 5. تجربة النموذج على سؤال جديد
new_question = ["What should I take for a headache?"]
prediction = model.predict(new_question)
print(f"Question: {new_question[0]} \nPredicted Category: {prediction[0]}")
