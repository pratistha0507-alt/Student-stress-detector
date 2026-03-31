import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("Starting training...")

df = pd.read_csv('data/student_stress_data.csv')

X = df[['sleep_hours', 'study_hours', 'screen_time_hours', 'caffeine_cups', 'exercise_hours', 'social_hours']]
y = df['stress_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_model.predict(X_test)))

os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/stress_model.pkl')
print("Model trained and saved successfully!")
