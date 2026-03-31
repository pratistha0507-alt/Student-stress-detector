import joblib
import argparse

# Loading the model I trained
model = joblib.load('models/stress_model.pkl')

parser = argparse.ArgumentParser(description="Check your stress level")
parser.add_argument('--sleep', type=float, required=True, help="Sleep hours")
parser.add_argument('--study', type=float, required=True, help="Study hours")
parser.add_argument('--screen', type=float, required=True, help="Screen time hours")
parser.add_argument('--caffeine', type=float, required=True, help="Caffeine cups")
parser.add_argument('--exercise', type=float, required=True, help="Exercise hours")
parser.add_argument('--social', type=float, default=2.0, help="Social hours")

args = parser.parse_args()

features = [[args.sleep, args.study, args.screen, args.caffeine, args.exercise, args.social]]
prediction = model.predict(features)[0]
confidence = model.predict_proba(features).max() * 100

levels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
print(f"Your predicted stress level is: {levels[prediction]}")
print(f"Confidence: {confidence:.1f}%")
