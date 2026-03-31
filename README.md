# Student Stress Level Detector

Hey everyone!  

I made this small ML project because I saw so many first-year students (including me) getting stressed in college or even in school. Late-night submissions, hostel mess, phone addiction, and no proper sleep — everything adds up. This tool takes your daily habits and tells you whether your stress level is Low, Medium or High.

It’s my BYOP Capstone for the Fundamentals of AI and ML course.

## Why I built this
In first semester I was always worried about my CGPA and felt tired all the time. I wanted to understand what actually affects my mood and energy. So I created this simple predictor using the concepts we learned in class.

## How to use it 

### Step 1: Clone the repo
```bash
git clone https://github.com/pratistha0507-alt/Student-stress-detector.git
cd Student-stress-detector
```
### Step 2: Install packages
```bash
pip install -r requirements.txt
```
### Step 3: Train the model (do this only once)
```bash
python src/train.py
```
### Step 4: Check your stress level
```bash
python src/predict.py --sleep 6 --study 7 --screen 8 --caffeine 3 --exercise 1
```
You will get output like:
```bash
Your predicted stress level is: HIGH
Confidence: 82%
```
### What I used
Python 3
pandas
scikit-learn (Random Forest and Logistic Regression)
joblib
