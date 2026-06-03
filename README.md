# Heart Attack Prediction Using Machine Learning

## Overview

This project predicts the likelihood of a heart attack using patient medical data and Machine Learning. A **Logistic Regression** model was trained on a heart attack dataset containing clinical measurements such as age, blood pressure, blood sugar levels, CK-MB, and Troponin.

The trained model achieves approximately **80% accuracy** on the test dataset and can be deployed for real-time predictions using Flask, Streamlit, or other web frameworks.

---

## Dataset

Dataset Source:

* Heart Attack Dataset by Tarik A. Rashid

The dataset contains **1319 patient records** with the following features:

| Feature                  | Description                                 |
| ------------------------ | ------------------------------------------- |
| Age                      | Patient age                                 |
| Gender                   | Gender (0 = Female, 1 = Male)               |
| Heart rate               | Heart rate (BPM)                            |
| Systolic blood pressure  | Upper blood pressure reading                |
| Diastolic blood pressure | Lower blood pressure reading                |
| Blood sugar              | Blood glucose level                         |
| CK-MB                    | Creatine Kinase-MB enzyme level             |
| Troponin                 | Troponin protein level                      |
| Result                   | Heart attack prediction (Positive/Negative) |

---

## Project Workflow

### 1. Data Collection

Dataset downloaded using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download(
    "fatemehmohammadinia/heart-attack-dataset-tarik-a-rashid"
)
```

### 2. Data Preprocessing

* Loaded dataset using Pandas
* Removed missing values
* Checked and removed duplicates
* Encoded target labels using LabelEncoder
* Standardized numerical features
* One-Hot Encoded categorical features

### 3. Feature Engineering

Numerical Features:

* Age
* Heart rate
* Systolic blood pressure
* Diastolic blood pressure
* Blood sugar
* CK-MB
* Troponin

Categorical Feature:

* Gender

### 4. Data Splitting

```python
train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)
```

* Training Data: 80%
* Testing Data: 20%

### 5. Model Training

Algorithm Used:

* Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)
model.fit(X_train_processed, y_train)
```

---

## Model Performance

### Accuracy Score

```text
Model Accuracy: 0.7992
```

### Dataset Distribution

| Class    | Count |
| -------- | ----- |
| Positive | 810   |
| Negative | 509   |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* KaggleHub

---

## Project Structure

```text
Heart-Attack-Prediction/
│
├── heart_attack_prediction.ipynb
├── unniappam.pkl
├── requirements.txt
├── README.md
└── dataset/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/heart-attack-prediction.git

cd heart-attack-prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

```text
numpy
pandas
scikit-learn
joblib
kagglehub
```

Install manually:

```bash
pip install numpy pandas scikit-learn joblib kagglehub
```

---

## Saving the Model

The trained model is saved using Joblib:

```python
import joblib

joblib.dump(model, "unniappam.pkl")
```

Load the model:

```python
model = joblib.load("unniappam.pkl")
```

---

## Sample Prediction

```python
sample_data = [[
    55,
    1,
    70,
    140,
    80,
    150,
    2.5,
    0.04
]]

prediction = model.predict(sample_data)
print(prediction)
```

Output:

```text
[1]
```

Where:

* 1 → Positive
* 0 → Negative

---

## Future Improvements

* Deploy using Flask or Streamlit
* Compare multiple ML algorithms
* Hyperparameter tuning
* Feature selection
* Deep Learning implementation
* Real-time patient risk assessment

---

## Results

The Logistic Regression model successfully predicts heart attack risk using clinical parameters and achieved an accuracy of approximately **80%** on unseen test data.

---

