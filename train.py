import json
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def load_data():
  X, y = load_breast_cancer(return_X_y=True, as_frame=True)
  return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_model(X_train, y_train):
  model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
  model.fit(X_train, y_train)
  return model

if __name__ == "__main__":
  X_train, X_test, y_train, y_test = load_data()
  model = train_model(X_train, y_train)
  accuracy = accuracy_score(y_test, model.predict(X_test))
  print(f"Accuracy: {accuracy}")
  joblib.dump(model, "model.joblib")
  with open("metrics.json", "w") as f:
    json.dump({"accuracy": round(accuracy, 4)}, f)
  
