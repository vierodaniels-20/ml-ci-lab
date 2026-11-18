
import numpy as np
from sklearn.metrics import accuracy_score

from train import load_data, train_model

MIN_ACCURACY = 0.95

def test_data_shape():
  X_train, X_test, _, _ = load.data()
  assert X_train.shape{1} == 30
  assert len(X_train) + len(X_test) == 569

def test_no_missing_values():
  X_train, X_test, _, _ = load_data()
  assert not X_train.isna().any().any()
  assert not X_test.isna().any().any()

def test_prediction_shape():
  X_train, X_test, y_train, _= load_data()
  preds = train_model(X_train, y_train).predict(X_test)
  assert preds.shape == (len(X_test),)
  assert set(np.unique(preds)) <= {0,1}

def test_training_is_reproducible():
  X_train, X_test, y_train, _ = load_data()
  model_1 = train_model(X_train, y_train).predict(X_test)
  model_2 = train_model(X_train, y_train).predict(X_test)
  assert np.array_equal(model_1, model_2)

def test_accuracy_above_treshold():
  X_train, X_test, y_train, _ = load_data()
  model = train_model(X_train, y_train)
  preds = model.predict(X_test)
  accuracy = accuracy_score(y_test, preds)
  assert accuracy >= MIN_ACCURACY
