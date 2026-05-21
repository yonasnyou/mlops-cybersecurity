import mlflow
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from preprocessing import load_data, preprocess

mlflow.set_experiment("Cybersecurity_MLOps")

df = load_data("data/raw/KDDTrain+.txt")

X_train, X_test, y_train, y_test = preprocess(df)

models = {
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier()
}

best_model = None
best_score = 0

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_metric("accuracy", accuracy)

        print(f"{name}: {accuracy}")

        if accuracy > best_score:
            best_score = accuracy
            best_model = model

joblib.dump(best_model, "models/model.pkl")