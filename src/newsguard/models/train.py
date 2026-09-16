from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from newsguard.features.preprocessing import clean_text


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FAKE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Fake.csv"
TRUE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "True.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ============================================================
# MLflow configuration
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "NewsGuard AI - Docker"

mlflow.set_experiment(EXPERIMENT_NAME)

MODEL_NAME = "NewsGuard-Fake-News-Classifier"
CHAMPION_ALIAS = "champion"

mlflow_client = MlflowClient()


# ============================================================
# Load data
# ============================================================

print("Loading datasets...")

fake = pd.read_csv(FAKE_DATA_PATH)
true = pd.read_csv(TRUE_DATA_PATH)

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)

data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# Prepare text
# ============================================================

data["content"] = (
    data["title"].fillna("")
    + " "
    + data["text"].fillna("")
)

data["content"] = data["content"].apply(clean_text)

X = data["content"]
y = data["label"]


# ============================================================
# Train / test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ============================================================
# Models
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),

    "Multinomial Naive Bayes": MultinomialNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        random_state=42,
        n_jobs=-1,
    ),
}


# ============================================================
# Train models
# ============================================================

results = []


for model_name, classifier in models.items():

    print(f"\nTraining {model_name}...")

    # Complete ML pipeline
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000
            ),
        ),
        (
            "classifier",
            classifier,
        ),
    ])

    with mlflow.start_run(run_name=model_name):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        pipeline.fit(X_train, y_train)

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        y_pred = pipeline.predict(X_test)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model_name",
            model_name,
        )

        mlflow.log_param(
            "test_size",
            0.2,
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        mlflow.log_param(
            "tfidf_max_features",
            10000,
        )

        # ----------------------------------------------------
        # Log model parameters
        # ----------------------------------------------------

        for param_name, param_value in classifier.get_params().items():

            # Avoid logging None values
            if param_value is not None:
                mlflow.log_param(
                    param_name,
                    param_value,
                )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "f1_score",
            f1,
        )

        # ----------------------------------------------------
        # Save classification report
        # ----------------------------------------------------

        report_path = (
            MODELS_DIR
            / f"{model_name.replace(' ', '_')}_classification_report.txt"
        )

        from sklearn.metrics import classification_report

        report = classification_report(
            y_test,
            y_pred,
            target_names=["Fake", "True"],
        )

        report_path.write_text(
            report,
            encoding="utf-8",
        )

        mlflow.log_artifact(
            str(report_path)
        )


        # ----------------------------------------------------
        # Log COMPLETE pipeline to MLflow
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
        )

        # ----------------------------------------------------
        # Save local copy
        # ----------------------------------------------------

        local_model_path = (
            MODELS_DIR
            / f"{model_name.replace(' ', '_')}_pipeline.pkl"
        )

        joblib.dump(
            pipeline,
            local_model_path,
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "run_id": mlflow.active_run().info.run_id,
        })

        print(
            f"Accuracy : {accuracy:.6f}"
        )
        print(
            f"Precision: {precision:.6f}"
        )
        print(
            f"Recall   : {recall:.6f}"
        )
        print(
            f"F1 Score : {f1:.6f}"
        )


# ============================================================
# Compare models
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False,
)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    results_df.to_string(
        index=False
    )
)

print("\nBest model:")

print(
    results_df.iloc[0]
)

# ============================================================
# Automatic Model Registration
# ============================================================

best_model = results_df.iloc[0]

best_run_id = best_model["run_id"]
best_f1 = float(best_model["f1_score"])
best_model_name = best_model["model"]

print("\n==============================")
print("AUTOMATIC MODEL REGISTRATION")
print("==============================")

print(f"Best model : {best_model_name}")
print(f"F1 Score   : {best_f1:.6f}")
print(f"Run ID     : {best_run_id}")


# ------------------------------------------------------------
# Check current champion
# ------------------------------------------------------------

try:

    champion_model = mlflow_client.get_model_version_by_alias(
        MODEL_NAME,
        CHAMPION_ALIAS,
    )

    champion_run = mlflow_client.get_run(
        champion_model.run_id
    )

    champion_f1 = champion_run.data.metrics.get(
        "f1_score",
        0.0,
    )

    print("\nCurrent champion:")
    print(f"Version    : {champion_model.version}")
    print(f"F1 Score   : {champion_f1:.6f}")

except Exception:

    champion_model = None
    champion_f1 = 0.0

    print("\nNo existing champion model found.")


# ------------------------------------------------------------
# Promote only if better
# ------------------------------------------------------------

if best_f1 > champion_f1:

    print("\nNew model is better.")
    print("Registering new model...")

    model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    new_version = registered_model.version

    print(
        f"Registered version: {new_version}"
    )

    # Assign champion alias
    mlflow_client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=new_version,
    )

    print(
        f"Champion → Version {new_version}"
    )

else:

    print("\nCurrent champion is better or equal.")
    print("Keeping the current champion.")
