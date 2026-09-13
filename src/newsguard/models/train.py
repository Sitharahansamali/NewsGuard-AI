from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from newsguard.features.preprocessing import clean_text


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FAKE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Fake.csv"
TRUE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "True.csv"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MLflow Configuration
# ============================================================

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

EXPERIMENT_NAME = (
    "NewsGuard AI - Fake News Classification"
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# Load Dataset
# ============================================================

def load_data():

    print("Loading datasets...")

    fake = pd.read_csv(
        FAKE_DATA_PATH
    )

    true = pd.read_csv(
        TRUE_DATA_PATH
    )

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat(
        [fake, true],
        ignore_index=True
    )

    data = data.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print(
        f"Total samples: {len(data)}"
    )

    return data


# ============================================================
# Prepare Text
# ============================================================

def prepare_text(data):

    data["content"] = (
        data["title"].fillna("")
        + " "
        + data["text"].fillna("")
    )

    print("Cleaning text...")

    data["cleaned_content"] = (
        data["content"].apply(clean_text)
    )

    return data


# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Fake News",
            "Real News"
        ],
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
    }


# ============================================================
# Main Training Pipeline
# ============================================================

def train_models():

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    data = load_data()

    # --------------------------------------------------------
    # 2. Prepare text
    # --------------------------------------------------------

    data = prepare_text(data)

    X = data["cleaned_content"]

    y = data["label"]

    # --------------------------------------------------------
    # 3. Train / Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # 4. TF-IDF
    # --------------------------------------------------------

    print(
        "\nCreating TF-IDF features..."
    )

    vectorizer = TfidfVectorizer(
        max_features=10000
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    print(
        f"TF-IDF shape: {X_train_tfidf.shape}"
    )

    # --------------------------------------------------------
    # 5. Models
    # --------------------------------------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Multinomial Naive Bayes":
            MultinomialNB(),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),
    }

    # --------------------------------------------------------
    # 6. Train each model
    # --------------------------------------------------------

    all_results = {}

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        # ----------------------------------------------------
        # Start MLflow Run
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name=model_name
        ):

            # ------------------------------------------------
            # Parameters
            # ------------------------------------------------

            mlflow.log_param(
                "model_name",
                model_name
            )

            mlflow.log_param(
                "test_size",
                0.2
            )

            mlflow.log_param(
                "random_state",
                42
            )

            mlflow.log_param(
                "tfidf_max_features",
                10000
            )

            # Log model-specific parameters
            model_params = model.get_params()

            for parameter, value in model_params.items():

                mlflow.log_param(
                    parameter,
                    value
                )

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            model.fit(
                X_train_tfidf,
                y_train
            )

            # ------------------------------------------------
            # Evaluate
            # ------------------------------------------------

            metrics = evaluate_model(
                model,
                X_test_tfidf,
                y_test,
                model_name
            )

            # ------------------------------------------------
            # Log Metrics
            # ------------------------------------------------

            mlflow.log_metric(
                "accuracy",
                metrics["accuracy"]
            )

            mlflow.log_metric(
                "precision",
                metrics["precision"]
            )

            mlflow.log_metric(
                "recall",
                metrics["recall"]
            )

            mlflow.log_metric(
                "f1_score",
                metrics["f1_score"]
            )

            # ------------------------------------------------
            # Save classification report
            # ------------------------------------------------

            report_path = (
                MODEL_DIR
                / f"{model_name}_classification_report.txt"
            )

            with open(
                report_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    metrics["classification_report"]
                )

            # ------------------------------------------------
            # Log Classification Report
            # ------------------------------------------------

            mlflow.log_artifact(
                report_path
            )

            # ------------------------------------------------
            # Log trained model
            # ------------------------------------------------

            mlflow.sklearn.log_model(
                model,
                name="model"
            )

            # ------------------------------------------------
            # Save model locally
            # ------------------------------------------------

            filename = (
                model_name
                .lower()
                .replace(" ", "_")
            )

            model_path = (
                MODEL_DIR
                / f"{filename}.pkl"
            )

            joblib.dump(
                model,
                model_path
            )

            print(
                f"Model saved to: {model_path}"
            )

            # ------------------------------------------------
            # Log run information
            # ------------------------------------------------

            run_id = mlflow.active_run().info.run_id

            print(
                f"MLflow Run ID: {run_id}"
            )

            all_results[model_name] = {
                "run_id": run_id,
                "metrics": metrics
            }

    # --------------------------------------------------------
    # Save vectorizer
    # --------------------------------------------------------

    vectorizer_path = (
        MODEL_DIR
        / "tfidf_vectorizer.pkl"
    )

    joblib.dump(
        vectorizer,
        vectorizer_path
    )

    print(
        f"\nVectorizer saved to: {vectorizer_path}"
    )

    # --------------------------------------------------------
    # Model Comparison
    # --------------------------------------------------------

    comparison = {}

    for model_name, result in all_results.items():

        comparison[model_name] = {
            "accuracy":
                result["metrics"]["accuracy"],

            "precision":
                result["metrics"]["precision"],

            "recall":
                result["metrics"]["recall"],

            "f1_score":
                result["metrics"]["f1_score"],
        }

    results_df = pd.DataFrame(
        comparison
    ).T

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(results_df)

    best_model = results_df[
        "f1_score"
    ].idxmax()

    print("\nBest model:")
    print(best_model)

    print(
        f"Best F1 Score: "
        f"{results_df.loc[best_model, 'f1_score']:.4f}"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    train_models()