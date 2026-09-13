from pathlib import Path

import joblib
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
# Load dataset
# ============================================================

def load_data():
    """Load and combine fake and real news datasets."""

    print("Loading datasets...")

    fake = pd.read_csv(FAKE_DATA_PATH)
    true = pd.read_csv(TRUE_DATA_PATH)

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat(
        [fake, true],
        ignore_index=True
    )

    # Shuffle dataset
    data = data.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print(f"Total samples: {len(data)}")

    return data


# ============================================================
# Prepare text
# ============================================================

def prepare_text(data):
    """Combine relevant text columns and clean them."""

    # Your dataset contains title and text.
    data["content"] = (
        data["title"].fillna("")
        + " "
        + data["text"].fillna("")
    )

    print("Cleaning text...")

    data["cleaned_content"] = data["content"].apply(
        clean_text
    )

    return data


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model."""

    predictions = model.predict(X_test)

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

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Fake News",
                "Real News"
            ],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


# ============================================================
# Main training pipeline
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
    # 3. Train / test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # --------------------------------------------------------
    # 4. TF-IDF
    # --------------------------------------------------------

    print("\nCreating TF-IDF features...")

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
        f"TF-IDF feature shape: "
        f"{X_train_tfidf.shape}"
    )

    # --------------------------------------------------------
    # 5. Define models
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
    # 6. Train and evaluate
    # --------------------------------------------------------

    results = {}

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        model.fit(
            X_train_tfidf,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test_tfidf,
            y_test,
            model_name
        )

        results[model_name] = metrics

        # Save model
        filename = (
            model_name
            .lower()
            .replace(" ", "_")
        )

        model_path = (
            MODEL_DIR / f"{filename}.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"Model saved to: {model_path}"
        )

    # --------------------------------------------------------
    # 7. Save vectorizer
    # --------------------------------------------------------

    vectorizer_path = (
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )

    joblib.dump(
        vectorizer,
        vectorizer_path
    )

    print(
        f"\nVectorizer saved to: "
        f"{vectorizer_path}"
    )

    # --------------------------------------------------------
    # 8. Display comparison
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(results).T

    print(results_df)

    # --------------------------------------------------------
    # 9. Find best model
    # --------------------------------------------------------

    best_model = results_df[
        "f1_score"
    ].idxmax()

    print("\nBest model:")
    print(best_model)

    print(
        f"Best F1 Score: "
        f"{results_df.loc[best_model, 'f1_score']:.4f}"
    )

    return results


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    train_models()