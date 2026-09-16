from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FAKE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Fake.csv"
TRUE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "True.csv"


def load_data():
    """Load and combine fake and real news datasets."""

    fake = pd.read_csv(FAKE_DATA_PATH)
    true = pd.read_csv(TRUE_DATA_PATH)

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

    return data