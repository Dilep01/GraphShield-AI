import csv
import json

import pytest

from graphshield.pipeline import HEADERS


@pytest.fixture
def project(tmp_path):
    """Pytest base temp is explicitly set inside the real project by test.ps1."""
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/phase1.json").write_text(json.dumps({
        "dataset_id": "test-fixture-not-paysim", "train_fraction": 0.5,
        "validation_fraction": 0.25, "memory_limit": "256MB", "threads": 1,
        "compute_components": True,
    }), encoding="utf-8")
    return tmp_path


@pytest.fixture
def write_csv(project):
    def write(rows):
        path = project / "data/raw/fixture.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADERS)
            writer.writerows(rows)
        return path
    return write


@pytest.fixture
def rows():
    return [
        [1, "TRANSFER", "10.00", "A", 20, 10, "B", 0, 10, 0, 0],
        [2, "TRANSFER", "10.00", "B", 10, 0, "C", 0, 10, 1, 0],
        [2, "TRANSFER", "10.00", "B", 10, 0, "C", 0, 10, 1, 0],
        [3, "PAYMENT", "0.00", "A", 10, 10, "M", 0, 0, 0, 0],
        [4, "CASH_OUT", "5.00", "D", 5, 0, "C", 10, 15, 0, 0],
        [5, "TRANSFER", "9.00", "E", 9, 0, "F", 0, 9, 0, 0],
        [6, "TRANSFER", "8.00", "F", 9, 1, "G", 0, 8, 1, 0],
    ]
