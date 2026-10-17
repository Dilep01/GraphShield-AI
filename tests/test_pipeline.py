import json

import duckdb
import pytest

from graphshield.paths import inside_root
from graphshield.pipeline import component_summary, history_before, run_pipeline, split_boundaries


def test_temporal_splits_never_shuffle_or_split_a_step():
    assert split_boundaries([4, 1, 1, 3, 2, 6, 5], .5, .25) == (3, 4)


def test_scientific_notation_preserves_exact_amount(project, write_csv, rows):
    rows[0][2] = '1.0E7'
    path = write_csv(rows)
    run_pipeline(path, 'config/phase1.json', 'scientific', root=project)
    with duckdb.connect() as con:
        amount = con.execute('SELECT amount FROM read_parquet(?) WHERE source_row_number = 1',
                             [str(project / 'data/processed/scientific/transactions.parquet')]).fetchone()[0]
        assert str(amount) == '10000000.00'


@pytest.mark.parametrize("steps,train,val", [([1, 2], .6, .2), ([1, 2, 3], .8, .4), ([1, 2, 3], 0, .2)])
def test_invalid_split_is_rejected(steps, train, val):
    with pytest.raises(ValueError):
        split_boundaries(steps, train, val)


def test_connected_components_include_singletons_and_ignore_direction():
    result = component_summary(7, [[(0, 1), (2, 1), (0, 1)], [(3, 4), (4, 3), (5, 5)]])
    assert result["count"] == 4
    assert result["largest_component_accounts"] == 3
    assert result["component_size_histogram"] == {1: 2, 2: 1, 3: 1}


def test_paths_cannot_escape_project(project):
    with pytest.raises(ValueError, match="inside project"):
        inside_root("../outside.txt", project)
    assert inside_root("data/raw/x.csv", project) == project / "data/raw/x.csv"


def test_end_to_end_preserves_rows_labels_and_temporal_boundaries(project, write_csv, rows):
    csv_path = write_csv(rows)
    report = run_pipeline(csv_path, "config/phase1.json", "test", root=project)
    assert report["quality"]["rows"] == 7
    assert report["quality"]["exact_duplicate_extra_rows"] == 1
    assert report["quality"]["zero_amount_rows"] == 1
    assert [entry["rows"] for entry in report["splits"]] == [4, 1, 2]
    assert report["training_label_summary"]["positives"] == 2
    assert report["graph"]["accounts"]["accounts"] == 4
    assert report["graph"]["weak_components"]["largest_component_accounts"] == 4
    assert report["graph"]["validation_cold_start"][0]["unseen_sender"] == 1
    data = project / "data/processed/test"
    with duckdb.connect() as con:
        tx = con.read_parquet(str(data / "transactions.parquet"))
        assert set(tx.columns).isdisjoint({"isFraud", "is_fraud", "isFlaggedFraud", "oldbalanceOrg", "newbalanceOrig"})
        assert tx.aggregate("count(DISTINCT transaction_id)").fetchone()[0] == 7
        labels = con.read_parquet(str(data / "labels.parquet"))
        assert labels.count("*").fetchone()[0] == 7
        assert labels.filter("label_available_step IS NOT NULL").count("*").fetchone()[0] == 0
        assert len(history_before(con, data / "transactions.parquet", 2)) == 1
    manifest = json.loads((project / "artifacts/phase1/test/manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert not (project / ".cache/runs/test/ingest.duckdb").exists()
    with pytest.raises(ValueError, match="already exists"):
        run_pipeline(csv_path, "config/phase1.json", "test", root=project)


@pytest.mark.parametrize("column,value", [(0, "1.5"), (2, "NaN"), (2, "-1"), (2, "1.001"),
                                         (1, "UNKNOWN"), (3, ""), (9, "2")])
def test_bad_data_fails_without_publishing_parquet(project, write_csv, rows, column, value):
    rows[0][column] = value
    path = write_csv(rows)
    with pytest.raises(ValueError, match="validation failed"):
        run_pipeline(path, "config/phase1.json", "bad", root=project)
    assert not list((project / "data/processed/bad").glob("*.parquet"))
    manifest = json.loads((project / "artifacts/phase1/bad/manifest.json").read_text())
    assert manifest["status"] == "failed"


def test_future_events_cannot_change_past_history(project, write_csv, rows):
    path = write_csv(rows)
    run_pipeline(path, "config/phase1.json", "original", root=project)
    parquet = project / "data/processed/original/transactions.parquet"
    with duckdb.connect() as con:
        before = history_before(con, parquet, 3)
        con.execute("CREATE TABLE events AS SELECT * FROM read_parquet(?)", [str(parquet)])
        con.execute("INSERT INTO events VALUES ('future', 999, 99, 'TRANSFER', 999, 'A', 'NEW', 'test')")
        augmented = project / "data/processed/augmented.parquet"
        con.execute("COPY events TO ? (FORMAT PARQUET)", [str(augmented)])
        assert history_before(con, augmented, 3) == before


def test_repeated_runs_have_identical_ids_and_splits(project, write_csv, rows):
    path = write_csv(rows)
    for name in ("one", "two"):
        run_pipeline(path, "config/phase1.json", name, root=project)
    with duckdb.connect() as con:
        one = con.read_parquet(str(project / "data/processed/one/transactions.parquet")).order("source_row_number").fetchall()
        two = con.read_parquet(str(project / "data/processed/two/transactions.parquet")).order("source_row_number").fetchall()
        assert one == two
