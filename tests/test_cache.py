import sqlite3

import numpy as np
import pandas as pd
import pytest

from mmappeteer import (
    CacheKeyExistsError,
    CacheValidationError,
    PredictionCache,
)


@pytest.fixture
def cache(tmp_path):
    return PredictionCache.create(
        tmp_path / "cache",
        ["b1", "b2", "y1", "y2"],
        model_names=["Prosit_2020_intensity_HCD", "local-finetune-v2"],
    )


def test_create_populates_and_numbers_annotations(cache):
    annotations = cache.annotations()

    assert annotations.to_dict("records") == [
        {"annotation_id": 0, "annotation": "b1"},
        {"annotation_id": 1, "annotation": "b2"},
        {"annotation_id": 2, "annotation": "y1"},
        {"annotation_id": 3, "annotation": "y2"},
    ]

    with sqlite3.connect(cache.database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute(
            "SELECT value FROM metadata WHERE key = 'annotation_count'"
        ).fetchone()[0] == "4"
        assert connection.execute(
            "SELECT value FROM metadata WHERE key = 'model_names'"
        ).fetchone()[0] == (
            '["Prosit_2020_intensity_HCD","local-finetune-v2"]'
        )
    assert cache.model_names() == (
        "Prosit_2020_intensity_HCD",
        "local-finetune-v2",
    )


@pytest.mark.parametrize(
    "model_names",
    [[], [""], ["   "], ["same", "same"], ["valid", 1]],
)
def test_create_rejects_invalid_model_names(tmp_path, model_names):
    with pytest.raises((TypeError, ValueError), match="model"):
        PredictionCache.create(
            tmp_path / "cache",
            ["b1"],
            model_names=model_names,
        )


def test_append_and_lookup_preserve_requested_order_and_report_missing(cache):
    cache.append(
        charge=2,
        collision_energy=30.0,
        sequence="PEPTIDE",
        predicted_intensities=np.array([0.2, 0.8], dtype=np.float32),
        annotation_ids=np.array([0, 2], dtype=np.uint16),
    )
    cache.append(
        charge=3,
        collision_energy=25.0,
        sequence="OTHER",
        predicted_intensities=np.array([0.1, 0.3, 0.6], dtype=np.float32),
        annotation_ids=np.array([1, 2, 3], dtype=np.uint16),
    )

    requested = pd.DataFrame(
        {
            "charge": [3, 1, 2],
            "collision_energy": [25.0, 20.0, 30.0],
            "sequence": ["OTHER", "MISSING", "PEPTIDE"],
        },
        index=["second", "absent", "first"],
    )
    result = cache.lookup(requested)

    assert list(result.hits.index) == ["second", "first"]
    assert result.starts.tolist() == [2, 0]
    assert result.ends.tolist() == [5, 2]
    assert list(result.missing.index) == ["absent"]
    assert result.missing.iloc[0].to_dict() == {
        "charge": 1,
        "collision_energy": 20.0,
        "sequence": "MISSING",
    }

    slices = list(result.iter_arrays())
    np.testing.assert_array_equal(
        slices[0][0], np.array([0.1, 0.3, 0.6], dtype=np.float32)
    )
    np.testing.assert_array_equal(slices[0][1], [1, 2, 3])
    np.testing.assert_array_equal(
        slices[1][0], np.array([0.2, 0.8], dtype=np.float32)
    )
    np.testing.assert_array_equal(slices[1][1], [0, 2])


def test_lookup_supports_duplicate_submitted_keys(cache):
    cache.append(
        charge=2,
        collision_energy=30,
        sequence="PEPTIDE",
        predicted_intensities=[1.0],
        annotation_ids=[0],
    )
    keys = pd.DataFrame(
        {
            "charge": [2, 2],
            "collision_energy": [30, 30],
            "sequence": ["PEPTIDE", "PEPTIDE"],
        }
    )

    result = cache.lookup(keys)

    assert result.starts.tolist() == [0, 0]
    assert result.ends.tolist() == [1, 1]
    assert result.missing.empty


def test_collision_energy_is_consistently_normalized_to_float32(cache):
    energy = 10.123456789
    cache.append(
        charge=2,
        collision_energy=energy,
        sequence="PEPTIDE",
        predicted_intensities=[1.0],
        annotation_ids=[0],
    )

    result = cache.lookup(
        pd.DataFrame(
            {
                "charge": [2],
                "collision_energy": [np.float32(energy)],
                "sequence": ["PEPTIDE"],
            }
        )
    )

    assert len(result.hits) == 1


def test_key_normalization_handles_vectorized_columns(cache):
    keys = pd.DataFrame(
        {
            "charge": pd.Series([2, 3], dtype="Int64"),
            "collision_energy": np.array([20.0, 25.0], dtype=np.float64),
            "sequence": pd.Series(["ONE", "TWO"], dtype="string"),
        }
    )
    cache.append_many(
        keys,
        predicted_intensities=[[0.2], [0.3]],
        annotation_ids=[[0], [1]],
    )

    result = cache.lookup(keys.iloc[::-1])

    assert result.starts.tolist() == [1, 0]
    assert result.missing.empty


def test_append_many_and_reject_existing_key_without_appending(cache):
    keys = pd.DataFrame(
        {
            "charge": [2, 3],
            "collision_energy": [20.0, 25.0],
            "sequence": ["ONE", "TWO"],
        }
    )
    ranges = cache.append_many(
        keys,
        predicted_intensities=[[0.2], [0.3, 0.7]],
        annotation_ids=[[0], [1, 2]],
    )
    assert ranges.to_dict("records") == [
        {"start": 0, "end": 1},
        {"start": 1, "end": 3},
    ]

    with pytest.raises(CacheKeyExistsError):
        cache.append(
            charge=2,
            collision_energy=20.0,
            sequence="ONE",
            predicted_intensities=[9.0],
            annotation_ids=[0],
        )

    storage = cache.lookup(keys.iloc[:0])
    assert len(storage.predicted_intensities) == 3


@pytest.mark.parametrize("charge", [0, -1, 1.5, True])
def test_invalid_charge_is_rejected(cache, charge):
    with pytest.raises((TypeError, ValueError)):
        cache.append(
            charge=charge,
            collision_energy=20,
            sequence="PEPTIDE",
            predicted_intensities=[1],
            annotation_ids=[0],
        )


def test_invalid_annotation_id_is_rejected_before_writing(cache):
    with pytest.raises(ValueError, match="annotation_ids"):
        cache.append(
            charge=2,
            collision_energy=20,
            sequence="PEPTIDE",
            predicted_intensities=[1],
            annotation_ids=[4],
        )

    assert len(cache.lookup(pd.DataFrame(columns=[
        "charge", "collision_energy", "sequence"
    ])).predicted_intensities) == 0


def test_validation_detects_bad_annotation_count(cache):
    with sqlite3.connect(cache.database_path) as connection:
        connection.execute(
            "UPDATE metadata SET value = '174' WHERE key = 'annotation_count'"
        )

    with pytest.raises(CacheValidationError, match="annotation_count"):
        PredictionCache(cache.path)


def test_validation_detects_invalid_model_names(cache):
    with sqlite3.connect(cache.database_path) as connection:
        connection.execute(
            "UPDATE metadata SET value = 'not-json' WHERE key = 'model_names'"
        )

    with pytest.raises(CacheValidationError, match="model_names"):
        PredictionCache(cache.path)
