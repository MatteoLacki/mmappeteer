import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mmappeteer import (
    CacheKeyExistsError,
    CacheValidationError,
    PackedPredictions,
    PredictionCache,
    PredictionKeys,
)


def make_keys(charge, collision_energy, sequence):
    return PredictionKeys.validate(
        charge=np.asarray(charge),
        collision_energy=np.asarray(collision_energy),
        sequence=np.asarray(sequence),
    )


@pytest.fixture
def cache(tmp_path):
    return PredictionCache.create(
        tmp_path / "cache",
        np.asarray(["b1", "b2", "y1", "y2"]),
        model_names=np.asarray(["Prosit_2020_intensity_HCD", "local-finetune-v2"]),
    )


def test_create_populates_numpy_metadata(cache):
    annotations = cache.annotations()

    np.testing.assert_array_equal(annotations.ids, [0, 1, 2, 3])
    np.testing.assert_array_equal(annotations.names, ["b1", "b2", "y1", "y2"])
    np.testing.assert_array_equal(
        cache.model_names(),
        ["Prosit_2020_intensity_HCD", "local-finetune-v2"],
    )

    with sqlite3.connect(cache.database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT value FROM metadata WHERE key = 'annotation_count'"
            ).fetchone()[0]
            == "4"
        )
        assert connection.execute(
            "SELECT value FROM metadata WHERE key = 'model_names'"
        ).fetchone()[0] == ('["Prosit_2020_intensity_HCD","local-finetune-v2"]')


@pytest.mark.parametrize(
    "model_names",
    [
        np.asarray([]),
        np.asarray([""]),
        np.asarray(["   "]),
        np.asarray(["same", "same"]),
        np.asarray(["valid", 1], dtype=object),
    ],
)
def test_create_rejects_invalid_model_names(tmp_path, model_names):
    with pytest.raises((TypeError, ValueError), match="model"):
        PredictionCache.create(
            tmp_path / "cache",
            np.asarray(["b1"]),
            model_names=model_names,
        )


def test_append_and_lookup_are_aligned_with_submitted_keys(cache):
    cache.append(
        charge=2,
        collision_energy=30.0,
        sequence="PEPTIDE",
        predicted_intensities=np.asarray([0.2, 0.8], dtype=np.float32),
        annotation_ids=np.asarray([0, 2], dtype=np.uint16),
    )
    cache.append(
        charge=3,
        collision_energy=25.0,
        sequence="OTHER",
        predicted_intensities=np.asarray([0.1, 0.3, 0.6], dtype=np.float32),
        annotation_ids=np.asarray([1, 2, 3], dtype=np.uint16),
    )

    requested = make_keys(
        charge=[3, 1, 2],
        collision_energy=[25.0, 20.0, 30.0],
        sequence=["OTHER", "MISSING", "PEPTIDE"],
    )
    result = cache.lookup(requested)

    np.testing.assert_array_equal(result.starts, [2, -1, 0])
    np.testing.assert_array_equal(result.ends, [5, -1, 2])
    np.testing.assert_array_equal(result.found, [True, False, True])
    np.testing.assert_array_equal(result.missing_positions, [1])
    missing_keys = requested.take(result.missing_positions)
    np.testing.assert_array_equal(missing_keys.sequence, ["MISSING"])

    slices = list(result.iter_arrays())
    np.testing.assert_array_equal(
        slices[0][0], np.asarray([0.1, 0.3, 0.6], dtype=np.float32)
    )
    np.testing.assert_array_equal(slices[0][1], [1, 2, 3])
    np.testing.assert_array_equal(
        slices[1][0], np.asarray([0.2, 0.8], dtype=np.float32)
    )
    np.testing.assert_array_equal(slices[1][1], [0, 2])


def test_lookup_supports_duplicate_submitted_keys(cache):
    cache.append(
        charge=2,
        collision_energy=30,
        sequence="PEPTIDE",
        predicted_intensities=np.asarray([1.0]),
        annotation_ids=np.asarray([0]),
    )

    result = cache.lookup(make_keys([2, 2], [30, 30], ["PEPTIDE", "PEPTIDE"]))

    np.testing.assert_array_equal(result.starts, [0, 0])
    np.testing.assert_array_equal(result.ends, [1, 1])
    assert np.all(result.found)


def test_collision_energy_is_normalized_to_float32(cache):
    energy = 10.123456789
    keys = make_keys([2], [energy], ["PEPTIDE"])
    assert keys.collision_energy.dtype == np.dtype(np.float32)

    cache.append(
        charge=2,
        collision_energy=energy,
        sequence="PEPTIDE",
        predicted_intensities=np.asarray([1.0]),
        annotation_ids=np.asarray([0]),
    )

    assert cache.lookup(keys).found[0]


def test_append_many_writes_flattened_arrays_and_returns_ranges(cache):
    keys = make_keys([2, 3], [20.0, 25.0], ["ONE", "TWO"])
    predictions = PackedPredictions.validate(
        predicted_intensities=np.asarray([0.2, 0.3, 0.7]),
        annotation_ids=np.asarray([0, 1, 2]),
        offsets=np.asarray([0, 1, 3]),
    )

    ranges = cache.append_many(keys, predictions)

    np.testing.assert_array_equal(ranges.starts, [0, 1])
    np.testing.assert_array_equal(ranges.ends, [1, 3])
    lookup = cache.lookup(keys)
    np.testing.assert_array_equal(
        lookup.predicted_intensities,
        np.asarray([0.2, 0.3, 0.7], dtype=np.float32),
    )
    np.testing.assert_array_equal(lookup.annotation_ids, [0, 1, 2])


def test_existing_key_is_rejected_without_appending(cache):
    keys = make_keys([2], [20], ["ONE"])
    predictions = PackedPredictions.validate(
        predicted_intensities=np.asarray([0.2]),
        annotation_ids=np.asarray([0]),
        offsets=np.asarray([0, 1]),
    )
    cache.append_many(keys, predictions)

    with pytest.raises(CacheKeyExistsError):
        cache.append_many(keys, predictions)

    assert len(cache.lookup(make_keys([], [], [])).predicted_intensities) == 1


def test_duplicate_batch_keys_are_rejected_before_append(cache):
    keys = make_keys([2, 2], [20, 20], ["ONE", "ONE"])
    predictions = PackedPredictions.validate(
        predicted_intensities=np.asarray([0.2, 0.3]),
        annotation_ids=np.asarray([0, 1]),
        offsets=np.asarray([0, 1, 2]),
    )

    with pytest.raises(CacheKeyExistsError, match="duplicate"):
        cache.append_many(keys, predictions)

    assert len(cache.lookup(make_keys([], [], [])).predicted_intensities) == 0


@pytest.mark.parametrize(
    "charge",
    [np.asarray([0]), np.asarray([-1]), np.asarray([1.5]), np.asarray([True])],
)
def test_invalid_charge_is_rejected(charge):
    with pytest.raises((TypeError, ValueError)):
        PredictionKeys.validate(
            charge=charge,
            collision_energy=np.asarray([20]),
            sequence=np.asarray(["PEPTIDE"]),
        )


@pytest.mark.parametrize(
    "offsets",
    [
        np.asarray([]),
        np.asarray([1, 2]),
        np.asarray([0, 2, 1]),
        np.asarray([0, 1]),
        np.asarray([0.0, 2.0]),
    ],
)
def test_invalid_packed_offsets_are_rejected(offsets):
    with pytest.raises((TypeError, ValueError), match="offset"):
        PackedPredictions.validate(
            predicted_intensities=np.asarray([0.2, 0.8]),
            annotation_ids=np.asarray([0, 1]),
            offsets=offsets,
        )


def test_invalid_annotation_id_is_rejected_before_writing(cache):
    with pytest.raises(ValueError, match="annotation_ids"):
        cache.append(
            charge=2,
            collision_energy=20,
            sequence="PEPTIDE",
            predicted_intensities=np.asarray([1]),
            annotation_ids=np.asarray([4]),
        )

    assert len(cache.lookup(make_keys([], [], [])).predicted_intensities) == 0


def test_empty_batch_is_supported(cache):
    result = cache.append_many(
        make_keys([], [], []),
        PackedPredictions.validate(
            predicted_intensities=np.asarray([]),
            annotation_ids=np.asarray([], dtype=np.uint16),
            offsets=np.asarray([0]),
        ),
    )

    assert result.starts.dtype == np.dtype(np.int64)
    assert len(result.starts) == 0
    lookup = cache.lookup(make_keys([], [], []))
    assert len(lookup.found) == 0


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


def test_importing_mmappeteer_does_not_import_pandas():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.fspath((Path(__file__).parents[1] / "src").resolve())
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, mmappeteer; assert 'pandas' not in sys.modules",
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
