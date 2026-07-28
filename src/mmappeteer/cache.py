from __future__ import annotations

import fcntl
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from mmappet import DatasetWriter, open_dataset_dct


SCHEMA_VERSION = 2
KEY_COLUMNS = ("charge", "collision_energy", "sequence")
INTENSITY_COLUMN = "predicted_intensity"
ANNOTATION_COLUMN = "annotation_id"
MAX_ANNOTATIONS = np.iinfo(np.uint16).max + 1


class CacheError(RuntimeError):
    """Base class for cache-specific failures."""


class CacheKeyExistsError(CacheError):
    """Raised when an append would replace an existing cache entry."""


class CacheValidationError(CacheError):
    """Raised when the SQLite index and mmappet storage disagree."""


@dataclass(frozen=True)
class LookupResult:
    """A zero-copy view of cache storage plus ordered ranges for matching keys.

    ``hits`` is in the same relative order as the submitted keys. Its ``start``
    and ``end`` columns select matching slices from both storage arrays.
    ``missing`` contains submitted keys that were not present, preserving their
    order and original pandas index.
    """

    predicted_intensities: np.ndarray
    annotation_ids: np.ndarray
    hits: pd.DataFrame
    missing: pd.DataFrame

    @property
    def starts(self) -> np.ndarray:
        return self.hits["start"].to_numpy(dtype=np.int64, copy=False)

    @property
    def ends(self) -> np.ndarray:
        return self.hits["end"].to_numpy(dtype=np.int64, copy=False)

    def iter_arrays(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield an intensity and annotation-ID view for every matching key."""

        for start, end in zip(self.starts, self.ends):
            yield (
                self.predicted_intensities[start:end],
                self.annotation_ids[start:end],
            )


class PredictionCache:
    """Append-only cache keyed by charge, collision energy, and sequence."""

    database_filename = "index.sqlite3"
    dataset_filename = "arrays.mmappet"
    lock_filename = "write.lock"

    def __init__(self, path: str | Path, *, validate: bool = True):
        self.path = Path(path)
        self.database_path = self.path / self.database_filename
        self.dataset_path = self.path / self.dataset_filename
        self.lock_path = self.path / self.lock_filename

        if not self.database_path.is_file():
            raise FileNotFoundError(f"Cache database not found: {self.database_path}")
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Cache dataset not found: {self.dataset_path}")
        if validate:
            self.validate()

    @classmethod
    def create(
        cls,
        path: str | Path,
        annotations: Sequence[str],
        *,
        model_names: str | Sequence[str],
    ) -> "PredictionCache":
        """Create a cache for predictions produced by one or more models."""

        path = Path(path)
        annotation_names = _validate_annotation_names(annotations)
        normalized_model_names = _validate_model_names(model_names)
        database_path = path / cls.database_filename
        dataset_path = path / cls.dataset_filename

        path.mkdir(parents=True, exist_ok=True)
        if database_path.exists() or dataset_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing cache at {path}"
            )

        connection = sqlite3.connect(database_path)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = FULL;

                CREATE TABLE metadata (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) STRICT, WITHOUT ROWID;

                CREATE TABLE annotations (
                    annotation_id INTEGER PRIMARY KEY
                                  CHECK (annotation_id >= 0),
                    annotation    TEXT NOT NULL UNIQUE
                ) STRICT;

                CREATE TABLE cache_entries (
                    charge           INTEGER NOT NULL CHECK (charge >= 1),
                    collision_energy REAL NOT NULL,
                    sequence         TEXT NOT NULL,
                    start            INTEGER NOT NULL CHECK (start >= 0),
                    end              INTEGER NOT NULL CHECK (end >= start),
                    PRIMARY KEY (charge, collision_energy, sequence)
                ) STRICT, WITHOUT ROWID;
                """
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("annotation_count", str(len(annotation_names))),
                    ("collision_energy_dtype", "float32"),
                    ("intensity_dtype", "float32"),
                    ("annotation_id_dtype", "uint16"),
                    (
                        "model_names",
                        json.dumps(
                            normalized_model_names,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    ),
                ),
            )
            connection.executemany(
                "INSERT INTO annotations(annotation_id, annotation) VALUES (?, ?)",
                enumerate(annotation_names),
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        except Exception:
            connection.close()
            database_path.unlink(missing_ok=True)
            raise
        finally:
            if connection:
                connection.close()

        try:
            with DatasetWriter.new(
                dataset_path,
                predicted_intensity=np.float32,
                annotation_id=np.uint16,
            ):
                pass
        except Exception:
            database_path.unlink(missing_ok=True)
            raise

        (path / cls.lock_filename).touch(exist_ok=True)

        return cls(path)

    def annotations(self) -> pd.DataFrame:
        """Return the complete annotation vocabulary in ID order."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT annotation_id, annotation
                FROM annotations
                ORDER BY annotation_id
                """
            ).fetchall()
        return pd.DataFrame(rows, columns=["annotation_id", "annotation"])

    def model_names(self) -> tuple[str, ...]:
        """Return the model provenance recorded when the cache was created."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'model_names'"
            ).fetchone()
        if row is None:
            raise CacheValidationError("metadata is missing model_names")
        return tuple(_decode_model_names(row[0]))

    def append(
        self,
        *,
        charge: int,
        collision_energy: float,
        sequence: str,
        predicted_intensities: Sequence[float] | np.ndarray,
        annotation_ids: Sequence[int] | np.ndarray,
    ) -> tuple[int, int]:
        """Append one prediction to the cache.

        Parameters
        ----------
        charge
            Precursor charge; must be an integer greater than or equal to one.
        collision_energy
            Collision energy, normalized to ``float32`` for the cache key.
        sequence
            Peptide sequence used in the cache key.
        predicted_intensities
            One-dimensional intensity vector. Values are stored as ``float32``.
        annotation_ids
            One-dimensional annotation-ID vector aligned with
            ``predicted_intensities``. Values are stored as ``uint16``.

        Returns
        -------
        start : int
            Inclusive row offset of the appended vectors in mmappet storage.
        end : int
            Exclusive row offset of the appended vectors in mmappet storage.
            The stored vectors are selected with ``storage[start:end]``.
        """

        keys = pd.DataFrame(
            {
                "charge": [charge],
                "collision_energy": [collision_energy],
                "sequence": [sequence],
            }
        )
        ranges = self.append_many(
            keys,
            predicted_intensities=[predicted_intensities],
            annotation_ids=[annotation_ids],
        )
        return int(ranges.iloc[0]["start"]), int(ranges.iloc[0]["end"])

    def append_many(
        self,
        keys: pd.DataFrame,
        *,
        predicted_intensities: Sequence[Sequence[float] | np.ndarray],
        annotation_ids: Sequence[Sequence[int] | np.ndarray],
    ) -> pd.DataFrame:
        """Append several predictions in submitted-key order.

        Parameters
        ----------
        keys
            DataFrame containing ``charge``, ``collision_energy``, and
            ``sequence`` columns. Its row order determines append order.
        predicted_intensities
            One one-dimensional intensity vector per row in ``keys``.
        annotation_ids
            One one-dimensional annotation-ID vector per row in ``keys``.
            Each vector must align with its corresponding intensity vector.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with ``start`` and ``end`` columns and the same index
            and row order as ``keys``. Each row describes a half-open mmappet
            range ``[start, end)`` shared by the corresponding intensity and
            annotation-ID vectors.
        """

        normalized_keys = _normalize_keys(keys)
        intensity_vectors = list(predicted_intensities)
        annotation_vectors = list(annotation_ids)
        if not (
            len(normalized_keys)
            == len(intensity_vectors)
            == len(annotation_vectors)
        ):
            raise ValueError(
                "keys, predicted_intensities, and annotation_ids must have "
                "the same number of entries"
            )
        if normalized_keys.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError("keys contains duplicate cache keys")

        annotation_count = len(self.annotations())
        arrays = [
            _normalize_arrays(intensities, ids, annotation_count)
            for intensities, ids in zip(intensity_vectors, annotation_vectors)
        ]

        ranges: list[tuple[int, int]] = []
        self.lock_path.touch(exist_ok=True)
        with self.lock_path.open("rb") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = _lookup_rows(connection, normalized_keys)
                    if any(row[4] is not None for row in existing):
                        duplicate_positions = [
                            row[0] for row in existing if row[4] is not None
                        ]
                        raise CacheKeyExistsError(
                            "Cache keys already exist at submitted positions "
                            f"{duplicate_positions}"
                        )

                    with DatasetWriter(self.dataset_path, append_ok=True) as writer:
                        for intensities, ids in arrays:
                            start = len(writer)
                            writer.append(
                                predicted_intensity=intensities,
                                annotation_id=ids,
                            )
                            ranges.append((start, len(writer)))
                        writer.flush()

                    connection.executemany(
                        """
                        INSERT INTO cache_entries(
                            charge, collision_energy, sequence, start, end
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            (
                                int(key.charge),
                                float(key.collision_energy),
                                key.sequence,
                                start,
                                end,
                            )
                            for key, (start, end) in zip(
                                normalized_keys.itertuples(index=False), ranges
                            )
                        ),
                    )
                    connection.commit()
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        return pd.DataFrame(ranges, columns=["start", "end"], index=keys.index)

    def lookup(self, keys: pd.DataFrame) -> LookupResult:
        """Look up keys without copying the cached vectors.

        Hits and missing keys both preserve submission order. Duplicate
        submitted keys are supported.
        """

        normalized_keys = _normalize_keys(keys)
        with self._connect() as connection:
            rows = _lookup_rows(connection, normalized_keys)

        hit_positions = [row[0] for row in rows if row[4] is not None]
        missing_positions = [row[0] for row in rows if row[4] is None]

        hits = keys.iloc[hit_positions][list(KEY_COLUMNS)].copy()
        hits["start"] = [rows[pos][4] for pos in hit_positions]
        hits["end"] = [rows[pos][5] for pos in hit_positions]
        hits["start"] = hits["start"].astype(np.int64)
        hits["end"] = hits["end"].astype(np.int64)
        missing = keys.iloc[missing_positions][list(KEY_COLUMNS)].copy()

        storage = open_dataset_dct(self.dataset_path)
        return LookupResult(
            predicted_intensities=storage[INTENSITY_COLUMN],
            annotation_ids=storage[ANNOTATION_COLUMN],
            hits=hits,
            missing=missing,
        )

    def validate(self) -> None:
        """Check schema versions, annotation numbering, and stored ranges."""

        storage = open_dataset_dct(self.dataset_path)
        if set(storage) != {INTENSITY_COLUMN, ANNOTATION_COLUMN}:
            raise CacheValidationError(
                f"Unexpected mmappet columns: {list(storage)}"
            )
        if storage[INTENSITY_COLUMN].dtype != np.dtype(np.float32):
            raise CacheValidationError("predicted_intensity must have dtype float32")
        if storage[ANNOTATION_COLUMN].dtype != np.dtype(np.uint16):
            raise CacheValidationError("annotation_id must have dtype uint16")
        if len(storage[INTENSITY_COLUMN]) != len(storage[ANNOTATION_COLUMN]):
            raise CacheValidationError("mmappet columns have unequal lengths")

        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise CacheValidationError(
                    f"Unsupported schema version {version}; expected {SCHEMA_VERSION}"
                )

            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            expected_metadata = {
                "schema_version": str(SCHEMA_VERSION),
                "collision_energy_dtype": "float32",
                "intensity_dtype": "float32",
                "annotation_id_dtype": "uint16",
            }
            for key, expected in expected_metadata.items():
                if metadata.get(key) != expected:
                    raise CacheValidationError(
                        f"Invalid metadata {key!r}: {metadata.get(key)!r}"
                    )
            if "model_names" not in metadata:
                raise CacheValidationError("metadata is missing model_names")
            _decode_model_names(metadata["model_names"])

            annotation_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT annotation_id FROM annotations ORDER BY annotation_id"
                )
            ]
            if annotation_ids != list(range(len(annotation_ids))):
                raise CacheValidationError(
                    "annotations.annotation_id values must be contiguous from zero"
                )
            if metadata.get("annotation_count") != str(len(annotation_ids)):
                raise CacheValidationError(
                    "metadata annotation_count does not match annotations table"
                )

            bad_range = connection.execute(
                """
                SELECT charge, collision_energy, sequence, start, end
                FROM cache_entries
                WHERE end > ?
                LIMIT 1
                """,
                (len(storage[INTENSITY_COLUMN]),),
            ).fetchone()
            if bad_range is not None:
                raise CacheValidationError(
                    f"Cache entry points outside mmappet storage: {bad_range}"
                )

            overlap = connection.execute(
                """
                WITH ordered AS (
                    SELECT start, end,
                           MAX(end) OVER (
                               ORDER BY start, end
                               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                           ) AS previous_max_end
                    FROM cache_entries
                )
                SELECT start, end, previous_max_end
                FROM ordered
                WHERE start < previous_max_end
                LIMIT 1
                """
            ).fetchone()
            if overlap is not None:
                raise CacheValidationError(
                    f"Overlapping cache ranges found: {overlap}"
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _validate_annotation_names(annotations: Sequence[str]) -> list[str]:
    names = list(annotations)
    if not names:
        raise ValueError("annotations must contain at least one entry")
    if len(names) > MAX_ANNOTATIONS:
        raise ValueError(
            f"At most {MAX_ANNOTATIONS} annotations fit in the uint16 storage"
        )
    if any(not isinstance(name, str) for name in names):
        raise TypeError("every annotation must be a string")
    if len(set(names)) != len(names):
        raise ValueError("annotations must be unique")
    return names


def _validate_model_names(model_names: str | Sequence[str]) -> list[str]:
    names = [model_names] if isinstance(model_names, str) else list(model_names)
    if not names:
        raise ValueError("model_names must contain at least one entry")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("every model name must be a string")
    if any(not name.strip() for name in names):
        raise ValueError("model names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique")
    return names


def _decode_model_names(value: str) -> list[str]:
    try:
        names = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise CacheValidationError(
            "metadata model_names must be a JSON string list"
        ) from error
    if not isinstance(names, list):
        raise CacheValidationError("metadata model_names must be a JSON string list")
    try:
        return _validate_model_names(names)
    except (TypeError, ValueError) as error:
        raise CacheValidationError(f"Invalid metadata model_names: {error}") from error


def _normalize_keys(keys: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(keys, pd.DataFrame):
        raise TypeError("keys must be a pandas DataFrame")
    missing_columns = [column for column in KEY_COLUMNS if column not in keys]
    if missing_columns:
        raise ValueError(f"keys is missing columns: {missing_columns}")

    normalized = keys.loc[:, list(KEY_COLUMNS)].copy()
    if normalized.empty:
        normalized["charge"] = np.empty(0, dtype=np.int64)
        normalized["collision_energy"] = np.empty(0, dtype=np.float32)
        return normalized

    charge_kind = pd.api.types.infer_dtype(normalized["charge"], skipna=False)
    if charge_kind not in {"integer", "floating", "mixed-integer-float"}:
        raise TypeError("charge must contain integers")
    try:
        charge_values = normalized["charge"].to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("charge must contain integers") from error
    valid_charges = (
        np.isfinite(charge_values)
        & (charge_values >= 1)
        & (charge_values == np.floor(charge_values))
        & (charge_values <= np.iinfo(np.int64).max)
    )
    if not np.all(valid_charges):
        invalid = np.flatnonzero(~valid_charges).tolist()
        raise ValueError(
            "charge must contain integers >= 1; invalid submitted positions "
            f"{invalid}"
        )

    energy_kind = pd.api.types.infer_dtype(
        normalized["collision_energy"], skipna=False
    )
    if energy_kind not in {
        "integer",
        "floating",
        "mixed-integer-float",
        "decimal",
    }:
        raise TypeError("collision_energy must contain numeric values")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            energy_values = normalized["collision_energy"].to_numpy(
                dtype=np.float32
            )
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("collision_energy must contain numeric values") from error
    valid_energies = np.isfinite(energy_values)
    if not np.all(valid_energies):
        invalid = np.flatnonzero(~valid_energies).tolist()
        raise ValueError(
            "collision_energy must contain finite values; invalid submitted "
            f"positions {invalid}"
        )

    sequence_kind = pd.api.types.infer_dtype(
        normalized["sequence"], skipna=False
    )
    if sequence_kind not in {"string", "unicode"}:
        raise TypeError("sequence must contain strings")

    normalized["charge"] = charge_values.astype(np.int64)
    normalized["collision_energy"] = energy_values
    return normalized


def _normalize_arrays(
    predicted_intensities: Sequence[float] | np.ndarray,
    annotation_ids: Sequence[int] | np.ndarray,
    annotation_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    intensities = np.asarray(predicted_intensities, dtype=np.float32)
    raw_ids = np.asarray(annotation_ids)
    if intensities.ndim != 1 or raw_ids.ndim != 1:
        raise ValueError("predicted_intensities and annotation_ids must be 1-D")
    if len(intensities) != len(raw_ids):
        raise ValueError(
            "predicted_intensities and annotation_ids must have equal lengths"
        )
    if not np.issubdtype(raw_ids.dtype, np.integer):
        raise TypeError("annotation_ids must contain integers")
    if len(raw_ids) and (
        np.any(raw_ids < 0) or np.any(raw_ids >= annotation_count)
    ):
        raise ValueError(
            f"annotation_ids must be between 0 and {annotation_count - 1}"
        )
    return intensities, raw_ids.astype(np.uint16, copy=False)


def _lookup_rows(
    connection: sqlite3.Connection,
    normalized_keys: pd.DataFrame,
) -> list[tuple[int, int, float, str, int | None, int | None]]:
    connection.execute("DROP TABLE IF EXISTS temp.requested_keys")
    connection.execute(
        """
        CREATE TEMP TABLE requested_keys (
            request_position INTEGER PRIMARY KEY,
            charge           INTEGER NOT NULL,
            collision_energy REAL NOT NULL,
            sequence         TEXT NOT NULL
        ) STRICT
        """
    )
    connection.executemany(
        """
        INSERT INTO requested_keys(
            request_position, charge, collision_energy, sequence
        ) VALUES (?, ?, ?, ?)
        """,
        (
            (
                position,
                int(key.charge),
                float(key.collision_energy),
                key.sequence,
            )
            for position, key in enumerate(
                normalized_keys.itertuples(index=False)
            )
        ),
    )
    return connection.execute(
        """
        SELECT requested_keys.request_position,
               requested_keys.charge,
               requested_keys.collision_energy,
               requested_keys.sequence,
               cache_entries.start,
               cache_entries.end
        FROM requested_keys
        LEFT JOIN cache_entries USING (charge, collision_energy, sequence)
        ORDER BY requested_keys.request_position
        """
    ).fetchall()
