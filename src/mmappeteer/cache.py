from __future__ import annotations

import fcntl
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

SCHEMA_VERSION = 2
INTENSITY_COLUMN = "predicted_intensity"
ANNOTATION_COLUMN = "annotation_id"
MAX_ANNOTATIONS = np.iinfo(np.uint16).max + 1


def _load_mmappet():
    """Import the storage backend only when cache storage is accessed."""

    import mmappet

    return mmappet


class CacheError(RuntimeError):
    """Base class for cache-specific failures."""


class CacheKeyExistsError(CacheError):
    """Raised when an append would replace an existing cache entry."""


class CacheValidationError(CacheError):
    """Raised when the SQLite index and mmappet storage disagree."""


@dataclass(frozen=True)
class PredictionKeys:
    """Parallel NumPy arrays forming ordered prediction cache keys."""

    charge: npt.NDArray[np.int64]
    collision_energy: npt.NDArray[np.float32]
    sequence: npt.NDArray[np.object_]

    @classmethod
    def validate(
        cls,
        *,
        charge: npt.ArrayLike,
        collision_energy: npt.ArrayLike,
        sequence: npt.ArrayLike,
    ) -> PredictionKeys:
        """Normalize arrays, validate key invariants, and construct keys."""

        normalized_charge = _normalize_integer_array(charge, "charge", minimum=1)
        normalized_energy = _normalize_collision_energy(collision_energy)
        normalized_sequence = _normalize_string_array(sequence, "sequence")
        lengths = (
            len(normalized_charge),
            len(normalized_energy),
            len(normalized_sequence),
        )
        if len(set(lengths)) != 1:
            raise ValueError(
                "charge, collision_energy, and sequence must have equal "
                f"lengths; got {lengths}"
            )
        return cls(
            charge=normalized_charge,
            collision_energy=normalized_energy,
            sequence=normalized_sequence,
        )

    def __len__(self) -> int:
        return len(self.charge)

    def take(self, indices: npt.ArrayLike) -> PredictionKeys:
        """Select keys by an integer index array or Boolean mask."""

        selector = np.asarray(indices)
        if selector.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if selector.dtype.kind == "b":
            if len(selector) != len(self):
                raise ValueError(
                    "a Boolean selector must have the same length as the keys"
                )
            positions = np.flatnonzero(selector)
        elif selector.dtype.kind in "iu":
            positions = selector.astype(np.intp, copy=False)
        else:
            raise TypeError("indices must contain integers or Booleans")
        return PredictionKeys.validate(
            charge=np.take(self.charge, positions),
            collision_energy=np.take(self.collision_energy, positions),
            sequence=np.take(self.sequence, positions),
        )


@dataclass(frozen=True)
class PackedPredictions:
    """Flattened variable-length prediction vectors and their offsets."""

    predicted_intensities: npt.NDArray[np.float32]
    annotation_ids: npt.NDArray[np.uint16]
    offsets: npt.NDArray[np.int64]

    @classmethod
    def validate(
        cls,
        *,
        predicted_intensities: npt.ArrayLike,
        annotation_ids: npt.ArrayLike,
        offsets: npt.ArrayLike,
    ) -> PackedPredictions:
        """Normalize arrays, validate packed invariants, and construct data."""

        intensities = np.asarray(predicted_intensities, dtype=np.float32)
        normalized_annotation_ids = _normalize_integer_array(
            annotation_ids,
            "annotation_ids",
            minimum=0,
            maximum=np.iinfo(np.uint16).max,
            dtype=np.uint16,
        )
        normalized_offsets = _normalize_integer_array(
            offsets, "offsets", minimum=0, dtype=np.int64
        )
        if intensities.ndim != 1:
            raise ValueError("predicted_intensities must be one-dimensional")
        if len(intensities) != len(normalized_annotation_ids):
            raise ValueError(
                "predicted_intensities and annotation_ids must have equal lengths"
            )
        if len(normalized_offsets) == 0:
            raise ValueError("offsets must contain at least the initial zero")
        if normalized_offsets[0] != 0:
            raise ValueError("offsets must start at zero")
        if np.any(normalized_offsets[1:] < normalized_offsets[:-1]):
            raise ValueError("offsets must be non-decreasing")
        if normalized_offsets[-1] != len(intensities):
            raise ValueError("the final offset must equal the flattened vector length")
        return cls(
            predicted_intensities=intensities,
            annotation_ids=normalized_annotation_ids,
            offsets=normalized_offsets,
        )

    def __len__(self) -> int:
        return len(self.offsets) - 1


@dataclass(frozen=True)
class AppendResult:
    """Half-open mmappet ranges aligned with appended keys."""

    starts: npt.NDArray[np.int64]
    ends: npt.NDArray[np.int64]


@dataclass(frozen=True)
class AnnotationVocabulary:
    """Contiguously numbered annotation names."""

    ids: npt.NDArray[np.uint16]
    names: npt.NDArray[np.object_]

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class LookupResult:
    """Mmap-backed storage and ranges aligned with every submitted key.

    Missing keys have ``start == end == -1`` and ``found == False``.
    """

    predicted_intensities: np.ndarray
    annotation_ids: np.ndarray
    starts: npt.NDArray[np.int64]
    ends: npt.NDArray[np.int64]
    found: npt.NDArray[np.bool_]

    @property
    def missing_positions(self) -> npt.NDArray[np.int64]:
        return np.flatnonzero(~self.found)

    def iter_arrays(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield array views for found keys in submitted-key order."""

        for position in np.flatnonzero(self.found):
            start = self.starts[position]
            end = self.ends[position]
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
        annotations: npt.ArrayLike,
        *,
        model_names: str | npt.ArrayLike,
    ) -> PredictionCache:
        """Create a cache for predictions produced by one or more models."""

        path = Path(path)
        annotation_names = _validate_names(annotations, "annotations")
        if len(annotation_names) > MAX_ANNOTATIONS:
            raise ValueError(
                f"At most {MAX_ANNOTATIONS} annotations fit in uint16 storage"
            )
        normalized_model_names = _validate_names(
            [model_names] if isinstance(model_names, str) else model_names,
            "model_names",
        )
        database_path = path / cls.database_filename
        dataset_path = path / cls.dataset_filename

        path.mkdir(parents=True, exist_ok=True)
        if database_path.exists() or dataset_path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing cache at {path}")

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
                    annotation_id INTEGER PRIMARY KEY CHECK (annotation_id >= 0),
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
                            normalized_model_names.tolist(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    ),
                ),
            )
            connection.executemany(
                "INSERT INTO annotations(annotation_id, annotation) VALUES (?, ?)",
                enumerate(annotation_names.tolist()),
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        except Exception:
            connection.close()
            database_path.unlink(missing_ok=True)
            raise
        finally:
            connection.close()

        try:
            with _load_mmappet().DatasetWriter.new(
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

    def annotations(self) -> AnnotationVocabulary:
        """Return the complete annotation vocabulary as NumPy arrays."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT annotation_id, annotation
                FROM annotations
                ORDER BY annotation_id
                """
            ).fetchall()
        return AnnotationVocabulary(
            ids=np.fromiter((row[0] for row in rows), dtype=np.uint16),
            names=np.asarray([row[1] for row in rows], dtype=object),
        )

    def model_names(self) -> npt.NDArray[np.object_]:
        """Return model provenance as a one-dimensional string array."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'model_names'"
            ).fetchone()
        if row is None:
            raise CacheValidationError("metadata is missing model_names")
        return _decode_model_names(row[0])

    def append(
        self,
        *,
        charge: int,
        collision_energy: float,
        sequence: str,
        predicted_intensities: npt.ArrayLike,
        annotation_ids: npt.ArrayLike,
    ) -> tuple[int, int]:
        """Append one prediction to the cache.

        Returns
        -------
        start : int
            Inclusive row offset of the appended vectors.
        end : int
            Exclusive row offset of the appended vectors.
        """

        intensities = np.asarray(predicted_intensities, dtype=np.float32)
        result = self.append_many(
            PredictionKeys.validate(
                charge=np.asarray([charge]),
                collision_energy=np.asarray([collision_energy]),
                sequence=np.asarray([sequence]),
            ),
            PackedPredictions.validate(
                predicted_intensities=intensities,
                annotation_ids=np.asarray(annotation_ids),
                offsets=np.asarray([0, len(intensities)], dtype=np.int64),
            ),
        )
        return int(result.starts[0]), int(result.ends[0])

    def append_many(
        self,
        keys: PredictionKeys,
        predictions: PackedPredictions,
    ) -> AppendResult:
        """Append a packed prediction batch with one write per mmappet column.

        Returns
        -------
        AppendResult
            ``starts`` and ``ends`` are ``int64`` arrays aligned with ``keys``.
            Each pair describes a half-open range ``[start, end)`` shared by
            the corresponding intensity and annotation-ID vectors.
        """

        if not isinstance(keys, PredictionKeys):
            raise TypeError("keys must be a PredictionKeys instance")
        if not isinstance(predictions, PackedPredictions):
            raise TypeError("predictions must be a PackedPredictions instance")
        keys = PredictionKeys.validate(
            charge=keys.charge,
            collision_energy=keys.collision_energy,
            sequence=keys.sequence,
        )
        predictions = PackedPredictions.validate(
            predicted_intensities=predictions.predicted_intensities,
            annotation_ids=predictions.annotation_ids,
            offsets=predictions.offsets,
        )
        if len(keys) != len(predictions):
            raise ValueError(
                f"keys has {len(keys)} entries but predictions has {len(predictions)}"
            )

        with self._connect() as connection:
            annotation_count = connection.execute(
                "SELECT COUNT(*) FROM annotations"
            ).fetchone()[0]
        if len(predictions.annotation_ids) and np.any(
            predictions.annotation_ids >= annotation_count
        ):
            raise ValueError(
                f"annotation_ids must be between 0 and {annotation_count - 1}"
            )

        self.lock_path.touch(exist_ok=True)
        with self.lock_path.open("rb") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    _prepare_requested_keys(connection, keys)
                    duplicate = connection.execute(
                        """
                        SELECT MIN(request_position)
                        FROM requested_keys
                        GROUP BY charge, collision_energy, sequence
                        HAVING COUNT(*) > 1
                        LIMIT 1
                        """
                    ).fetchone()
                    if duplicate is not None:
                        raise CacheKeyExistsError(
                            "duplicate cache keys in the submitted batch"
                        )

                    existing = connection.execute(
                        """
                        SELECT requested_keys.request_position
                        FROM requested_keys
                        JOIN cache_entries USING (
                            charge, collision_energy, sequence
                        )
                        ORDER BY requested_keys.request_position
                        """
                    ).fetchall()
                    if existing:
                        positions = [row[0] for row in existing]
                        raise CacheKeyExistsError(
                            "Cache keys already exist at submitted positions "
                            f"{positions}"
                        )

                    if len(keys):
                        with _load_mmappet().DatasetWriter(
                            self.dataset_path, append_ok=True
                        ) as writer:
                            storage_start = len(writer)
                            writer.append(
                                predicted_intensity=(predictions.predicted_intensities),
                                annotation_id=predictions.annotation_ids,
                            )
                            writer.flush()
                    else:
                        storage_start = self._storage_length()

                    ranges = storage_start + predictions.offsets
                    starts = ranges[:-1].astype(np.int64, copy=False)
                    ends = ranges[1:].astype(np.int64, copy=False)
                    connection.executemany(
                        """
                        INSERT INTO cache_entries(
                            charge, collision_energy, sequence, start, end
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        zip(
                            map(int, keys.charge),
                            map(float, keys.collision_energy),
                            map(str, keys.sequence),
                            map(int, starts),
                            map(int, ends),
                        ),
                    )
                    connection.commit()
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        return AppendResult(starts=starts, ends=ends)

    def lookup(self, keys: PredictionKeys) -> LookupResult:
        """Return mmap storage and result ranges aligned with submitted keys."""

        if not isinstance(keys, PredictionKeys):
            raise TypeError("keys must be a PredictionKeys instance")
        keys = PredictionKeys.validate(
            charge=keys.charge,
            collision_energy=keys.collision_energy,
            sequence=keys.sequence,
        )
        with self._connect() as connection:
            rows = _lookup_rows(connection, keys)

        starts = np.full(len(keys), -1, dtype=np.int64)
        ends = np.full(len(keys), -1, dtype=np.int64)
        found = np.zeros(len(keys), dtype=bool)
        for position, start, end in rows:
            if start is not None and end is not None:
                starts[position] = start
                ends[position] = end
                found[position] = True

        storage = _load_mmappet().open_dataset_dct(self.dataset_path)
        return LookupResult(
            predicted_intensities=storage[INTENSITY_COLUMN],
            annotation_ids=storage[ANNOTATION_COLUMN],
            starts=starts,
            ends=ends,
            found=found,
        )

    def validate(self) -> None:
        """Check schema versions, annotation numbering, and stored ranges."""

        storage = _load_mmappet().open_dataset_dct(self.dataset_path)
        if set(storage) != {INTENSITY_COLUMN, ANNOTATION_COLUMN}:
            raise CacheValidationError(f"Unexpected mmappet columns: {list(storage)}")
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

            annotation_ids = np.fromiter(
                (
                    row[0]
                    for row in connection.execute(
                        "SELECT annotation_id FROM annotations ORDER BY annotation_id"
                    )
                ),
                dtype=np.int64,
            )
            if not np.array_equal(
                annotation_ids, np.arange(len(annotation_ids), dtype=np.int64)
            ):
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
                raise CacheValidationError(f"Overlapping cache ranges found: {overlap}")

    def _storage_length(self) -> int:
        return len(
            _load_mmappet().open_dataset_dct(self.dataset_path)[INTENSITY_COLUMN]
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _normalize_integer_array(
    values: npt.ArrayLike,
    name: str,
    *,
    minimum: int,
    maximum: int = np.iinfo(np.int64).max,
    dtype: npt.DTypeLike = np.int64,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0:
        return np.empty(0, dtype=dtype)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    if len(array) and (np.any(array < minimum) or np.any(array > maximum)):
        raise ValueError(f"{name} values must be between {minimum} and {maximum}")
    return array.astype(dtype, copy=False)


def _normalize_collision_energy(values: npt.ArrayLike) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("collision_energy must be one-dimensional")
    if array.dtype.kind not in "iuf":
        raise TypeError("collision_energy must contain numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("collision_energy must contain finite values")
    return normalized


def _normalize_string_array(values: npt.ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0:
        return np.empty(0, dtype=object)
    if array.dtype.kind == "U":
        return array.astype(object)
    if array.dtype.kind != "O":
        raise TypeError(f"{name} must contain strings")
    valid = np.fromiter(
        (isinstance(value, str) for value in array),
        dtype=bool,
        count=len(array),
    )
    if not np.all(valid):
        raise TypeError(f"{name} must contain strings")
    return array


def _validate_names(values: npt.ArrayLike, name: str) -> np.ndarray:
    names = _normalize_string_array(values, name)
    if len(names) == 0:
        raise ValueError(f"{name} must contain at least one entry")
    nonempty = np.fromiter(
        (bool(value.strip()) for value in names), dtype=bool, count=len(names)
    )
    if not np.all(nonempty):
        raise ValueError(f"{name} entries must not be empty")
    if len(set(names.tolist())) != len(names):
        raise ValueError(f"{name} entries must be unique")
    return names


def _decode_model_names(value: str) -> np.ndarray:
    try:
        names = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise CacheValidationError(
            "metadata model_names must be a JSON string list"
        ) from error
    if not isinstance(names, list):
        raise CacheValidationError("metadata model_names must be a JSON string list")
    try:
        return _validate_names(names, "model_names")
    except (TypeError, ValueError) as error:
        raise CacheValidationError(f"Invalid metadata model_names: {error}") from error


def _prepare_requested_keys(
    connection: sqlite3.Connection, keys: PredictionKeys
) -> None:
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
        zip(
            range(len(keys)),
            map(int, keys.charge),
            map(float, keys.collision_energy),
            map(str, keys.sequence),
        ),
    )


def _lookup_rows(
    connection: sqlite3.Connection, keys: PredictionKeys
) -> list[tuple[int, int | None, int | None]]:
    _prepare_requested_keys(connection, keys)
    return connection.execute(
        """
        SELECT requested_keys.request_position,
               cache_entries.start,
               cache_entries.end
        FROM requested_keys
        LEFT JOIN cache_entries USING (charge, collision_energy, sequence)
        ORDER BY requested_keys.request_position
        """
    ).fetchall()
