from __future__ import annotations

import fcntl
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

SCHEMA_VERSION = 3
INTENSITY_COLUMN = "predicted_intensity"
ANNOTATION_COLUMN = "annotation_id"
MAX_ANNOTATIONS = np.iinfo(np.uint16).max + 1

# Allowed on-disk dtypes for the two mmappet columns. Deliberately a small,
# closed set rather than accepting arbitrary numpy dtypes -- narrower widths
# exist for one concrete reason (a small, e.g. <=256-entry, annotation
# vocabulary; a predictor whose native output is already float16) rather
# than as a fully general knob.
_ANNOTATION_ID_DTYPES = (np.uint8, np.uint16)
_INTENSITY_DTYPES = (np.float16, np.float32)


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
        intensity_dtype: npt.DTypeLike = np.float32,
        annotation_id_dtype: npt.DTypeLike = np.uint16,
    ) -> PackedPredictions:
        """Normalize arrays, validate packed invariants, and construct data.

        ``intensity_dtype``/``annotation_id_dtype`` default to this class's
        original fixed widths so every existing caller (that doesn't know
        about a cache's configured dtypes) is unaffected; ``PredictionCache``
        passes its own instance dtypes explicitly.
        """

        intensity_dtype = np.dtype(intensity_dtype)
        annotation_id_dtype = np.dtype(annotation_id_dtype)
        intensities = np.asarray(predicted_intensities, dtype=intensity_dtype)
        normalized_annotation_ids = _normalize_integer_array(
            annotation_ids,
            "annotation_ids",
            minimum=0,
            maximum=int(np.iinfo(annotation_id_dtype).max),
            dtype=annotation_id_dtype,
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
class ScalarLookupResult:
    """Scalar values and a found mask aligned with every submitted key.

    Used by the RT/IM scalar caches (one value per key, no ragged storage),
    unlike ``LookupResult``'s mmap-backed ranges for the intensity cache.
    """

    values: npt.NDArray[np.float64]
    found: npt.NDArray[np.bool_]

    @property
    def missing_positions(self) -> npt.NDArray[np.int64]:
        return np.flatnonzero(~self.found)


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

        # Read back this cache's own configured dtypes (chosen once, at
        # `create()` time) so `append`/`append_many` normalize against
        # whatever this instance was actually built with, not a fixed
        # float32/uint16 default -- needed functionally, not just for the
        # optional `validate()` check above, so this happens unconditionally.
        with self._connect() as connection:
            metadata = dict(
                connection.execute(
                    "SELECT key, value FROM metadata "
                    "WHERE key IN ('intensity_dtype', 'annotation_id_dtype')"
                )
            )
        self._intensity_dtype = np.dtype(metadata["intensity_dtype"])
        self._annotation_id_dtype = np.dtype(metadata["annotation_id_dtype"])

    @classmethod
    def create(
        cls,
        path: str | Path,
        annotations: npt.ArrayLike,
        *,
        model_names: str | npt.ArrayLike,
        annotation_id_dtype: npt.DTypeLike = np.uint16,
        intensity_dtype: npt.DTypeLike = np.float32,
    ) -> PredictionCache:
        """Create a cache for predictions produced by one or more models.

        ``annotation_id_dtype``/``intensity_dtype`` default to this class's
        original widths (`uint16`/`float32`) -- pass a narrower dtype (from
        `_ANNOTATION_ID_DTYPES`/`_INTENSITY_DTYPES`) when the annotation
        vocabulary is small and/or the source predictions are already
        lower-precision, to avoid storing false precision/range. Fixed once
        at creation time; every later `append`/`append_many`/`lookup` on
        this instance uses whatever was chosen here (read back from
        `metadata` in `__init__`).
        """

        path = Path(path)
        annotation_id_dtype = np.dtype(annotation_id_dtype)
        intensity_dtype = np.dtype(intensity_dtype)
        if annotation_id_dtype.type not in _ANNOTATION_ID_DTYPES:
            raise ValueError(
                f"annotation_id_dtype must be one of {_ANNOTATION_ID_DTYPES}, "
                f"got {annotation_id_dtype}"
            )
        if intensity_dtype.type not in _INTENSITY_DTYPES:
            raise ValueError(
                f"intensity_dtype must be one of {_INTENSITY_DTYPES}, "
                f"got {intensity_dtype}"
            )
        max_annotations = int(np.iinfo(annotation_id_dtype).max) + 1
        annotation_names = _validate_names(annotations, "annotations")
        if len(annotation_names) > max_annotations:
            raise ValueError(
                f"At most {max_annotations} annotations fit in "
                f"{annotation_id_dtype} storage"
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

                CREATE TABLE rt_cache_entries (
                    sequence       TEXT NOT NULL,
                    retention_time REAL NOT NULL,
                    PRIMARY KEY (sequence)
                ) STRICT, WITHOUT ROWID;

                CREATE TABLE im_cache_entries (
                    sequence     TEXT NOT NULL,
                    charge       INTEGER NOT NULL CHECK (charge >= 1),
                    ion_mobility REAL NOT NULL,
                    PRIMARY KEY (sequence, charge)
                ) STRICT, WITHOUT ROWID;
                """
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("annotation_count", str(len(annotation_names))),
                    ("collision_energy_dtype", "float32"),
                    ("intensity_dtype", intensity_dtype.name),
                    ("annotation_id_dtype", annotation_id_dtype.name),
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
                predicted_intensity=intensity_dtype,
                annotation_id=annotation_id_dtype,
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
            ids=np.fromiter((row[0] for row in rows), dtype=self._annotation_id_dtype),
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

        n = len(np.asarray(predicted_intensities))
        result = self.append_many(
            PredictionKeys.validate(
                charge=np.asarray([charge]),
                collision_energy=np.asarray([collision_energy]),
                sequence=np.asarray([sequence]),
            ),
            PackedPredictions.validate(
                predicted_intensities=predicted_intensities,
                annotation_ids=np.asarray(annotation_ids),
                offsets=np.asarray([0, n], dtype=np.int64),
                intensity_dtype=self._intensity_dtype,
                annotation_id_dtype=self._annotation_id_dtype,
            ),
        )
        return int(result.starts[0]), int(result.ends[0])

    def append_many(
        self,
        keys: PredictionKeys,
        predictions: PackedPredictions,
    ) -> AppendResult:
        """Append a packed prediction batch with one write per mmappet column.

        Keys repeated within the same batch are de-duplicated automatically
        (first occurrence wins) -- every submitted position still gets an
        aligned range in the result, including repeated positions, which
        point at the single physically-stored entry. A key that already
        exists in the cache from a *previous* call still raises
        ``CacheKeyExistsError`` -- callers are expected to filter
        previously-cached keys via ``lookup()`` first.

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
            intensity_dtype=self._intensity_dtype,
            annotation_id_dtype=self._annotation_id_dtype,
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

        first_occurrence = _first_occurrence_positions(
            list(
                zip(
                    map(int, keys.charge),
                    map(float, keys.collision_energy),
                    map(str, keys.sequence),
                )
            )
        )
        first_indices = np.flatnonzero(first_occurrence == np.arange(len(keys)))

        self.lock_path.touch(exist_ok=True)
        with self.lock_path.open("rb") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    _prepare_requested_keys(connection, keys)

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
                        (
                            (
                                int(keys.charge[i]),
                                float(keys.collision_energy[i]),
                                str(keys.sequence[i]),
                                int(starts[i]),
                                int(ends[i]),
                            )
                            for i in first_indices
                        ),
                    )
                    connection.commit()
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        # Duplicate positions report their group's first-occurrence range;
        # their own slice is still physically present in mmappet storage,
        # just unreferenced by any cache_entries row.
        aligned_starts = starts[first_occurrence]
        aligned_ends = ends[first_occurrence]
        return AppendResult(starts=aligned_starts, ends=aligned_ends)

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

    def append_rt(
        self,
        *,
        sequence: npt.ArrayLike,
        retention_time: npt.ArrayLike,
    ) -> None:
        """Append retention-time predictions keyed by sequence alone.

        Scalar-per-key, no ragged storage -- unlike ``append_many``, there is
        no mmappet array file or range to return. Keys repeated within the
        same batch are de-duplicated (first occurrence wins). A sequence
        that already has a cached retention time still raises
        ``CacheKeyExistsError``.
        """

        sequences = _normalize_string_array(sequence, "sequence")
        values = _normalize_float_array(retention_time, "retention_time")
        if len(sequences) != len(values):
            raise ValueError(
                f"sequence has {len(sequences)} entries but retention_time "
                f"has {len(values)}"
            )

        first_occurrence = _first_occurrence_positions(list(map(str, sequences)))
        first_indices = np.flatnonzero(
            first_occurrence == np.arange(len(sequences))
        )

        self.lock_path.touch(exist_ok=True)
        with self.lock_path.open("rb") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    if len(sequences):
                        placeholders = ",".join("?" * len(sequences))
                        existing = connection.execute(
                            f"""
                            SELECT sequence FROM rt_cache_entries
                            WHERE sequence IN ({placeholders})
                            """,
                            [str(s) for s in sequences],
                        ).fetchall()
                        if existing:
                            raise CacheKeyExistsError(
                                "Retention-time cache keys already exist: "
                                f"{sorted(row[0] for row in existing)}"
                            )
                    connection.executemany(
                        """
                        INSERT INTO rt_cache_entries(sequence, retention_time)
                        VALUES (?, ?)
                        """,
                        (
                            (str(sequences[i]), float(values[i]))
                            for i in first_indices
                        ),
                    )
                    connection.commit()
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def lookup_rt(self, *, sequence: npt.ArrayLike) -> ScalarLookupResult:
        """Return retention-time values and a found mask aligned with keys."""

        sequences = _normalize_string_array(sequence, "sequence")
        values = np.zeros(len(sequences), dtype=np.float64)
        found = np.zeros(len(sequences), dtype=bool)
        if len(sequences):
            with self._connect() as connection:
                connection.execute("DROP TABLE IF EXISTS temp.requested_rt_keys")
                connection.execute(
                    """
                    CREATE TEMP TABLE requested_rt_keys (
                        request_position INTEGER PRIMARY KEY,
                        sequence         TEXT NOT NULL
                    ) STRICT
                    """
                )
                connection.executemany(
                    """
                    INSERT INTO requested_rt_keys(request_position, sequence)
                    VALUES (?, ?)
                    """,
                    enumerate(map(str, sequences)),
                )
                rows = connection.execute(
                    """
                    SELECT requested_rt_keys.request_position,
                           rt_cache_entries.retention_time
                    FROM requested_rt_keys
                    LEFT JOIN rt_cache_entries USING (sequence)
                    ORDER BY requested_rt_keys.request_position
                    """
                ).fetchall()
            for position, retention_time in rows:
                if retention_time is not None:
                    values[position] = retention_time
                    found[position] = True
        return ScalarLookupResult(values=values, found=found)

    def append_im(
        self,
        *,
        sequence: npt.ArrayLike,
        charge: npt.ArrayLike,
        ion_mobility: npt.ArrayLike,
    ) -> None:
        """Append ion-mobility predictions keyed by ``(sequence, charge)``.

        Scalar-per-key, same shape as ``append_rt``. Keys repeated within
        the same batch are de-duplicated (first occurrence wins). A key
        that already has a cached ion mobility still raises
        ``CacheKeyExistsError``.
        """

        sequences = _normalize_string_array(sequence, "sequence")
        charges = _normalize_integer_array(charge, "charge", minimum=1)
        values = _normalize_float_array(ion_mobility, "ion_mobility")
        lengths = (len(sequences), len(charges), len(values))
        if len(set(lengths)) != 1:
            raise ValueError(
                f"sequence, charge, and ion_mobility must have equal lengths; "
                f"got {lengths}"
            )

        first_occurrence = _first_occurrence_positions(
            list(zip(map(str, sequences), map(int, charges)))
        )
        first_indices = np.flatnonzero(
            first_occurrence == np.arange(len(sequences))
        )

        self.lock_path.touch(exist_ok=True)
        with self.lock_path.open("rb") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    if len(sequences):
                        pairs = ",".join("(?, ?)" for _ in sequences)
                        existing = connection.execute(
                            f"""
                            SELECT sequence, charge FROM im_cache_entries
                            WHERE (sequence, charge) IN ({pairs})
                            """,
                            [
                                value
                                for seq, ch in zip(sequences, charges)
                                for value in (str(seq), int(ch))
                            ],
                        ).fetchall()
                        if existing:
                            raise CacheKeyExistsError(
                                "Ion-mobility cache keys already exist: "
                                f"{sorted(existing)}"
                            )
                    connection.executemany(
                        """
                        INSERT INTO im_cache_entries(sequence, charge, ion_mobility)
                        VALUES (?, ?, ?)
                        """,
                        (
                            (str(sequences[i]), int(charges[i]), float(values[i]))
                            for i in first_indices
                        ),
                    )
                    connection.commit()
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def lookup_im(
        self, *, sequence: npt.ArrayLike, charge: npt.ArrayLike
    ) -> ScalarLookupResult:
        """Return ion-mobility values and a found mask aligned with keys."""

        sequences = _normalize_string_array(sequence, "sequence")
        charges = _normalize_integer_array(charge, "charge", minimum=1)
        if len(sequences) != len(charges):
            raise ValueError(
                f"sequence has {len(sequences)} entries but charge has "
                f"{len(charges)}"
            )
        values = np.zeros(len(sequences), dtype=np.float64)
        found = np.zeros(len(sequences), dtype=bool)
        if len(sequences):
            with self._connect() as connection:
                connection.execute("DROP TABLE IF EXISTS temp.requested_im_keys")
                connection.execute(
                    """
                    CREATE TEMP TABLE requested_im_keys (
                        request_position INTEGER PRIMARY KEY,
                        sequence         TEXT NOT NULL,
                        charge           INTEGER NOT NULL
                    ) STRICT
                    """
                )
                connection.executemany(
                    """
                    INSERT INTO requested_im_keys(request_position, sequence, charge)
                    VALUES (?, ?, ?)
                    """,
                    zip(range(len(sequences)), map(str, sequences), map(int, charges)),
                )
                rows = connection.execute(
                    """
                    SELECT requested_im_keys.request_position,
                           im_cache_entries.ion_mobility
                    FROM requested_im_keys
                    LEFT JOIN im_cache_entries USING (sequence, charge)
                    ORDER BY requested_im_keys.request_position
                    """
                ).fetchall()
            for position, ion_mobility in rows:
                if ion_mobility is not None:
                    values[position] = ion_mobility
                    found[position] = True
        return ScalarLookupResult(values=values, found=found)

    def validate(self) -> None:
        """Check schema versions, annotation numbering, and stored ranges."""

        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise CacheValidationError(
                    f"Unsupported schema version {version}; expected {SCHEMA_VERSION}"
                )

            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if metadata.get("schema_version") != str(SCHEMA_VERSION):
                raise CacheValidationError(
                    f"Invalid metadata 'schema_version': "
                    f"{metadata.get('schema_version')!r}"
                )
            if metadata.get("collision_energy_dtype") != "float32":
                raise CacheValidationError(
                    "Invalid metadata 'collision_energy_dtype': "
                    f"{metadata.get('collision_energy_dtype')!r}"
                )
            # Narrower `intensity_dtype`/`annotation_id_dtype` are allowed
            # (see `_INTENSITY_DTYPES`/`_ANNOTATION_ID_DTYPES`) -- validated
            # against that closed set, not a single fixed literal, then
            # cross-checked against the mmappet storage's actual dtype below
            # for self-consistency.
            intensity_dtype_name = metadata.get("intensity_dtype")
            annotation_id_dtype_name = metadata.get("annotation_id_dtype")
            if intensity_dtype_name not in {d.__name__ for d in _INTENSITY_DTYPES}:
                raise CacheValidationError(
                    f"Invalid metadata 'intensity_dtype': {intensity_dtype_name!r}"
                )
            if annotation_id_dtype_name not in {
                d.__name__ for d in _ANNOTATION_ID_DTYPES
            }:
                raise CacheValidationError(
                    "Invalid metadata 'annotation_id_dtype': "
                    f"{annotation_id_dtype_name!r}"
                )
            if "model_names" not in metadata:
                raise CacheValidationError("metadata is missing model_names")
            _decode_model_names(metadata["model_names"])

            storage = _load_mmappet().open_dataset_dct(self.dataset_path)
            if set(storage) != {INTENSITY_COLUMN, ANNOTATION_COLUMN}:
                raise CacheValidationError(
                    f"Unexpected mmappet columns: {list(storage)}"
                )
            if storage[INTENSITY_COLUMN].dtype != np.dtype(intensity_dtype_name):
                raise CacheValidationError(
                    f"predicted_intensity dtype {storage[INTENSITY_COLUMN].dtype} "
                    f"does not match metadata {intensity_dtype_name!r}"
                )
            if storage[ANNOTATION_COLUMN].dtype != np.dtype(annotation_id_dtype_name):
                raise CacheValidationError(
                    f"annotation_id dtype {storage[ANNOTATION_COLUMN].dtype} "
                    f"does not match metadata {annotation_id_dtype_name!r}"
                )
            if len(storage[INTENSITY_COLUMN]) != len(storage[ANNOTATION_COLUMN]):
                raise CacheValidationError("mmappet columns have unequal lengths")

            existing_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            missing_tables = (
                {"cache_entries", "rt_cache_entries", "im_cache_entries"}
                - existing_tables
            )
            if missing_tables:
                raise CacheValidationError(
                    f"Missing required tables: {sorted(missing_tables)}"
                )

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


def _first_occurrence_positions(key_tuples: list) -> np.ndarray:
    """For each position, return the index of its group's first occurrence.

    Shared de-duplication logic for ``append_many``/``append_rt``/
    ``append_im``: a position whose key hasn't been seen before maps to
    itself; a repeat maps to the earlier position holding the same key.
    """

    seen: dict = {}
    first_occurrence = np.empty(len(key_tuples), dtype=np.intp)
    for i, key in enumerate(key_tuples):
        first_occurrence[i] = seen.setdefault(key, i)
    return first_occurrence


def _normalize_float_array(values: npt.ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0:
        return np.empty(0, dtype=np.float64)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{name} must contain finite values")
    return normalized


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
