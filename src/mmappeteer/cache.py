from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

SCHEMA_VERSION = 4
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


def _load_duckdb():
    """Import the query engine only when the index is read or written."""

    import duckdb

    return duckdb


def _load_pyarrow():
    """Import Arrow only when index shards are written or keys registered."""

    import pyarrow

    return pyarrow


class CacheError(RuntimeError):
    """Base class for cache-specific failures."""


class CacheKeyExistsError(CacheError):
    """Raised when an append would replace an existing cache entry."""


class CacheValidationError(CacheError):
    """Raised when the parquet index and mmappet storage disagree."""


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

    metadata_filename = "metadata.parquet"
    annotations_filename = "annotations.parquet"
    entries_dirname = "entries"
    rt_entries_dirname = "rt_entries"
    im_entries_dirname = "im_entries"
    dataset_filename = "arrays.mmappet"
    lock_filename = "write.lock"

    def __init__(self, path: str | Path, *, validate: bool = True):
        self.path = Path(path)
        self.metadata_path = self.path / self.metadata_filename
        self.annotations_path = self.path / self.annotations_filename
        self.entries_path = self.path / self.entries_dirname
        self.rt_entries_path = self.path / self.rt_entries_dirname
        self.im_entries_path = self.path / self.im_entries_dirname
        self.dataset_path = self.path / self.dataset_filename
        self.lock_path = self.path / self.lock_filename

        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Cache metadata not found: {self.metadata_path}")
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Cache dataset not found: {self.dataset_path}")
        self._connection = None
        if validate:
            self.validate()

        # Read back this cache's own configured dtypes (chosen once, at
        # `create()` time) so `append`/`append_many` normalize against
        # whatever this instance was actually built with, not a fixed
        # float32/uint16 default -- needed functionally, not just for the
        # optional `validate()` check above, so this happens unconditionally.
        metadata = self._metadata()
        self._intensity_dtype = np.dtype(metadata["intensity_dtype"])
        self._annotation_id_dtype = np.dtype(metadata["annotation_id_dtype"])

    @classmethod
    def exists(cls, path: str | Path) -> bool:
        """Whether `path` holds a cache this class can open."""

        path = Path(path)
        return (path / cls.metadata_filename).is_file() and (
            path / cls.dataset_filename
        ).is_dir()

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
        dataset_path = path / cls.dataset_filename
        metadata_path = path / cls.metadata_filename

        path.mkdir(parents=True, exist_ok=True)
        if metadata_path.exists() or dataset_path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing cache at {path}")

        try:
            _write_table(
                metadata_path,
                {
                    "key": np.array(
                        [
                            "schema_version",
                            "annotation_count",
                            "collision_energy_dtype",
                            "intensity_dtype",
                            "annotation_id_dtype",
                            "model_names",
                        ],
                        dtype=object,
                    ),
                    "value": np.array(
                        [
                            str(SCHEMA_VERSION),
                            str(len(annotation_names)),
                            "float32",
                            intensity_dtype.name,
                            annotation_id_dtype.name,
                            json.dumps(
                                normalized_model_names.tolist(),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        ],
                        dtype=object,
                    ),
                },
            )
            _write_table(
                path / cls.annotations_filename,
                {
                    "annotation_id": np.arange(len(annotation_names), dtype=np.int64),
                    "annotation": annotation_names,
                },
            )
            for dirname in (
                cls.entries_dirname,
                cls.rt_entries_dirname,
                cls.im_entries_dirname,
            ):
                (path / dirname).mkdir(exist_ok=True)
        except Exception:
            metadata_path.unlink(missing_ok=True)
            raise

        try:
            with _load_mmappet().DatasetWriter.new(
                dataset_path,
                predicted_intensity=intensity_dtype,
                annotation_id=annotation_id_dtype,
            ):
                pass
        except Exception:
            metadata_path.unlink(missing_ok=True)
            raise

        (path / cls.lock_filename).touch(exist_ok=True)
        return cls(path)

    def annotations(self) -> AnnotationVocabulary:
        """Return the complete annotation vocabulary as NumPy arrays."""

        rows = (
            self._connect()
            .execute(
                "SELECT annotation_id, annotation FROM read_parquet(?) "
                "ORDER BY annotation_id",
                [str(self.annotations_path)],
            )
            .fetchall()
        )
        return AnnotationVocabulary(
            ids=np.fromiter((row[0] for row in rows), dtype=self._annotation_id_dtype),
            names=np.asarray([row[1] for row in rows], dtype=object),
        )

    def model_names(self) -> npt.NDArray[np.object_]:
        """Return model provenance as a one-dimensional string array."""

        value = self._metadata().get("model_names")
        if value is None:
            raise CacheValidationError("metadata is missing model_names")
        return _decode_model_names(value)

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

        annotation_count = int(self._metadata()["annotation_count"])
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
                existing = self._matching_entries(keys, columns="1")
                if existing is not None:
                    positions = np.sort(existing["request_position"]).tolist()
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
                # The shard lands after the mmappet append: a crash in between
                # leaves unreferenced storage rows, never an index entry
                # pointing at rows that were never written.
                _append_shard(
                    self.entries_path,
                    {
                        "charge": keys.charge[first_indices],
                        "collision_energy": keys.collision_energy[
                            first_indices
                        ].astype(np.float64),
                        "sequence": keys.sequence[first_indices],
                        "start": starts[first_indices],
                        "end": ends[first_indices],
                    },
                )
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
        starts = np.full(len(keys), -1, dtype=np.int64)
        ends = np.full(len(keys), -1, dtype=np.int64)
        found = np.zeros(len(keys), dtype=bool)
        matches = self._matching_entries(keys, columns='e.start AS start, e."end" AS "end"')
        if matches is not None:
            positions = matches["request_position"]
            starts[positions] = matches["start"]
            ends[positions] = matches["end"]
            found[positions] = True

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
                existing = self._matching_scalar_entries(
                    self.rt_entries_path,
                    {"sequence": sequences},
                    columns="e.sequence AS sequence",
                )
                if existing is not None:
                    raise CacheKeyExistsError(
                        "Retention-time cache keys already exist: "
                        f"{sorted(str(value) for value in existing['sequence'])}"
                    )
                _append_shard(
                    self.rt_entries_path,
                    {
                        "sequence": sequences[first_indices],
                        "retention_time": values[first_indices],
                    },
                )
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def lookup_rt(self, *, sequence: npt.ArrayLike) -> ScalarLookupResult:
        """Return retention-time values and a found mask aligned with keys."""

        sequences = _normalize_string_array(sequence, "sequence")
        values = np.zeros(len(sequences), dtype=np.float64)
        found = np.zeros(len(sequences), dtype=bool)
        matches = self._matching_scalar_entries(
            self.rt_entries_path,
            {"sequence": sequences},
            columns="e.retention_time AS value",
        )
        if matches is not None:
            positions = matches["request_position"]
            values[positions] = matches["value"]
            found[positions] = True
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
                existing = self._matching_scalar_entries(
                    self.im_entries_path,
                    {"sequence": sequences, "charge": charges},
                    columns='e.sequence AS sequence, e.charge AS charge',
                )
                if existing is not None:
                    raise CacheKeyExistsError(
                        "Ion-mobility cache keys already exist: "
                        f"{sorted(zip(map(str, existing['sequence']), map(int, existing['charge'])))}"
                    )
                _append_shard(
                    self.im_entries_path,
                    {
                        "sequence": sequences[first_indices],
                        "charge": charges[first_indices],
                        "ion_mobility": values[first_indices],
                    },
                )
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
        matches = self._matching_scalar_entries(
            self.im_entries_path,
            {"sequence": sequences, "charge": charges},
            columns="e.ion_mobility AS value",
        )
        if matches is not None:
            positions = matches["request_position"]
            values[positions] = matches["value"]
            found[positions] = True
        return ScalarLookupResult(values=values, found=found)

    def validate(self) -> None:
        """Check schema versions, annotation numbering, and stored ranges."""

        metadata = self._metadata()
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
        if annotation_id_dtype_name not in {d.__name__ for d in _ANNOTATION_ID_DTYPES}:
            raise CacheValidationError(
                "Invalid metadata 'annotation_id_dtype': "
                f"{annotation_id_dtype_name!r}"
            )
        if "model_names" not in metadata:
            raise CacheValidationError("metadata is missing model_names")
        _decode_model_names(metadata["model_names"])

        storage = _load_mmappet().open_dataset_dct(self.dataset_path)
        if set(storage) != {INTENSITY_COLUMN, ANNOTATION_COLUMN}:
            raise CacheValidationError(f"Unexpected mmappet columns: {list(storage)}")
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

        missing_directories = sorted(
            path.name
            for path in (self.entries_path, self.rt_entries_path, self.im_entries_path)
            if not path.is_dir()
        )
        if missing_directories:
            raise CacheValidationError(
                f"Missing required directories: {missing_directories}"
            )
        if not self.annotations_path.is_file():
            raise CacheValidationError(
                f"Missing annotations table: {self.annotations_path}"
            )

        annotation_ids = np.asarray(
            self._connect()
            .execute(
                "SELECT annotation_id FROM read_parquet(?) ORDER BY annotation_id",
                [str(self.annotations_path)],
            )
            .fetchnumpy()["annotation_id"],
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

        shards = self._shards(self.entries_path)
        if not shards:
            return
        bad_range = (
            self._connect()
            .execute(
                """
                SELECT charge, collision_energy, sequence, start, "end"
                FROM read_parquet(?)
                WHERE "end" > ?
                LIMIT 1
                """,
                [shards, len(storage[INTENSITY_COLUMN])],
            )
            .fetchone()
        )
        if bad_range is not None:
            raise CacheValidationError(
                f"Cache entry points outside mmappet storage: {bad_range}"
            )

        overlap = (
            self._connect()
            .execute(
                """
                WITH ordered AS (
                    SELECT start, "end",
                           MAX("end") OVER (
                               ORDER BY start, "end"
                               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                           ) AS previous_max_end
                    FROM read_parquet(?)
                )
                SELECT start, "end", previous_max_end
                FROM ordered
                WHERE start < previous_max_end
                LIMIT 1
                """,
                [shards],
            )
            .fetchone()
        )
        if overlap is not None:
            raise CacheValidationError(f"Overlapping cache ranges found: {overlap}")

    def compact(self) -> None:
        """Merge each index directory's shards into a single parquet file.

        Purely a read-performance measure -- shard count only ever costs
        per-file open/footer parsing, never correctness.
        """

        self.lock_path.touch(exist_ok=True)
        with self.lock_path.open("rb") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                for directory in (
                    self.entries_path,
                    self.rt_entries_path,
                    self.im_entries_path,
                ):
                    shards = self._shards(directory)
                    if len(shards) < 2:
                        continue
                    merged = directory / "compacting.parquet"
                    # The destination cannot be a bound parameter: DuckDB
                    # binds a COPY's TO placeholder before the query's own.
                    self._connect().execute(
                        "COPY (SELECT * FROM read_parquet(?)) "
                        f"TO {_sql_string(merged)} (FORMAT PARQUET)",
                        [shards],
                    )
                    for shard in shards:
                        Path(shard).unlink()
                    merged.rename(directory / _shard_filename(0))
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _storage_length(self) -> int:
        return len(
            _load_mmappet().open_dataset_dct(self.dataset_path)[INTENSITY_COLUMN]
        )

    def _connect(self):
        """One in-process DuckDB connection per cache instance."""

        if self._connection is None:
            self._connection = _load_duckdb().connect()
        return self._connection

    def _metadata(self) -> dict[str, str]:
        rows = (
            self._connect()
            .execute(
                "SELECT key, value FROM read_parquet(?)", [str(self.metadata_path)]
            )
            .fetchall()
        )
        return {str(key): str(value) for key, value in rows}

    @staticmethod
    def _shards(directory: Path) -> list[str]:
        return sorted(str(path) for path in directory.glob("*.parquet"))

    def _matching_entries(self, keys: PredictionKeys, *, columns: str):
        """Join `keys` against the intensity index; None when nothing matches."""

        return self._matching_scalar_entries(
            self.entries_path,
            {
                "charge": keys.charge,
                "collision_energy": keys.collision_energy.astype(np.float64),
                "sequence": keys.sequence,
            },
            columns=columns,
        )

    def _matching_scalar_entries(
        self, directory: Path, key_columns: dict[str, np.ndarray], *, columns: str
    ):
        """Inner-join requested keys against one index directory.

        Returns the matched rows as NumPy arrays (always including
        `request_position`), or None when there is nothing to match -- no
        requested keys, no shards, or no hits.
        """

        n_requested = len(next(iter(key_columns.values())))
        shards = self._shards(directory)
        if n_requested == 0 or not shards:
            return None
        pa = _load_pyarrow()
        requested = pa.table(
            {
                "request_position": np.arange(n_requested, dtype=np.int64),
                **{name: _arrow_column(values) for name, values in key_columns.items()},
            }
        )
        using = ", ".join(key_columns)
        connection = self._connect()
        connection.register("requested_keys", requested)
        try:
            matches = connection.execute(
                f"""
                SELECT r.request_position AS request_position, {columns}
                FROM requested_keys r
                JOIN read_parquet(?) e USING ({using})
                """,
                [shards],
            ).fetchnumpy()
        finally:
            connection.unregister("requested_keys")
        if len(matches["request_position"]) == 0:
            return None
        return matches


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
    # A list goes straight to an object array holding the same str objects.
    # Plain `np.asarray` would first build a fixed-width unicode array sized
    # to the longest string -- 4.56 GB for F9477's 6.3M peptides at <U181.
    if isinstance(values, np.ndarray):
        array = values
    else:
        array = np.asarray(values, dtype=object)
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


def _sql_string(value) -> str:
    """Quote `value` as a SQL string literal."""

    return "'" + str(value).replace("'", "''") + "'"


def _shard_filename(index: int) -> str:
    return f"{index:08d}.parquet"


def _arrow_column(values: np.ndarray):
    """Arrow-ready view of one key column (object arrays become strings)."""

    if values.dtype == object:
        return _load_pyarrow().array(values.tolist(), type=_load_pyarrow().string())
    return values


def _write_table(path: Path, columns: dict[str, np.ndarray]) -> None:
    pa = _load_pyarrow()
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table({name: _arrow_column(values) for name, values in columns.items()}),
        path,
    )


def _append_shard(directory: Path, columns: dict[str, np.ndarray]) -> None:
    """Write one new parquet shard; never touches the existing ones.

    Callers hold the cache's write lock, so picking the next free number by
    looking at the directory cannot race another writer.
    """

    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob("*.parquet"))
    next_index = 0
    if existing:
        next_index = int(existing[-1].stem) + 1
    _write_table(directory / _shard_filename(next_index), columns)
