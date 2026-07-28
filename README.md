# mmappeteer

`mmappeteer` is an append-only NumPy prediction cache built from two pieces:

- SQLite stores searchable keys and half-open array ranges.
- `mmappet` stores growing, mmap-backed NumPy columns.

The canonical cache maps `(charge, collision_energy, sequence)` to two aligned
vectors: predicted intensities (`float32`) and annotation IDs (`uint16`). Its
public API uses NumPy arrays rather than pandas DataFrames.

## Create a cache

Supply annotation names in canonical order. Their row numbers become the
annotation IDs stored in mmappet.

```python
import numpy as np
from mmappeteer import PredictionCache

cache = PredictionCache.create(
    "prediction-cache",
    annotations=np.asarray([...]),  # for example, 174 annotation names
    model_names=np.asarray(["Prosit_2020_intensity_HCD"]),
)
```

The cache-level model provenance and annotation vocabulary are available as
NumPy arrays:

```python
cache.model_names()
cache.annotations().ids
cache.annotations().names
```

## Append one prediction

```python
start, end = cache.append(
    charge=2,
    collision_energy=30.0,
    sequence="PEPTIDE",
    predicted_intensities=np.asarray([0.2, 0.8], dtype=np.float32),
    annotation_ids=np.asarray([12, 98], dtype=np.uint16),
)
```

`start` is inclusive and `end` is exclusive.

## Append a packed batch

Variable-length predictions use flattened value arrays plus offsets. This lets
the complete batch reach mmappet in one append operation per storage column.

```python
from mmappeteer import PackedPredictions, PredictionKeys

keys = PredictionKeys.validate(
    charge=np.asarray([2, 3]),
    collision_energy=np.asarray([30.0, 25.0]),
    sequence=np.asarray(["PEPTIDE", "OTHER"]),
)

predictions = PackedPredictions.validate(
    predicted_intensities=np.asarray(
        [0.2, 0.8, 0.1, 0.3, 0.6], dtype=np.float32
    ),
    annotation_ids=np.asarray([12, 98, 4, 27, 91], dtype=np.uint16),
    offsets=np.asarray([0, 2, 5], dtype=np.int64),
)

ranges = cache.append_many(keys, predictions)
# ranges.starts == [0, 2]
# ranges.ends   == [2, 5]
```

For `n` keys, `offsets` has `n + 1` entries, starts at zero, is
non-decreasing, and ends at the flattened vector length.

## Ordered lookup

Lookup metadata has exactly one element per submitted key:

```python
requested = PredictionKeys.validate(
    charge=np.asarray([3, 1, 2]),
    collision_energy=np.asarray([25.0, 20.0, 30.0]),
    sequence=np.asarray(["OTHER", "MISSING", "PEPTIDE"]),
)

result = cache.lookup(requested)

# result.starts == [2, -1, 0]
# result.ends   == [5, -1, 2]
# result.found  == [True, False, True]
```

Missing keys have `start == end == -1`. Recover them from the original key
arrays without a separate table abstraction:

```python
missing_keys = requested.take(result.missing_positions)
```

The complete storage arrays remain mmap-backed. Found slices can be consumed
without copying:

```python
for intensities, annotation_ids in result.iter_arrays():
    ...
```

Collision energy is normalized to `float32` before insertion and lookup.
Existing cache keys are never replaced. Writers coordinate with the advisory
`write.lock`; readers do not acquire it.
