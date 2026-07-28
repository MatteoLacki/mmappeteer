# mmappeteer

`mmappeteer` is an example of an append-only NumPy cache built from two small
pieces:

- SQLite stores searchable keys and half-open array ranges.
- `mmappet` stores the growing, mmap-backed
  NumPy columns.

The canonical cache maps `(charge, collision_energy, sequence)` to two aligned
vectors:

- predicted intensities (`float32`)
- annotation IDs (`uint16`)

Annotation IDs refer to a fixed vocabulary in the SQLite `annotations` table.

## Creating a cache

The real annotation vocabulary should be supplied in its canonical order. A
174-entry vocabulary is supported but not hard-coded:

```python
from mmappeteer import PredictionCache

annotation_names = [...]  # for example, the 174 possible annotations
cache = PredictionCache.create(
    "prediction-cache",
    annotation_names,
    model_names=["Prosit_2020_intensity_HCD"],
)
```

Creation writes:

```text
prediction-cache/
├── index.sqlite3
├── arrays.mmappet/
│   ├── 0.bin
│   ├── 1.bin
│   └── schema.txt
└── write.lock
```

The SQLite schema and contiguous annotation numbering are validated whenever
the cache is opened. Model provenance is stored as a JSON list in the
`metadata` table and is available as:

```python
cache.model_names()
# ("Prosit_2020_intensity_HCD",)
```

Multiple names can describe an ensemble or processing chain. They are
cache-level provenance and are not part of an individual entry's lookup key.

## Appending

```python
start, end = cache.append(
    charge=2,
    collision_energy=30.0,
    sequence="PEPTIDE",
    predicted_intensities=[0.2, 0.8],
    annotation_ids=[12, 98],
)
```

The mmappet files grow naturally. No preallocation or shard ID is needed.
Existing keys are never replaced.

Collision energy is normalized to `float32` before insertion and lookup.

## Ordered lookup

```python
import pandas as pd

keys = pd.DataFrame(
    {
        "charge": [3, 2],
        "collision_energy": [25.0, 30.0],
        "sequence": ["OTHER", "PEPTIDE"],
    }
)

result = cache.lookup(keys)
```

`result.hits` contains `start` and `end` in submitted-key order. They slice
both mmap-backed arrays:

```python
for start, end in zip(result.starts, result.ends):
    intensities = result.predicted_intensities[start:end]
    annotation_ids = result.annotation_ids[start:end]
```

`result.iter_arrays()` provides the same slices directly. Missing keys are
returned as `result.missing`, a DataFrame preserving their submission order
and original index.

The arrays are views over mmapped storage: lookup does not copy the complete
cache into memory.
