# mmappeteer AI Guidance

## Purpose

`mmappeteer` is an append-only prediction cache. SQLite indexes keys and
half-open ranges; `mmappet` stores aligned, mmap-backed NumPy columns. The
canonical key is `(charge, collision_energy, sequence)`. Each key maps to an
intensity (`float32`) and annotation-ID (`uint16`) vector of equal length.

Start with `README.md`. Implementation is in `src/mmappeteer/cache.py`; tests
are in `tests/test_cache.py`.

## NumPy API

- `PredictionKeys` owns parallel `charge`, `collision_energy`, and `sequence`
  arrays of equal length.
- `PackedPredictions` represents ragged vectors as flat intensity and
  annotation-ID arrays plus `int64 offsets` of length `n + 1`.
- Construct both through their `.validate(...)` class methods. These normalize
  caller-provided array-like values before invoking the plain dataclass
  constructors; do not mutate frozen instances in `__post_init__`.
- `append_many()` performs one append per mmappet column for the complete batch.
- `AppendResult.starts` and `.ends` align one-to-one with submitted keys.
- `LookupResult.starts`, `.ends`, and `.found` also align one-to-one with
  submitted keys. Missing ranges are `[-1, -1)`.
- Lookup storage arrays remain mmap-backed; do not copy the complete cache.
- Keep the core free of pandas and DataFrame adapters.

## Storage invariants

- `charge` is an integer greater than or equal to one.
- Normalize collision energy to `float32` before insertion and lookup.
- Cache keys are unique. Never replace, update, or delete an existing entry.
- SQLite ranges are half-open: `[start, end)`.
- Both mmappet columns always have the same row count.
- Every stored annotation ID refers to the SQLite `annotations` table.
- Annotation IDs are contiguous from zero and match
  `metadata.annotation_count`.
- `metadata.model_names` is a non-empty JSON list of unique, non-empty names.
- Do not introduce preallocation or shards without a concrete need.

## Concurrency and write ordering

`write.lock` is an advisory POSIX `flock`. Every writer must acquire it;
readers do not. Never write directly to mmappet binary files.

SQLite and mmappet do not share an atomic transaction. Preserve this order:

1. Validate all keys, flat arrays, offsets, and annotation IDs.
2. Acquire `write.lock`.
3. Start the SQLite write transaction and reject duplicate/existing keys.
4. Append and flush both mmappet columns.
5. Insert ranges and commit SQLite.

A failed append can leave unreferenced tail data. Never make metadata visible
before its array bytes.

## Schema changes

The strict SQLite table has a composite primary key, which is also the lookup
index. If persisted schema semantics change, increment `SCHEMA_VERSION` and
add migration or explicit compatibility behavior and tests.

## Development

The neighboring mmappet checkout is configured at `../../mmappet`.

```bash
uv run --extra dev pytest -q
```

Add regression tests for schema creation, validation, packed append behavior,
submitted-key alignment, duplicate handling, and missing-key masks.
