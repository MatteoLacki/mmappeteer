# mmappeteer AI Guidance

## Purpose

`mmappeteer` is an append-only prediction cache:

- SQLite indexes cache keys and half-open ranges.
- `mmappet` stores aligned, mmap-backed NumPy columns.
- The canonical key is `(charge, collision_energy, sequence)`.
- Each key maps to predicted-intensity (`float32`) and annotation-ID (`uint16`)
  vectors of equal length.

Start with `README.md` for the public workflow. The implementation is in
`src/mmappeteer/cache.py`; behavioral tests are in `tests/test_cache.py`.

## Storage invariants

- `charge` is an integer greater than or equal to one.
- Normalize collision energy to `float32` before both insertion and lookup.
- Cache keys are unique. Never replace, update, or delete an existing entry.
- SQLite ranges are half-open: `[start, end)`.
- Both mmappet columns always have the same row count.
- Every stored annotation ID must refer to the SQLite `annotations` table.
- Annotation IDs are contiguous from zero. `metadata.annotation_count` must
  equal the number of annotation rows.
- `metadata.model_names` is a non-empty JSON list of unique, non-empty model
  names. It records cache-level provenance and is immutable after creation.
- `LookupResult.hits` preserves the relative order and pandas index of matching
  submitted keys. `LookupResult.missing` does the same for absent keys.
- Lookup arrays remain mmap-backed; avoid copying the complete storage.

Do not introduce preallocation or multiple shards without a concrete need. The
current mmappet dataset grows through normal append operations.

## Concurrency and write ordering

`write.lock` is an advisory POSIX `flock`. Every writer must acquire it; readers
do not. Do not write directly to the mmappet binary files.

SQLite and mmappet do not share an atomic transaction. Preserve this order:

1. Validate all keys and arrays.
2. Acquire `write.lock`.
3. Start the SQLite write transaction and reject existing keys.
4. Append and flush both mmappet columns.
5. Insert the ranges and commit SQLite.

This ensures committed metadata never intentionally points to unwritten data.
A failed append can leave unreferenced tail data; do not make metadata visible
before its array bytes.

## Schema changes

The SQLite schema uses strict tables and a composite primary key, which is also
the lookup index. Keep dtype and annotation metadata consistent with the
mmappet schema. If persisted schema semantics change, increment
`SCHEMA_VERSION` and add migration or explicit compatibility behavior and
tests.

## Development

The neighboring mmappet checkout is configured through `tool.uv.sources` at
`../../mmappet`.

Run the complete test suite:

```bash
uv run --extra dev pytest -q
```

Add regression tests for changes to schema creation, validation, append
behavior, submitted-key ordering, duplicate handling, or missing-key results.
