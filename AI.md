# mmappeteer AI Guidance

## Purpose

`mmappeteer` is an append-only prediction cache. Parquet shards index keys
and half-open ranges; `mmappet` stores aligned, mmap-backed NumPy columns. The
canonical key is `(charge, collision_energy, sequence)`. Each key maps to an
intensity and annotation-ID vector of equal length -- `float32`/`uint16` by
default, or a narrower pair chosen at `create()` time (see "Configurable
dtypes" below). Two scalar caches (retention time, ion mobility) have their
own index directories and skip mmappet entirely -- see "Scalar caches" below.

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

## Scalar caches (RT/IM)

- `append_rt`/`lookup_rt` key on `sequence` alone; `append_im`/`lookup_im` key
  on `(sequence, charge)`. Both store one scalar (`float64`) per key in a
  own index directory (`rt_entries/`, `im_entries/`) -- no mmappet file, no
  ranges, since there's nothing ragged to pack.
- Both return `ScalarLookupResult` (`values`, `found`, `missing_positions`),
  the scalar counterpart to `LookupResult`.
- Same de-duplication and existing-key-raise rules as `append_many` (below)
  apply here too.

## Configurable dtypes (2026-08-27)

`PredictionCache.create()` accepts optional `annotation_id_dtype`
(`np.uint8`/`np.uint16`, default `np.uint16`) and `intensity_dtype`
(`np.float16`/`np.float32`, default `np.float32`) — a closed set, not
arbitrary numpy dtypes, since narrower widths exist for one concrete reason
(a small annotation vocabulary; a source predictor whose native output is
already lower-precision) rather than as a fully general knob. Chosen once
at `create()` time, recorded in `metadata`, and read back into
`self._intensity_dtype`/`self._annotation_id_dtype` on every subsequent
`PredictionCache(path)` open (unconditionally, not just under `validate=True`
— `append`/`append_many` need these regardless of whether validation runs).
`validate()` checks the *metadata* against this closed set (not a single
hardcoded literal) and then cross-checks the mmappet storage's actual dtype
against that metadata for self-consistency. Existing callers that never
pass these params are unaffected — defaults are exactly the original
`float32`/`uint16`, and `git/featureprediction`'s own RT/IIM cache and
`sagepy-rescore`'s intensity cache both still get that on-disk shape
unchanged. First real consumer of the narrower pair:
`git/featureprediction/fragment_intensity.py`'s Prosit MS2 cache
(`uint8`/`float16` — 174-entry vocabulary, FP16-native source model).

## Storage invariants

- `charge` is an integer greater than or equal to one.
- Normalize collision energy to `float32` before insertion and lookup.
- Cache keys are unique. Never replace, update, or delete an existing entry
  from a *previous* call -- `CacheKeyExistsError` guards this.
- Keys repeated *within one* `append_many`/`append_rt`/`append_im` call are
  de-duplicated automatically (first occurrence wins; every submitted
  position, including later repeats, gets that entry's range/value back).
  This is not a relaxation of the previous point -- it's resolving redundancy
  within a single caller-submitted batch, not replacing a stored entry.
- Index ranges are half-open: `[start, end)`.
- Both mmappet columns always have the same row count.
- Every stored annotation ID refers to `annotations.parquet`.
- Annotation IDs are contiguous from zero and match
  `metadata.annotation_count`.
- `metadata.model_names` is a non-empty JSON list of unique, non-empty names.
- Do not introduce preallocation without a concrete need.

## Concurrency and write ordering

`write.lock` is an advisory POSIX `flock`. Every writer must acquire it;
readers do not. Never write directly to mmappet binary files.

The index and mmappet do not share an atomic transaction. Preserve this order:

1. Validate all keys, flat arrays, offsets, and annotation IDs.
2. Acquire `write.lock`.
3. Reject duplicate/existing keys (a join against the index shards).
4. Append and flush both mmappet columns.
5. Write the new index shard.

A failed append can leave unreferenced tail data. Never make metadata visible
before its array bytes.

## On-disk layout (schema 4, 2026-09-18)

```
<cache>/
  metadata.parquet        key/value, rewritten whole
  annotations.parquet     annotation_id/annotation
  entries/*.parquet       charge, collision_energy, sequence, start, end
  rt_entries/*.parquet    sequence, retention_time
  im_entries/*.parquet    sequence, charge, ion_mobility
  arrays.mmappet          the ragged payload
  write.lock
```

Every append writes one **new** shard and never rewrites an existing one, the
same promise `arrays.mmappet` already makes; readers glob the directory and
take no lock. Lookups, existing-key checks and anti-joins all run through one
in-process DuckDB connection per cache instance (`_matching_scalar_entries`).
`compact()` merges a directory's shards into one file; shard count only ever
costs per-file open/footer parsing, never correctness.

Why not SQLite (which schema 3 used): its row-oriented B-tree, read through
DuckDB's `sqlite` scanner one row at a time, took **7.35s** to hand over the
F9477 fragment index's 16.87M rows, against **0.53s** for the join itself.
Columnar shards cut the warm F9477 export 13.4s -> 4.4s and the fill's
missing-key anti-join to 3.0s, and the index shrank 800MB -> 390MB. Key type
was never the problem: joining on a 64-bit hash instead of the sequence
string saved 0.2s and produced 18 false matches from hash collisions.

Convert a schema-3 cache with `python -m mmappeteer.migrate <cache>`; it reads
the old `index.sqlite3` through DuckDB, writes the parquet tables, leaves both
`index.sqlite3` and `arrays.mmappet` alone, and opens the result with full
validation.

A DuckDB caveat worth remembering: a `COPY ... TO ?` destination placeholder
binds *before* the copied query's own parameters, so the destination must be
inlined as a literal (`_sql_string`), not bound.

## Schema changes

If persisted schema semantics change, increment `SCHEMA_VERSION` and add
migration or explicit compatibility behavior and tests.

## Bulk key-membership checks at real scale

Never build a `WHERE key IN (?,?,?,...)` clause with one bound parameter per
submitted row. Schema 3 did this in `append_rt`/`append_im` and crashed on
SQLite's compiled variable limit (`too many SQL variables`) the first time
either ran at real scale -- millions of sequences. The parquet index has no
such limit, but the shape of the fix is the one to keep: register the
requested keys as one Arrow table and `JOIN`, never a parameter per row.


## Development

The neighboring mmappet checkout is configured at `../../mmappet`.

```bash
uv run --extra dev pytest -q
```

Add regression tests for schema creation, validation, packed append behavior,
submitted-key alignment, duplicate handling, and missing-key masks.
