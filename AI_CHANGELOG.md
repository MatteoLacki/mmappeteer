# AI_CHANGELOG — mmappeteer

## 2026-08-27 — configurable dtypes
- What: `create()` takes `annotation_id_dtype`/`intensity_dtype`; defaults stay
  `float32`/`uint16`, so featureprediction's RT/IIM cache and `sagepy-rescore`'s cache
  kept their on-disk shape. First narrower user: featureprediction's Prosit MS2 cache.

## 2026-09-18 — schema 4: parquet shards replace SQLite (schema 3)
- What: index moved from `index.sqlite3` to parquet shards queried by DuckDB;
  `python -m mmappeteer.migrate` converts a schema-3 cache.
- Why: schema 3's `append_rt`/`append_im` built `WHERE key IN (?,...)` per row and
  crashed on `too many SQL variables` at real scale; SQLite reads were row-at-a-time.
- Measured: warm F9477 export 13.4 s -> 4.4 s, anti-join to 3.0 s, index 800 MB -> 390 MB.
