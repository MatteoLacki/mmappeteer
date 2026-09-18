"""Convert a schema-3 (SQLite-indexed) cache to the schema-4 parquet layout.

Reads `index.sqlite3` through DuckDB's sqlite scanner and writes
`metadata.parquet`, `annotations.parquet` and one shard per entry table.
`arrays.mmappet` is untouched -- only the index moves.

The old `index.sqlite3` is left in place: delete it once the converted cache
has been exercised. Run as::

    python -m mmappeteer.migrate <cache-directory> [...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cache import SCHEMA_VERSION, PredictionCache, _load_duckdb, _sql_string

LEGACY_DATABASE_FILENAME = "index.sqlite3"

_ENTRY_TABLES = (
    ("cache_entries", PredictionCache.entries_dirname),
    ("rt_cache_entries", PredictionCache.rt_entries_dirname),
    ("im_cache_entries", PredictionCache.im_entries_dirname),
)


def migrate(path: str | Path) -> None:
    """Write the parquet index for the SQLite-indexed cache at `path`."""

    path = Path(path)
    database_path = path / LEGACY_DATABASE_FILENAME
    if not database_path.is_file():
        raise FileNotFoundError(f"No legacy cache index at {database_path}")
    metadata_path = path / PredictionCache.metadata_filename
    if metadata_path.exists():
        raise FileExistsError(f"Already migrated: {metadata_path} exists")

    connection = _load_duckdb().connect()
    connection.execute("INSTALL sqlite; LOAD sqlite;")
    connection.execute(f"ATTACH '{database_path}' AS legacy (TYPE sqlite, READ_ONLY);")

    def copy(query: str, destination: Path, params: list | None = None) -> None:
        # The destination is inlined, not bound: DuckDB binds a COPY's TO
        # placeholder before the query's own parameters.
        connection.execute(
            f"COPY ({query}) TO {_sql_string(destination)} (FORMAT PARQUET)",
            params or [],
        )

    copy(
        "SELECT CAST(key AS VARCHAR) AS key, "
        "CASE WHEN key = 'schema_version' THEN ? ELSE CAST(value AS VARCHAR) END "
        "AS value FROM legacy.metadata",
        metadata_path,
        [str(SCHEMA_VERSION)],
    )
    copy(
        "SELECT CAST(annotation_id AS BIGINT) AS annotation_id, "
        "CAST(annotation AS VARCHAR) AS annotation FROM legacy.annotations "
        "ORDER BY annotation_id",
        path / PredictionCache.annotations_filename,
    )
    for table, dirname in _ENTRY_TABLES:
        directory = path / dirname
        directory.mkdir(exist_ok=True)
        copy(f"SELECT * FROM legacy.{table}", directory / "00000000.parquet")

    # Opening with validation is the check that the converted index and the
    # untouched mmappet storage still agree.
    PredictionCache(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", nargs="+", type=Path, help="cache directory")
    arguments = parser.parse_args(argv)
    for cache in arguments.cache:
        print(f"migrating {cache}", file=sys.stderr)
        migrate(cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
