"""
Experiment: Force DuckDB to use a chosen internal vector type,
rebuild DuckDB, and benchmark query performance.

What this script does:
1. Backs up DuckDB source file(s).
2. Patches DuckDB's vector constructor/default vector type in src/common/types/vector.cpp.
3. Rebuilds DuckDB.
4. Creates a TPC-H database.
5. Runs benchmark queries multiple times.
6. Records query time and database file size.
7. Restores the original source code at the end.

This version is stricter and more stable than regex-searching multiple files.
"""

import os
import re
import csv
import time
import shutil
import subprocess
from pathlib import Path

# CONFIG

DUCKDB_ROOT = Path(__file__).resolve().parent.parent / "duckdb"
BUILD_DIR = DUCKDB_ROOT / "build"
DUCKDB_BIN = BUILD_DIR / "duckdb"

# The file we actually patch.
VECTOR_CPP = DUCKDB_ROOT / "src/common/types/vector.cpp"

# Vector types to test.
# Start with FLAT_VECTOR and DICTIONARY_VECTOR first.
TARGET_VECTOR_TYPES = [
    "CONSTANT_VECTOR",
    "FLAT_VECTOR",
]

REPEATS = 3
TPCH_SF = 0.1
RESULTS_CSV = Path(__file__).parent / "duckdb_vector_type_results.csv"

# Queries
Q1 = """
SELECT l_returnflag, l_linestatus,
       SUM(l_quantity), SUM(l_extendedprice)
FROM lineitem
GROUP BY l_returnflag, l_linestatus;
"""

Q6 = """
SELECT SUM(l_extendedprice * l_discount)
FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24;
"""

# HELPERS
def run_cmd(cmd, cwd=None):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)

def backup_file(path: Path):
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    return backup

def restore_file(original: Path, backup: Path):
    shutil.copy2(backup, original)

def clean_build():
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        print("[INFO] Removed build directory")

def build_duckdb():
    run_cmd(
        [
            "cmake",
            "-B", "build",
            "-S", ".",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_EXTENSIONS=tpch",
        ],
        cwd=DUCKDB_ROOT,
    )
    run_cmd(["cmake", "--build", "build", "--config", "Release"], cwd=DUCKDB_ROOT)

    if not DUCKDB_BIN.exists():
        raise RuntimeError(f"DuckDB binary not found at {DUCKDB_BIN}")

    print("[INFO] DuckDB build completed")

def run_sql(sql, db_file): # removed default :memory:
    """Runs an SQL query against the specified duckdb file."""
    cmd = [str(DUCKDB_BIN), db_file, "-c", sql]
    start = time.perf_counter()
    subprocess.run(cmd, cwd=DUCKDB_ROOT, check=True)
    return time.perf_counter() - start

def prepare_tpch(db_path):
    """Creates a TPC-H database file."""
    print(f"[INFO] Preparing TPC-H database at {db_path}")
    sql = f"""
    LOAD tpch;
    CALL dbgen(sf={TPCH_SF});
    """
    # Ensure the directory exists for the db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    run_sql(sql, db_path)
    print("[INFO] TPC-H database prepared")

def benchmark(query, db_path):
    """Runs a query multiple times and returns average time."""
    times = []
    for i in range(REPEATS):
        t = run_sql(query, db_path)
        print(f"   Run {i + 1}: {t:.4f}s")
        times.append(t)
    avg_time = sum(times) / len(times)
    print(f"   Average: {avg_time:.4f}s")
    return times, avg_time

def save_csv(results):
    """Saves the benchmark results to a CSV file."""
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "vector_type",
            "q1_run1", "q1_run2", "q1_run3", "q1_avg",
            "q6_run1", "q6_run2", "q6_run3", "q6_avg",
            "db_size_bytes",
            "db_size_mb",
        ])
        for r in results:
            writer.writerow([
                r["vector_type"],
                *r["q1"], r["q1_avg"],
                *r["q6"], r["q6_avg"],
                r["db_size_bytes"],
                r["db_size_mb"],
            ])
    print(f"[DONE] Results saved to {RESULTS_CSV}")

def get_db_size_bytes(db_path):
    """Gets the size of the database file in bytes."""
    return os.path.getsize(db_path) if os.path.exists(db_path) else 0

def show_patch_context(path: Path, token="vector_type"):
    """Prints context around a specific token in a file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, start=1):
        if token in line or "VectorType::" in line:
            start = max(1, i - 3)
            end = min(len(lines), i + 3)
            print("\n[CONTEXT]")
            for j in range(start, end + 1):
                print(f"{j:5d}: {lines[j - 1]}")
            break
def patch_vector_type_in_source(
    target_vector_type: str
):
    """
    Safe targeted patch for DuckDB vector.cpp

    This patches ONLY the FIRST occurrence of:
        VectorType::FLAT_VECTOR

    inside vector.cpp.

    Avoids:
    - switch case duplication
    - VectorTypeToString corruption
    - enum duplication
    """

    if not VECTOR_CPP.exists():

        raise FileNotFoundError(
            f"Vector source file not found: "
            f"{VECTOR_CPP}"
        )

    text = VECTOR_CPP.read_text(
        encoding="utf-8"
    )

    lines = text.splitlines(
        keepends=True
    )

    patched = False

    for idx, line in enumerate(lines):

        # Skip switch/case statements
        if "case VectorType::" in line:
            continue

        # Skip string conversion helpers
        if "VectorTypeToString" in line:
            continue

        # Only patch real initialization lines
        if (
            "VectorType::FLAT_VECTOR" in line
            and "case" not in line
        ):

            new_line = line.replace(
                "VectorType::FLAT_VECTOR",
                f"VectorType::{target_vector_type}",
                1
            )

            lines[idx] = new_line

            patched = True

            print(
                f"[PATCH] "
                f"Forced "
                f"{target_vector_type}"
            )

            print(
                f"[LINE] "
                f"{new_line.strip()}"
            )

            break

    if not patched:

        # DEBUG HELP
        print("\n[DEBUG] Candidate lines:\n")

        for idx, line in enumerate(lines):

            if "VectorType::" in line:

                print(
                    f"{idx+1}: "
                    f"{line.strip()}"
                )

        raise RuntimeError(
            "Could not find safe "
            "VectorType::FLAT_VECTOR "
            "occurrence to patch."
        )

    VECTOR_CPP.write_text(
        "".join(lines),
        encoding="utf-8"
    )

    return VECTOR_CPP

# MAIN
def main():
    if not DUCKDB_ROOT.exists():
        raise FileNotFoundError(f"DuckDB root not found: {DUCKDB_ROOT}")

    if not VECTOR_CPP.exists():
        raise FileNotFoundError(f"Expected DuckDB source file missing: {VECTOR_CPP}")

    backup = backup_file(VECTOR_CPP)
    results = []

    try:
        for vector_type in TARGET_VECTOR_TYPES:
            print("\n" + "=" * 70)
            print(f"FORCING VECTOR TYPE = {vector_type}")
            print("=" * 70)

            # Restore clean source before each patch.
            restore_file(VECTOR_CPP, backup)

            # Patch source.
            patched_file = patch_vector_type_in_source(vector_type)
            show_patch_context(patched_file, token="VectorType::")

            # Rebuild.
            clean_build()
            build_duckdb()

            # Prepare database.
            db_name = f"tpch_sf{TPCH_SF}_{vector_type.lower()}.duckdb"
            db_path = str(DUCKDB_ROOT / db_name)

            if os.path.exists(db_path):
                os.remove(db_path)

            prepare_tpch(db_path) # This now correctly passes db_path to run_sql

            # Measure DB file size after data generation.
            db_size_bytes = get_db_size_bytes(db_path)
            db_size_mb = db_size_bytes / (1024 * 1024)
            print(f"[INFO] Database file size: {db_size_mb:.2f} MB")

            # Benchmark queries.
            print("\n[Q1]")
            q1_times, q1_avg = benchmark(Q1, db_path)

            print("\n[Q6]")
            q6_times, q6_avg = benchmark(Q6, db_path)

            results.append({
                "vector_type": vector_type,
                "q1": q1_times,
                "q1_avg": q1_avg,
                "q6": q6_times,
                "q6_avg": q6_avg,
                "db_size_bytes": db_size_bytes,
                "db_size_mb": db_size_mb,
            })

            save_csv(results)

        print("\n[COMPLETE] All vector types benchmarked.")

    finally:
        # Restore original source file.
        if backup.exists():
            restore_file(VECTOR_CPP, backup)
            backup.unlink(missing_ok=True)
        print("[INFO] Restored original DuckDB source files")

if __name__ == "__main__":
    main()