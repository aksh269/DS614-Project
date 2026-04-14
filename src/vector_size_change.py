import os
import re
import csv
import time
import shutil
import subprocess
from pathlib import Path

# =========================
# CONFIG
# =========================
DUCKDB_ROOT = Path(r"C:\MSDS SEM2\BDE\aksh\DS614-Project\Duck_DB  clone\duckdb")
VECTOR_HEADER = DUCKDB_ROOT / "src/include/duckdb/common/vector_size.hpp"
BUILD_DIR = DUCKDB_ROOT / "build"
DUCKDB_BIN = BUILD_DIR / "Release" / "duckdb.exe" # Windows CMake output is typically here

print("DUCKDB_ROOT:", DUCKDB_ROOT)
print("VECTOR_HEADER:", VECTOR_HEADER)
print("Exists:", VECTOR_HEADER.exists())

VECTOR_SIZES = [64, 512, 2048, 8192]
REPEATS = 3
RESULTS_CSV = Path(r"C:\MSDS SEM2\BDE\aksh\DS614-Project\src\goldilocks_vector_size_results.csv")
TPCH_SF = 0.1   # start small

# =========================
# QUERIES
# =========================
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

# =========================
# HELPERS
# =========================

def run_cmd(cmd, cwd=None):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)

def backup_file(path):
    backup = path.with_suffix(".bak")
    shutil.copy(path, backup)
    return backup

def restore_file(original, backup):
    shutil.copy(backup, original)
    backup.unlink(missing_ok=True)

def patch_vector_size(header_file, new_size):
    text = header_file.read_text()
    
    # We replace DEFAULT_STANDARD_VECTOR_SIZE 2048U
    pattern = r'(#define\s+DEFAULT_STANDARD_VECTOR_SIZE\s+)\d+U?'
    
    if not re.search(pattern, text):
        raise RuntimeError("DEFAULT_STANDARD_VECTOR_SIZE not found in header file.")
        
    new_text = re.sub(pattern, rf'\g<1>{new_size}U', text)
    
    header_file.write_text(new_text)
    print(f"[INFO] Set DEFAULT_STANDARD_VECTOR_SIZE = {new_size}U")

def clean_build():
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        print("[INFO] Cleaned build directory")

def build_duckdb():
    run_cmd(["cmake", "-B", "build", "-S", ".", "-DCMAKE_BUILD_TYPE=Release"], cwd=DUCKDB_ROOT)
    run_cmd(["cmake", "--build", "build", "--config", "Release"], cwd=DUCKDB_ROOT)

    if not DUCKDB_BIN.exists():
        raise RuntimeError(f"duckdb.exe not found after build at {DUCKDB_BIN}")

    print("[INFO] Build successful")

def run_sql(sql, db_file=":memory:"):
    cmd = [
        str(DUCKDB_BIN),
        db_file,
        "-c",
        sql
    ]

    start = time.perf_counter()
    subprocess.run(cmd, cwd=DUCKDB_ROOT, check=True)
    return time.perf_counter() - start

def prepare_tpch(db_path):
    sql = f"""
    INSTALL tpch;
    LOAD tpch;
    CALL dbgen(sf={TPCH_SF});
    """
    run_sql(sql, db_path)

def benchmark(query, db_path):
    times = []
    for i in range(REPEATS):
        t = run_sql(query, db_path)
        print(f"   Run {i+1}: {t:.4f}s")
        times.append(t)
    return times, sum(times) / len(times)

# =========================
# MAIN
# =========================

def save_csv(results):
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "vector_size",
            "q1_run1", "q1_run2", "q1_run3", "q1_avg",
            "q6_run1", "q6_run2", "q6_run3", "q6_avg"
        ])
        for r in results:
            writer.writerow([
                r["vector_size"],
                *r["q1"], r["q1_avg"],
                *r["q6"], r["q6_avg"]
            ])
    print(f"\n[DONE] Results saved to {RESULTS_CSV}")

def main():
    if not VECTOR_HEADER.exists():
        raise FileNotFoundError(f"vector_size.hpp not found: {VECTOR_HEADER}")

    backup = backup_file(VECTOR_HEADER)
    results = []

    try:
        for size in VECTOR_SIZES:
            print("\n" + "="*60)
            print(f"VECTOR SIZE = {size}")
            print("="*60)

            # 1. Patch
            patch_vector_size(VECTOR_HEADER, size)

            # 2. Rebuild
            clean_build()
            build_duckdb()

            # 3. Prepare DB
            db_name = f"tpch_sf{TPCH_SF}_vec{size}.duckdb"
            db_path = str(DUCKDB_ROOT / db_name)

            if os.path.exists(db_path):
                os.remove(db_path)

            prepare_tpch(db_path)

            # 4. Benchmark
            print("\n[Q1]")
            q1_times, q1_avg = benchmark(Q1, db_path)

            print("\n[Q6]")
            q6_times, q6_avg = benchmark(Q6, db_path)

            results.append({
                "vector_size": size,
                "q1": q1_times,
                "q1_avg": q1_avg,
                "q6": q6_times,
                "q6_avg": q6_avg
            })

            # Save partially to avoid losing progress
            save_csv(results)

    finally:
        restore_file(VECTOR_HEADER, backup)
        print("[INFO] Restored original vector_size.hpp")

if __name__ == "__main__":
    main()
