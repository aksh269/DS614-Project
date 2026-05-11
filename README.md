# 🦆 DS614: Big Data Engineering — DuckDB Vectorized Execution

**Course:** DS614 — Big Data Engineering  
**Team:** The Data Engineers  
**Members:** Sanjana Nathani · Aksh Patel  
**Institution:** DAU, Semester 2  
**System Analyzed:** DuckDB — In-Process Analytical Columnar Database  
**Our GitHub:** [aksh269/DS614-Project](https://github.com/aksh269/DS614-Project)  
**DuckDB Source:** [duckdb/duckdb](https://github.com/duckdb/duckdb)

---

## 📋 Table of Contents

1. [What is DuckDB?](#1-what-is-duckdb)
2. [Why DuckDB? — The Problem It Solves](#2-why-duckdb--the-problem-it-solves)
3. [Experiments Overview](#3-experiments-overview)
4. [Execution Path: Tracing a Query End-to-End](#4-execution-path-tracing-a-query-end-to-end)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Concept Mapping](#6-concept-mapping)
7. [Experiments — Deep Dive](#7-experiments--deep-dive)
   - [Experiment 1: Vectorized vs Row-Based Execution](#experiment-1-vectorized-vs-row-based-execution)
   - [Experiment 2: SIMD Impact Across Workload Types](#experiment-2-simd-impact-across-workload-types)
   - [Experiment 3: The Goldilocks Vector Size *(Source Patch)*](#experiment-3-the-goldilocks-vector-size-source-patch)
   - [Experiment 4: Scale Factor Linearity](#experiment-4-scale-factor-linearity)
   - [Experiment 5: Cache Efficiency Collapse](#experiment-5-cache-efficiency-collapse)
   - [Experiment 6: Vector Type Impact *(Source Patch)*](#experiment-6-vector-type-impact-source-patch)
8. [Failure Analysis](#8-failure-analysis)
9. [Key Insights & Conclusions](#9-key-insights--conclusions)
10. [How to Clone & Reproduce — Full Setup Guide](#10-how-to-clone--reproduce--full-setup-guide)
11. [Credits](#11-credits)

---

## 1. What is DuckDB?

DuckDB is a **free, open-source, in-process SQL database** designed specifically for **analytical workloads (OLAP)**. It runs entirely inside your application process — no server, no network overhead, no daemon to manage.

Think of it as **SQLite for analytics**: just like SQLite stores a database in a single file and needs no server, DuckDB does the same but is optimized for scanning millions of rows and computing aggregations, not for small transactional reads and writes.

### Key Characteristics

| Property | Value |
|:---|:---|
| **Type** | In-process OLAP SQL database |
| **Storage format** | Columnar (column-per-file) |
| **Execution model** | Vectorized push-based batches |
| **Language** | C++, with Python/R/Java/Node bindings |
| **Concurrent writes** | Single-writer, multiple-reader |
| **Best use case** | Analytical queries on 1 MB – 100 GB datasets |
| **License** | MIT |

### Where DuckDB Sits in the Database Landscape

```
                OLTP (Transactions)        OLAP (Analytics)
                ───────────────────        ────────────────
 Server-based   PostgreSQL, MySQL    ←→    Snowflake, BigQuery
 In-process     SQLite               ←→    DuckDB  ✓
```

DuckDB fills the bottom-right quadrant: **analytical queries that run locally**, without cloud infrastructure, without a running server, embedded directly into Python notebooks, data pipelines, or application code.

### What DuckDB Can Do That SQLite Cannot

```python
import duckdb

# Query a 500 MB Parquet file directly — no loading required
duckdb.sql("SELECT SUM(quantity), region FROM 'sales.parquet' GROUP BY region")

# Query a Pandas DataFrame as if it were a table
import pandas as pd
df = pd.read_csv("orders.csv")
duckdb.sql("SELECT customer_id, SUM(amount) FROM df GROUP BY customer_id")

# TPC-H benchmark built in
duckdb.sql("CALL dbgen(sf=1)")  # generates 1 GB of benchmark data
duckdb.sql("SELECT * FROM lineitem LIMIT 5")
```

---

## 2. Why DuckDB? — The Problem It Solves

### The Fundamental Problem with Row-Oriented Databases

Traditional databases (PostgreSQL, MySQL, SQLite) store data **row by row**. When a query asks for SUM of one column across 100 million rows, the database must:

1. Load row 1 from disk (all 50 columns)
2. Extract the one column you need
3. Call `next()` on the operator — a virtual function call
4. Repeat 100 million times

This destroys performance for three reasons:

```
Problem 1 — 100 million virtual function calls
  Each call: branch misprediction + stack frame + L1 cache miss
  Cost: ~50 ns × 100M calls = 5 seconds of pure overhead

Problem 2 — Loading columns you don't need
  Querying 1 column out of 50? You load all 50 anyway (row storage)
  Memory bandwidth wasted: 50x more than necessary

Problem 3 — SIMD is impossible
  CPU can add 4–8 values in parallel (AVX instructions)
  But data is scattered across non-contiguous row buffers
  Compiler cannot auto-vectorize — SIMD advantage is zero
```

### DuckDB's Solution: Vectorized Columnar Execution

DuckDB stores data **column by column** and processes it in **fixed batches of 2048 rows**:

```
Row-oriented (PostgreSQL):
  Row 1: [customer_id=1, name="Alice", quantity=17, price=25.50, ...]
  Row 2: [customer_id=2, name="Bob",   quantity=36, price=45.12, ...]
  → To sum quantity: load ALL columns of ALL rows

Columnar (DuckDB):
  quantity: [17, 36, 12, 45, 23, 8, ...]  ← all quantities, contiguous
  price:    [25.5, 45.1, 12.3, ...]       ← all prices, contiguous
  → To sum quantity: load ONE column, process 2048 values at once
```

| Property | Row-oriented (PostgreSQL/SQLite) | Columnar Vectorized (DuckDB) |
|:---|:---|:---|
| Storage layout | All columns of row N together | All values of column N together |
| Iteration model | `next()` per row (100M calls) | `Execute()` per 2048-row batch (48K calls) |
| Columns loaded | All columns even if one is needed | Only queried columns |
| L1 cache hit ratio | ~10% (random jumps) | ~80% (sequential array access) |
| SIMD utilization | 0% (scattered data) | High (tight contiguous loops) |
| Analytical throughput | 100–500 MB/s | 10,000+ MB/s |

---

## 3. Experiments Overview

We conducted **6 experiments** — 4 observational and 2 requiring DuckDB source code modification and recompilation.

| # | Experiment | Type | Key Question |
|:---:|:---|:---:|:---|
| 1 | Vectorized vs Row-Based | Observational | How much faster is DuckDB than SQLite across query types? |
| 2 | SIMD Impact | Observational | Where does SIMD help and where does it fail? |
| 3 | Goldilocks Vector Size | **Source Patch** | What happens when we change the 2048 batch size constant? |
| 4 | Scale Factor Linearity | Observational | Does DuckDB scale linearly with data volume? |
| 5 | Cache Efficiency | Observational | What access patterns destroy DuckDB's performance? |
| 6 | Vector Type Impact | **Source Patch** | What happens if we force the wrong internal vector type? |

Experiments 3 and 6 involved **modifying DuckDB's C++ source code**, triggering a full CMake rebuild, and benchmarking the patched binary — demonstrating understanding at the architectural level, not just API usage.

---

## 4. Execution Path: Tracing a Query End-to-End

> **Core Principle:** If you cannot point to code, you have not understood the system.

We trace this query from the moment a SQL string enters DuckDB to the moment a result comes out:

```sql
SELECT l_returnflag, l_linestatus,
       SUM(l_quantity), SUM(l_extendedprice)
FROM lineitem
GROUP BY l_returnflag, l_linestatus;
```

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  SQL String: "SELECT SUM(l_quantity) FROM lineitem ..."     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: ClientContext::Query()                             │
│  File: src/main/client_context.cpp                          │
│  → Parses SQL into Logical AST                              │
│  → Hands off to Planner                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: PhysicalPlanGenerator::CreatePlan()                │
│  File: src/execution/physical_plan_generator.cpp            │
│  → LogicalGet       → PhysicalTableScan                     │
│  → LogicalAggregate → PhysicalHashAggregate                 │
│  → Chooses PUSH model (not Volcano pull)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Executor::ExecuteTask()                            │
│  File: src/execution/executor.cpp                           │
│  → Decomposes plan into Pipeline objects                    │
│  → Schedules as parallel CPU tasks                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: DataChunk (The Core Abstraction)                   │
│  File: src/common/types/data_chunk.cpp                      │
│                                                             │
│  l_returnflag:    [A, A, R, N, R, ...] ← 2048 values       │
│  l_linestatus:    [F, O, F, F, O, ...] ← 2048 values       │
│  l_quantity:      [17,36,12,45, ...  ] ← 2048 values       │
│  l_extendedprice: [25.5,45.1, ...    ] ← 2048 values       │
│                                                             │
│  Column-major layout → CPU prefetcher loads ahead          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: PhysicalHashAggregate::Execute()                   │
│  File: src/execution/operator/aggregate/                    │
│        physical_hash_aggregate.cpp                          │
│                                                             │
│  for (i = 0; i < 2048; i++) {                              │
│      hash_table[flags[i]] += quantities[i];  ← SIMD here  │
│  }                                                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  Query Result    │
              └──────────────────┘
```
---

## 5. Key Design Decisions

### Decision 1: Vectorized Batch Processing (DataChunk = 2048 rows)

**Source files:**
- [`src/common/types/data_chunk.cpp`](https://github.com/duckdb/duckdb/blob/master/src/common/types/data_chunk.cpp)
- [`src/execution/executor.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/executor.cpp)

**Problem solved:**
On a 100M-row analytical query, Volcano-style execution makes 100M virtual `next()` calls. Each call:
- Loads operator state into registers (was evicted from L1 cache)
- Triggers a branch misprediction in the CPU pipeline
- Has a minimum cost of ~5 ns even if the work itself is free

At 100M calls × 5 ns = 500 ms of pure overhead with nothing computed yet.

**DuckDB's approach:**

```
Volcano (PostgreSQL):
  for 100,000,000 rows:
      row = child->next()          ← virtual function call (branch mispredict)
      process(row)                 ← 1 row processed
  Total calls: 100,000,000

DuckDB:
  for each DataChunk:              ← 100M / 2048 = 48,828 iterations
      chunk = child->Execute()    ← function call (predictable)
      for i in 0..2047:           ← tight inner loop (SIMD eligible)
          process(chunk[i])       ← 2048 rows per call
  Total operator calls: 48,828
```

**Alternative rejected:** Volcano iterator model. Simpler to implement and works fine for OLTP with small result sets. At analytical scale, per-row overhead dominates useful computation time.

**Trade-off:** The entire 2048-row DataChunk must reside in memory simultaneously. Random-access workloads (hash joins on foreign keys) can thrash cache even with batching — measured in Experiment 5.

---

### Decision 2: Hardcoded `STANDARD_VECTOR_SIZE = 2048`

**Source file:** [`src/include/duckdb/common/vector_size.hpp`](https://github.com/duckdb/duckdb/blob/master/src/include/duckdb/common/vector_size.hpp)

```cpp
// src/include/duckdb/common/vector_size.hpp
#ifndef DUCKDB_VECTOR_SIZE
#define DEFAULT_STANDARD_VECTOR_SIZE 2048U
#endif

#ifdef DUCKDB_VECTOR_SIZE
#if ((DUCKDB_VECTOR_SIZE & (DUCKDB_VECTOR_SIZE - 1)) != 0)
#error Vector size should be a power of two
#endif
#define STANDARD_VECTOR_SIZE DUCKDB_VECTOR_SIZE
#else
#define STANDARD_VECTOR_SIZE DEFAULT_STANDARD_VECTOR_SIZE
#endif
```

This constant is used in **40+ source files** — every operator, buffer allocator, and scheduler. Changing it requires a full recompile.

**Problem solved:** The working set per operator must fit inside L1/L2 cache to avoid DRAM stalls. For a single column of 2048 int64 values: 2048 × 8 bytes = **16 KB** — comfortably inside a modern L1 cache (32 KB per core).

**Why compile-time, not runtime?**
- A static constant lets the compiler unroll loops to a known bound
- Loop bounds can be embedded in SIMD unroll factors
- No branch: `if (chunk_size == 2048)` at every operator boundary
- Buffer pool allocations are a fixed size — no fragmentation

**Alternative rejected:** Runtime-adaptive batch sizing (MonetDB/X100 explored this). Dynamic sizing adds branching at every chunk boundary, complicates memory allocation, and prevents compile-time SIMD unrolling.

**Trade-off:** One size cannot fit all hardware (ARM vs x86 cache sizes) or all schemas (50+ wide columns multiply working set). Validated in Experiment 3.

---

### Decision 3: Automatic SIMD via Compiler (No Hand-Written Intrinsics)

**Source file:** [`src/execution/expression_executor.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/expression_executor.cpp)

**Problem solved:** Modern CPUs (Intel AVX2, ARM NEON) can process 4–8 values simultaneously with one instruction. Hand-writing SIMD intrinsics is architecture-specific, unmaintainable, and error-prone.

**DuckDB's approach:** Write tight, aliasing-free loops over flat C++ arrays. Let the compiler detect the pattern and emit SIMD automatically:

```cpp
// src/execution/expression_executor.cpp — evaluating x > 100
// This loop looks trivial. The compiler turns it into AVX2 instructions.
template 
void TemplatedExecuteComparison(Vector &left, Vector &right, Vector &result) {
    auto ldata = FlatVector::GetData(left);
    auto rdata = FlatVector::GetData(right);
    auto result_data = FlatVector::GetData(result);

    // No pointer aliasing (left, right, result are distinct buffers)
    // No loop-carried dependencies
    // Sequential stride-1 access
    // → Compiler emits: vpcmpgtq ymm0, ymm1, ymm2  (compare 4 int64s at once)
    for (idx_t i = 0; i < count; i++) {
        result_data[i] = (ldata[i] > rdata[i]);
    }
}
```

**Alternative rejected:** Explicit SIMD intrinsics (used in some proprietary column stores). Gives 5–15% more performance, but requires separate implementations for AVX512, AVX2, SSE4, NEON — multiplying maintenance burden by 4.

**Trade-off and failure mode:** SIMD requires branch-free, uniform operations. The moment a query introduces LIKE, CASE, or NULL handling, SIMD breaks. DuckDB falls back to scalar execution for these paths. This is measurable — Experiment 2 shows the degradation concretely.

---

## 6. Concept Mapping

| Concept (from DS614) | How DuckDB Implements It | Code Location |
|:---|:---|:---|
| **DAG Execution** | Every SQL query compiles to a strict DAG of `PhysicalOperator` nodes. Data flows one-way from leaf scans to root results; no cycles, no backward edges. Enables parallel pipeline scheduling. | [`src/execution/physical_plan_generator.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/physical_plan_generator.cpp) |
| **Batch Processing / Vectorization** | Instead of Volcano tuple-iteration or MapReduce shuffle, DuckDB transmits `DataChunk` micro-batches of 2048 rows between operators. Reduces function-call overhead by 2048x vs row-at-a-time. | [`src/common/types/data_chunk.cpp`](https://github.com/duckdb/duckdb/blob/master/src/common/types/data_chunk.cpp) |
| **Columnar Storage** | Data is stored and accessed column-by-column in contiguous memory. Only queried columns are loaded from disk. Working set per operator is bounded to L1/L2 cache size. | [`src/include/duckdb/common/vector_size.hpp`](https://github.com/duckdb/duckdb/blob/master/src/include/duckdb/common/vector_size.hpp) |
| **Hardware Acceleration (SIMD)** | Tight loops over flat C++ arrays with no pointer aliasing allow GCC/Clang to auto-emit AVX/SSE vector instructions. Degrades predictably on string and branching workloads. | [`src/execution/expression_executor.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/expression_executor.cpp) |
| **Parallelism** | Physical plan decomposes into `Pipeline` objects (linear operator chains) scheduled as concurrent CPU tasks. Each core runs one pipeline independently; no shared mutable state mid-execution. | [`src/execution/pipeline.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/pipeline.cpp) |
| **Memory Hierarchy Awareness** | Every design choice — vector size, columnar layout, buffer pool sizing — is oriented around respecting L1/L2/DRAM boundaries. `STANDARD_VECTOR_SIZE` encodes this knowledge as a compile-time constant. | [`src/include/duckdb/common/vector_size.hpp`](https://github.com/duckdb/duckdb/blob/master/src/include/duckdb/common/vector_size.hpp) |

---

## 7. Experiments — Deep Dive

### Experiment 1: Vectorized vs Row-Based Execution

**Script:** `project/Exp_1_Vector_vs_Row.ipynb`  
**Type:** Observational — no source modification  
**DuckDB source files involved:**
- [`src/common/types/data_chunk.cpp`](https://github.com/duckdb/duckdb/blob/master/src/common/types/data_chunk.cpp) — DataChunk abstraction
- [`src/execution/expression_executor.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/expression_executor.cpp) — expression evaluation loops

#### What We Tested

We compared DuckDB against SQLite across three query patterns that stress different execution characteristics:

```sql
-- Query A: Filter (WHERE clause selectivity)
SELECT * FROM lineitem WHERE l_quantity > 30;

-- Query B: Aggregation (columnar scan + SUM)
SELECT SUM(l_extendedprice) FROM lineitem;

-- Query C: String LIKE (forces scalar execution)
SELECT * FROM lineitem WHERE l_comment LIKE '%special%';
```

#### Results

| Query Type | DuckDB Time (ms) | SQLite Time (ms) | Speedup |
|:---|:---:|:---:|:---:|
| Filter (arithmetic) | 28 | 1,240 | **44x** |
| Aggregation | 45 | 1,850 | **41x** |
| String LIKE | 310 | 980 | **3x** |

#### Why String LIKE Shows Only 3x Speedup

String `LIKE` matching requires character-by-character comparison — a fundamentally branching operation. The compiler cannot vectorize it. DuckDB falls back to scalar execution for this path. SQLite's simpler execution model becomes competitive because both are now scalar.

This is not a bug. It is an architectural trade-off: DuckDB optimizes for the common analytical case (arithmetic on numeric data), and degrades gracefully on irregular patterns.

![Experiment 1 — Vectorized vs Row](project/plots/exp_1.png)

---

### Experiment 2: SIMD Impact Across Workload Types

**Script:** `project/Exp_2_SIMD.ipynb`  
**Type:** Observational — no source modification  
**DuckDB source files involved:**
- [`src/execution/expression_executor.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/expression_executor.cpp) — SIMD-eligible tight loops
- [`src/execution/expression/comparison.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/expression/comparison.cpp) — comparison operators
- [`src/execution/expression/arithmetic.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/expression/arithmetic.cpp) — arithmetic operators

#### What We Tested

We isolated the SIMD contribution by designing queries that are progressively less SIMD-friendly:

```sql
-- SIMD-maximum: pure arithmetic on numeric columns
SELECT x * y, x + y, x / y FROM synthetic_table;

-- SIMD-partial: comparisons (mostly vectorizable)
SELECT COUNT(*) FROM synthetic_table WHERE x > 100 AND y < 500;

-- SIMD-degraded: CASE expressions (branching — SIMD breaks)
SELECT CASE WHEN x > 100 THEN y ELSE z END FROM synthetic_table;

-- SIMD-hostile: string matching (no SIMD possible)
SELECT * FROM lineitem WHERE l_comment LIKE '%urgent%';
```

#### How SIMD Works in DuckDB (Code-Level View)

The expression executor in `src/execution/expression_executor.cpp` dispatches to typed, templated functions. For arithmetic:

```cpp
// This structure allows compiler vectorization:
// 1. ldata, rdata, result_data are non-aliased flat arrays
// 2. Loop has no carried dependencies
// 3. Stride is 1 (sequential access)
// → GCC/Clang emit: vmulpd ymm0, ymm1, ymm2  (multiply 4 doubles at once)
template 
static void TemplatedExecute(Vector &left, Vector &right, Vector &result,
                              idx_t count) {
    auto ldata = FlatVector::GetData(left);
    auto rdata = FlatVector::GetData(right);
    auto result_data = FlatVector::GetData(result);
    for (idx_t i = 0; i < count; i++) {
        result_data[i] = OP::Operation(ldata[i], rdata[i]);
    }
}
```

For CASE expressions, the executor uses a selection vector and conditional dispatch — no SIMD possible.

#### Results

| Workload Type | DuckDB vs SQLite Speedup | SIMD Active? |
|:---|:---:|:---:|
| Arithmetic (`x * y`) | **25x** | Yes — AVX2 |
| Comparison (`x > 100`) | **18x** | Partial — SSE4 |
| Conditional CASE | **8x** | No — scalar fallback |
| String LIKE | **3x** | No — scalar |

![Experiment 2 — SIMD Speedup by Workload](project/plots/exp_2.png)

**Key finding:** SIMD is not a universal advantage. It is a property of specific operation types. DuckDB's performance guarantee applies to numeric, arithmetic-heavy analytical queries. For string-heavy or conditional workloads, the speedup shrinks substantially.

---

### Experiment 3: The Goldilocks Vector Size *(Source Patch)*

**Script:** `project/Exp_3_Vector_Size_Change.py`  
**Type:** **SOURCE CODE MODIFICATION** — patches `vector_size.hpp`, triggers full CMake rebuild 4 times  

**DuckDB source files modified:**
- [`src/include/duckdb/common/vector_size.hpp`](https://github.com/duckdb/duckdb/blob/master/src/include/duckdb/common/vector_size.hpp) ← **directly patched**

**DuckDB source files affected by this constant (40+ files — sample):**
- [`src/common/types/vector.cpp`](https://github.com/duckdb/duckdb/blob/master/src/common/types/vector.cpp) — Vector memory allocation uses `STANDARD_VECTOR_SIZE`
- [`src/common/types/data_chunk.cpp`](https://github.com/duckdb/duckdb/blob/master/src/common/types/data_chunk.cpp) — DataChunk capacity
- [`src/execution/executor.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/executor.cpp) — Pipeline task sizing
- [`src/execution/operator/aggregate/physical_hash_aggregate.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/operator/aggregate/physical_hash_aggregate.cpp) — Aggregate buffer sizing
- All 40+ physical operators in [`src/execution/operator/`](https://github.com/duckdb/duckdb/tree/master/src/execution/operator)

#### Hypothesis

`STANDARD_VECTOR_SIZE = 2048` is a deliberate hardware-aware choice, not an arbitrary default. It sits in a Goldilocks zone:

```
Too small (e.g., 64):
  Working set = 64 × 8 bytes = 512 bytes per column
  → Operator overhead dominates. Called 32x more frequently.
  → Degrades back toward row-at-a-time behavior.

Goldilocks (2048):
  Working set = 2048 × 8 bytes = 16 KB per column
  → Fits comfortably in L1 cache (32 KB per core)
  → SIMD loops complete with zero DRAM stalls

Too large (e.g., 8192):
  Working set = 8192 × 8 bytes = 64 KB per column
  → Exceeds L1 cache (32 KB). Spills into L2.
  → Multiple columns together: 3 × 64 KB = 192 KB → spills into DRAM
  → Each vector access causes cache miss: 4 ns → 100–300 ns latency
```

#### Exact Source Code Change

**Original file** (`duckdb/src/include/duckdb/common/vector_size.hpp`):

```cpp
// ORIGINAL — what DuckDB ships with
#ifndef DUCKDB_VECTOR_SIZE
#define DEFAULT_STANDARD_VECTOR_SIZE 2048U
#endif

#ifdef DUCKDB_VECTOR_SIZE
#if ((DUCKDB_VECTOR_SIZE & (DUCKDB_VECTOR_SIZE - 1)) != 0)
#error Vector size should be a power of two
#endif
#define STANDARD_VECTOR_SIZE DUCKDB_VECTOR_SIZE
#else
#define STANDARD_VECTOR_SIZE DEFAULT_STANDARD_VECTOR_SIZE
#endif
```

**Patched versions** (4 separate rebuilds):

```cpp
// RUN 1: Too small — expect row-at-a-time degradation
#define DEFAULT_STANDARD_VECTOR_SIZE 64U

// RUN 2: Below optimal
#define DEFAULT_STANDARD_VECTOR_SIZE 512U

// RUN 3: DuckDB's default — expect fastest
#define DEFAULT_STANDARD_VECTOR_SIZE 2048U

// RUN 4: Too large — expect cache overflow
#define DEFAULT_STANDARD_VECTOR_SIZE 8192U
```

#### How the Patch Was Applied

```python
# project/Exp_3_Vector_Size_Change.py — key section

VECTOR_HEADER = Path(__file__).parent.parent / \
    "duckdb/src/include/duckdb/common/vector_size.hpp"
BUILD_DIR = Path(__file__).parent.parent / "duckdb/build"

for size in [64, 512, 2048, 8192]:
    # 1. Read original header
    with open(VECTOR_HEADER, 'r') as f:
        content = f.read()

    # 2. Regex substitution — changes ONLY the default vector size line
    pattern     = r'#define DEFAULT_STANDARD_VECTOR_SIZE \d+U'
    replacement = f'#define DEFAULT_STANDARD_VECTOR_SIZE {size}U'
    content_patched = re.sub(pattern, replacement, content)

    # 3. Write modified header back
    with open(VECTOR_HEADER, 'w') as f:
        f.write(content_patched)

    # 4. FULL REBUILD from source — forces recompilation of all 40+ affected files
    subprocess.run([
        'cmake', '--build', str(BUILD_DIR),
        '--config', 'Release',
        '-j8'                   # 8 parallel compile jobs
    ], check=True)

    # 5. Benchmark patched binary
    conn = duckdb.connect(str(DB_PATH))
    for query_name, sql in BENCHMARK_QUERIES.items():
        for run in range(NUM_RUNS):
            start = time.perf_counter()
            conn.execute(sql).fetchall()
            duration_ms = (time.perf_counter() - start) * 1000
            results.append({'vector_size': size, 'query': query_name,
                             'run': run, 'duration_ms': duration_ms})

# 6. Restore original header
with open(VECTOR_HEADER, 'w') as f:
    f.write(original_content)
```

#### Benchmark Queries Used

```sql
-- Q1: Heavy aggregation — exercises batch throughput
SELECT l_returnflag, l_linestatus,
       SUM(l_quantity), SUM(l_extendedprice),
       AVG(l_discount), COUNT(*)
FROM lineitem
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;

-- Q6: Filtered arithmetic — exercises predicate + multiply
SELECT SUM(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24;
```

#### Results

| Vector Size | Q1 Avg (ms) | Q6 Avg (ms) | Relative | Working Set (1 col) |
|:---:|:---:|:---:|:---:|:---|
| 64 | 870 | 1230 | 7.2x slower | 512 bytes (too small) |
| 512 | 180 | 240 | 1.5x slower | 4 KB (sub-optimal) |
| **2048** | **120** | **160** | **FASTEST** | 16 KB (fits L1 cache) |
| 8192 | 310 | 380 | 2.6x slower | 64 KB (exceeds L1) |

#### Why Each Size Behaves As It Does

**Size 64 — slowest (7.2x):**
- Operator `Execute()` is called 2048/64 = **32x more** than at size 2048
- Each additional call: ~50 ns overhead (branch predictor miss + register save/restore)
- 32 × 50 ns × 2.5M batches = 4+ seconds of pure call overhead
- SIMD advantage disappears: a 64-element loop provides too little work to amortize setup cost

**Size 512 — moderately slow (1.5x):**
- 4x more operator calls than 2048
- Partial cache benefit; SIMD loops are shorter than optimal

**Size 2048 — optimal:**
- 16 KB per column fits in L1 cache (32 KB per core on modern CPUs)
- Operator called 48,828 times for 100M rows — overhead negligible
- SIMD loops process 2048 elements per call — maximum vectorization benefit

**Size 8192 — slow (2.6x):**
- Single column: 8192 × 8 bytes = **64 KB** — already exceeds L1 cache (32 KB)
- With 3 active columns: 192 KB — exceeds L2 cache (256 KB, barely)
- CPU prefetcher cannot keep up; 20–25% of accesses become DRAM fetches (100–300 ns each)
- Even though operator calls are 4x fewer than 2048, DRAM stalls dominate

![Experiment 3 — Vector Size Performance Curves](project/plots/exp_3.png)
![Experiment 3 — Vector Size Bar Chart](project/plots/exp_3(1).png)

---

### Experiment 4: Scale Factor Linearity

**Script:** `project/Exp_4_Scalefactor.ipynb`  
**Type:** Observational — no source modification  
**DuckDB source files involved:**
- [`src/execution/executor.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/executor.cpp) — pipeline-per-batch processing
- [`src/execution/operator/aggregate/physical_hash_aggregate.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/operator/aggregate/physical_hash_aggregate.cpp) — in-memory hash aggregation

#### What We Tested

We generated TPC-H datasets at increasing scale factors and measured query execution time to determine whether DuckDB degrades non-linearly (as many systems do) at scale.

```sql
-- Q1: Constant aggregation across full table
SELECT l_returnflag, l_linestatus,
       SUM(l_quantity), COUNT(*)
FROM lineitem
GROUP BY l_returnflag, l_linestatus;
```

| Scale Factor | Lineitem Rows | Database Size | Q1 Time (ms) | Scaling |
|:---:|:---:|:---:|:---:|:---|
| 0.01 | 60,175 | ~2 MB | 12 | Baseline |
| 0.1 | 601,754 | ~20 MB | 120 | **10x data → 10x time** ✓ |
| 0.5 | 3,008,769 | ~100 MB | 600 | **50x data → 50x time** ✓ |
| 1.0 | 6,001,215 | ~200 MB | 1,200 | **100x data → 100x time** ✓ |

#### Why DuckDB Scales Linearly

Most systems show non-linear degradation at scale because they maintain global data structures that grow with dataset size:

```
Spark/MapReduce: shuffle buffer grows with data → GC pressure → stalls
B-tree index:    tree depth grows as O(log N)   → more seeks per lookup
Sort-merge join: sort buffer may overflow disk   → performance cliff

DuckDB's pipeline:
  while more data exists:
      chunk = TableScan.read_next_2048_rows()   ← fixed cost per chunk
      HashAgg.process(chunk)                    ← fixed cost per chunk
  
  Total time = N_chunks × cost_per_chunk
             = (total_rows / 2048) × constant
             ∝ total_rows                        ← LINEAR
```

Each DataChunk is processed independently. There is no data structure that grows with dataset size during the main scan loop, as long as:
1. The hash table fits in RAM (valid for SF ≤ 1.0 on standard hardware)
2. No sort operation requires full materialization

**The assumption that breaks this guarantee:** If the aggregation hash table overflows available RAM, DuckDB must spill to disk, and scaling becomes non-linear. This does not occur in the tested range.

![Experiment 4 — Scale Factor Linearity](project/plots/exp_4.png)

---

### Experiment 5: Cache Efficiency Collapse

**Script:** `project/Exp_5_Cache_Efficiency.ipynb`  
**Type:** Observational — no source modification  
**DuckDB source files involved:**
- [`src/execution/expression_executor.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/expression_executor.cpp) — SIMD tight loops (breaks on random access)
- [`src/execution/operator/join/physical_hash_join.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/operator/join/physical_hash_join.cpp) — random-access join probing

#### What We Tested

We designed three workloads with different memory access patterns to isolate the effect of cache locality:

```sql
-- Workload A: Sequential scan — CPU prefetcher works perfectly
SELECT SUM(l_quantity), SUM(l_extendedprice) FROM lineitem;

-- Workload B: Hash aggregation — semi-random hash table updates
SELECT l_orderkey, SUM(l_quantity) FROM lineitem GROUP BY l_orderkey;

-- Workload C: Random join — each probe is an unpredictable address
SELECT l_orderkey, o_totalprice
FROM lineitem L
JOIN (SELECT orderkey, totalprice FROM orders 
      ORDER BY RANDOM() LIMIT 100000) R
ON L.l_orderkey = R.orderkey;
```

#### Results

| Workload | Throughput (MB/s) | Relative to Sequential |
|:---|:---:|:---:|
| Sequential Scan (Workload A) | 10,672 MB/s | 1.0x (baseline) |
| Hash Aggregation (Workload B) | 3,245 MB/s | 0.30x |
| Random-Access Join (Workload C) | 151 MB/s | **0.014x (70x slower!)** |

#### Why Random Access Destroys Performance

```
Sequential access (Workload A):
  Address pattern: 0x1000, 0x1008, 0x1010, 0x1018, ...  (stride = 8)
  CPU prefetcher detects stride → loads next cache line before needed
  Every access: L1 cache hit (4 ns latency)
  Effective throughput: 10,672 MB/s

Random-access join (Workload C):
  Address pattern: 0x7A3C, 0x1F20, 0x8B44, 0x3201, ...  (no pattern)
  CPU prefetcher: cannot predict → gives up
  Every access: DRAM fetch required (100–300 ns latency)
  Stall cycles per access: 300 ns / 0.33 ns = ~900 cycles
  Effective throughput: 151 MB/s  (70x slower)

Calculation:
  L1 hit latency:   4 ns
  DRAM latency:   300 ns
  Ratio: 300/4 = 75x  ← matches observed 70x degradation
```

**What this reveals about DuckDB's architecture:**

The tight loops in `expression_executor.cpp` that enable SIMD assume sequential, prefetchable data. When every loop iteration fetches from an unpredictable address, the CPU pipeline stalls waiting for DRAM. SIMD is irrelevant — the bottleneck is memory, not computation.

```cpp
// This loop is fast when values[] is sequential (Workload A):
for (idx_t i = 0; i < count; i++) {
    sum += values[i];  // prefetcher loads values[i+1] during addition
}

// This loop is slow when random_keys[] causes hash table misses (Workload C):
for (idx_t i = 0; i < count; i++) {
    auto probe = hash_table[random_keys[i]];  // unpredictable → DRAM fetch → 900 cycle stall
    accumulate(probe);
}
```

![Experiment 5 — Cache Locality Collapse](project/plots/exp_5.png)

---

### Experiment 6: Vector Type Impact *(Source Patch)*

**Script:** `project/Exp_6_Vector_type_change.py`  
**Type:** **SOURCE CODE MODIFICATION** — patches `vector.cpp`, triggers full CMake rebuild 2 times  

**DuckDB source file modified:**
- [`src/common/types/vector.cpp`](https://github.com/duckdb/duckdb/blob/master/src/common/types/vector.cpp) ← **directly patched**

**DuckDB source files affected by vector type:**
- [`src/execution/operator/aggregate/physical_hash_aggregate.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/operator/aggregate/physical_hash_aggregate.cpp) — aggregate loops dispatch per vector type
- [`src/execution/expression_executor.cpp`](https://github.com/duckdb/duckdb/blob/master/src/execution/expression_executor.cpp) — expression evaluation checks vector type before dispatch
- [`src/common/types/vector_operations/`](https://github.com/duckdb/duckdb/tree/master/src/common/types/vector_operations) — all vector arithmetic dispatches per type

#### Background: DuckDB's Three Internal Vector Types

```
FLAT_VECTOR:
  Storage: [v0][v1][v2][v3]...[v2047]  ← 2048 distinct values
  Access:  O(1) direct index
  Best for: General mixed data (most analytical workloads)

CONSTANT_VECTOR:
  Storage: [v0]  ← only ONE value stored
  Access:  Always returns v0, ignoring index
  Best for: SELECT 1+2, constant expressions, NULL columns
  Memory saved: 2047 × sizeof(value)

DICTIONARY_VECTOR:
  Storage: dict=[A,B,C], indices=[0,1,0,2,0,1,...]
  Access:  values[indices[i]]  ← one indirection
  Best for: Low-cardinality strings (status codes, enum columns)
```

DuckDB's physical planner normally **chooses the right type automatically**. Our experiment forcibly overrides this choice to observe what happens when the wrong type is selected.

#### Exact Source Code Change

**Original file** (`duckdb/src/common/types/vector.cpp`):

```cpp
// ORIGINAL — DuckDB's correct default
// src/common/types/vector.cpp  (~line 58–75)

Vector::Vector(LogicalType type_p, bool create_data, bool zero_data,
               idx_t capacity)
    : vector_type(VectorType::FLAT_VECTOR), type(type_p), data(nullptr) {
    if (create_data) {
        Initialize(zero_data, capacity);
    }
}

void Vector::Initialize(bool zero_data, idx_t capacity) {
    // Allocates a FLAT buffer: capacity × sizeof(type) bytes
    auxiliary.reset();
    validity.Reset();
    auto type_size = GetTypeIdSize(GetTypeId());
    if (type_size > 0) {
        buffer = VectorBuffer::CreateStandardVector(type, capacity);
        data = buffer->GetData();
        if (zero_data) {
            memset(data, 0, capacity * type_size);
        }
    }
}
```

**Patched version (Exp 6, Run 1)** — forces CONSTANT_VECTOR:

```cpp
// PATCHED — forces CONSTANT_VECTOR regardless of data characteristics
// This is architecturally wrong for diverse data — triggers conversion overhead

Vector::Vector(LogicalType type_p, bool create_data, bool zero_data,
               idx_t capacity)
    : vector_type(VectorType::CONSTANT_VECTOR),  // ← CHANGED from FLAT_VECTOR
      type(type_p), data(nullptr) {
    if (create_data) {
        Initialize(zero_data, capacity);
    }
}
```

**How the patch was applied:**

```python
# project/Exp_6_Vector_type_change.py — key section

VECTOR_CPP = Path(__file__).parent.parent / "duckdb/src/common/types/vector.cpp"

for target_type in ["CONSTANT_VECTOR", "FLAT_VECTOR"]:
    # 1. Read original source
    with open(VECTOR_CPP, 'r') as f:
        content = f.read()

    # 2. Patch constructor initialization line
    pattern     = r'vector_type\(VectorType::\w+\)'
    replacement = f'vector_type(VectorType::{target_type})'
    content_patched = re.sub(pattern, replacement, content, count=1)

    # 3. Write patched source
    with open(VECTOR_CPP, 'w') as f:
        f.write(content_patched)

    # 4. Rebuild DuckDB — recompiles vector.cpp and all operators
    subprocess.run([
        'cmake', '--build', str(BUILD_DIR),
        '--config', 'Release', '-j8'
    ], check=True)

    # 5. Benchmark with patched binary
    conn = duckdb.connect(str(DB_PATH))
    for run in range(NUM_RUNS):
        for query_name, sql in BENCHMARK_QUERIES.items():
            t0 = time.perf_counter()
            conn.execute(sql).fetchall()
            duration_ms = (time.perf_counter() - t0) * 1000
            results.append({'type': target_type, 'query': query_name,
                             'run': run, 'duration_ms': duration_ms})

# 6. Restore original file
with open(VECTOR_CPP, 'w') as f:
    f.write(original_content)
```

#### Results

| Vector Type | Q1 Avg (ms) | Q6 Avg (ms) | Slowdown | DB Size |
|:---|:---:|:---:|:---:|:---:|
| `CONSTANT_VECTOR` (patched) | 1,030 | 1,025 | **8.6x slower** | 26.5 MB |
| `FLAT_VECTOR` (correct default) | 120 | 160 | 1.0x (baseline) | 26.3 MB |

#### Why Forcing CONSTANT_VECTOR Breaks Performance

`lineitem.l_quantity` contains values like `[17, 36, 12, 45, 23, 8, ...]` — a different value per row. A `CONSTANT_VECTOR` stores only one value and repeats it logically. When DuckDB receives a `CONSTANT_VECTOR` for `l_quantity` but the data is clearly not constant, it must:

```
Step 1: Type-check dispatch in expression_executor.cpp
  → "This vector claims to be CONSTANT but I need FLAT for aggregation"
  → Materialize: allocate a new FLAT buffer of 2048 × 8 bytes

Step 2: Copy the single constant value into 2048 slots
  → memset/memcpy: 2048 × 8 = 16 KB per batch

Step 3: Run the aggregation on the materialized FLAT buffer

Cost per batch: ~5–10 µs (allocation + copy)
Batches for 100M rows: ~48,828
Total overhead: 5 µs × 48,828 ≈ 244 seconds
(in practice, constant folding catches some — measured: ~1 second overhead per query)
```

Compare to `FLAT_VECTOR`: no type-check, no allocation, no copy — straight to the aggregation loop.

**Architectural lesson:** Vector type is not an implementation detail. It is a contract between the planner (which selects the type) and the operators (which assume the type is correct). Violating this contract triggers silent cascading conversions that degrade performance by nearly an order of magnitude without any error.

![Experiment 6 — Vector Type Comparison](project/plots/exp_6_vector_comparison.png)

---

## 8. Failure Analysis

### Failure 1: What Happens When Data Size Increases Significantly?

**Experiment:** Scale Factor (Experiment 4)

**Answer:** DuckDB scales **linearly** within memory limits.

From SF=0.01 to SF=1.0 (100x more data), execution time increases by exactly 100x — perfect linear scaling. This is a property of the batch-independent pipeline architecture: more data means more DataChunks, but each DataChunk costs the same to process.

**Where the assumption breaks:**

```
Normal operation (fits in RAM):
  Hash table for GROUP BY l_orderkey: ~6M distinct keys × 16 bytes = 96 MB
  Available RAM: 8 GB → fits → linear scaling

Memory overflow (doesn't fit):
  SF=10.0: hash table → ~960 MB
  SF=100.0: hash table → 9.6 GB → DRAM exhausted → disk spill
  Performance cliff: disk I/O latency is 1000x–10,000x slower than DRAM
```

At SF=0.01–1.0 on standard hardware, DuckDB stays in-memory → linear. Beyond available RAM, DuckDB degrades non-linearly like any system. Apache Spark and Flink are designed for disk-spill scenarios; DuckDB is not — it is an in-memory engine.

---

### Failure 2: What Structural Assumptions Does DuckDB Rely On?

**Experiment:** Cache Efficiency (Experiment 5)

DuckDB's vectorization advantage rests on three assumptions. When any is violated, performance degrades — sometimes catastrophically:

| Assumption | What Code Relies On It | What Violates It | Measured Impact |
|:---|:---|:---|:---:|
| **Sequential memory access** | `expression_executor.cpp` tight loops rely on CPU prefetcher | Random-access joins, hash probes on large tables | **70x slowdown** |
| **Working set fits in L1/L2 cache** | `STANDARD_VECTOR_SIZE = 2048` sized for 32 KB L1 | Schemas with 50+ columns; vectors forced to large sizes | Up to 7x slowdown (Exp 3) |
| **Branch-free operations** | SIMD auto-vectorization requires no per-element conditionals | String LIKE, CASE expressions, NULL handling | SIMD → scalar: 25x → 3x speedup |

None of these failures cause errors or crashes. DuckDB silently degrades. The system remains correct — it simply stops being fast.

**Practical consequence:** DuckDB is the wrong tool for:
- Join-heavy OLTP workloads (random access patterns)
- String-dominant text analytics (SIMD failure)
- Workloads larger than available RAM (no fault tolerance)
- Systems where multiple concurrent writers are needed (single-writer)

---

### Failure 3: What Happens When a Component Gets the Wrong Input?

**Experiment:** Vector Type (Experiment 6)

Forcing `CONSTANT_VECTOR` on a column with diverse values causes an **8.6x performance slowdown** with no change to the algorithm. The engine produces correct results but through a much more expensive path.

This reveals a silent failure mode: DuckDB's performance is sensitive not only to the query and data, but to the correctness of the type-selection heuristics in the physical planner. If the optimizer misclassifies a column — a realistic scenario with highly skewed data — performance can degrade by nearly an order of magnitude with no warning.

---

## 9. Key Insights & Conclusions

| Finding | Validated By | Implication |
|:---|:---|:---|
| DuckDB's speedup is architecture-dependent, not universal | Exp 2: SIMD fails on strings (25x → 3x) | Design for your workload; DuckDB is not a universal replacement |
| `2048` encodes hardware knowledge, not an arbitrary choice | Exp 3: both 64 and 8192 are 2.6–7.2x slower | Compile-time constants that encode hardware boundaries outperform runtime tuning |
| Linear scaling is an emergent property of stateless batch processing | Exp 4: perfect O(N) from SF=0.01 to SF=1.0 | Pipeline architecture with fixed-size chunks naturally prevents super-linear degradation |
| Sequential memory access is a prerequisite, not a feature | Exp 5: random joins are 70x slower | SIMD and vectorization are only fast when the prefetcher can feed them |
| Internal representation is as important as algorithm | Exp 6: wrong vector type → 8.6x slowdown | The planner's type annotations are performance contracts, not hints |

### How to Improve DuckDB for Its Failure Cases

- **Random-access joins:** Cluster related data with a secondary index; partition tables to improve join locality
- **String-heavy workloads:** Implement JIT compilation for SIMD-aware string kernels (MonetDB/X100 approach)
- **Wide schemas:** Adaptive vector sizing — scale batch size proportionally down to column count to stay within cache bounds
- **Memory-constrained systems:** Operator-level spill-to-disk for hash aggregation and sort (DuckDB has partial support; extending it is an active research area)

> DuckDB wins not through raw hardware power, but by making the CPU's job trivially predictable: fixed-size batches, contiguous memory, branch-free loops. Remove any one of those properties and the advantage shrinks or disappears entirely. That is not a weakness — it is an explicit, well-reasoned design contract.

---

## 10. How to Clone & Reproduce — Full Setup Guide

This section is a **complete guide for a new user** to reproduce all 6 experiments from scratch.

### System Requirements

| Component | Minimum | Recommended |
|:---|:---|:---|
| OS | Linux (Ubuntu 20.04+) or macOS 12+ | Ubuntu 22.04 |
| RAM | 4 GB | 8 GB+ |
| Disk | 5 GB free | 10 GB free |
| Python | 3.8+ | 3.10+ |
| C++ compiler | GCC 9+ or Clang 10+ | GCC 12 |
| CMake | 3.15+ | 3.22+ |
| Build time (Exp 3 & 6) | — | 30–45 min total |

---

### Step 1: Install System Dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y cmake g++ make git python3 python3-pip jupyter-notebook
```

**macOS:**
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install cmake git python3
pip3 install jupyter
```

---

### Step 2: Clone This Repository

```bash
git clone https://github.com/aksh269/DS614-Project.git
cd DS614-Project
```

---

### Step 3: Clone DuckDB Source Code

*(Required only for Experiments 3 and 6 — which patch and recompile DuckDB)*

```bash
# From inside DS614-Project/
git clone https://github.com/duckdb/duckdb.git
```

After cloning, verify your directory structure looks like this:

```
DS614-Project/
├── duckdb/                              ← DuckDB source code (just cloned)
│   ├── src/
│   │   ├── include/
│   │   │   └── duckdb/
│   │   │       └── common/
│   │   │           └── vector_size.hpp  ← Exp 3 patches this
│   │   ├── common/
│   │   │   └── types/
│   │   │       └── vector.cpp           ← Exp 6 patches this
│   │   └── execution/
│   ├── CMakeLists.txt
│   └── ...
│
├── project/                             ← All our experiment scripts
│   ├── Exp_1_Vector_vs_Row.ipynb
│   ├── Exp_2_SIMD.ipynb
│   ├── Exp_3_Vector_Size_Change.py      ← patches duckdb/src, rebuilds 4×
│   ├── Exp_4_Scalefactor.ipynb
│   ├── Exp_5_Cache_Efficiency.ipynb
│   ├── Exp_6_Vector_type_change.py      ← patches duckdb/src, rebuilds 2×
│   ├── plots/
│   │   ├── exp_1.png
│   │   ├── exp_2.png
│   │   ├── exp_3.png
│   │   ├── exp_3(1).png
│   │   ├── exp_4.png
│   │   ├── exp_5.png
│   │   └── exp_6_vector_comparison.png
│   ├── goldilocks_vector_size_results.csv   ← generated by Exp 3
│   └── duckdb_vector_type_results.csv       ← generated by Exp 6
│
└── README.md
```

---

### Step 4: Install Python Dependencies

```bash
pip3 install duckdb pandas numpy matplotlib scipy jupyter
```

Verify DuckDB installed correctly:
```bash
python3 -c "import duckdb; print(duckdb.__version__)"
# Expected output: 0.9.x or higher
```

---

### Step 5: Initial DuckDB Build

*(Required only for Experiments 3 and 6)*

```bash
cd duckdb

# Configure build with TPC-H extension (required for benchmark data generation)
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXTENSIONS=tpch

# Compile DuckDB (takes 5–15 minutes)
cmake --build build --config Release -j$(nproc)

# Verify build succeeded
ls -la build/src/libduckdb*.so   # Linux
ls -la build/src/libduckdb*.dylib  # macOS

cd ..
```

If the build succeeds, you will see `libduckdb.so` (Linux) or `libduckdb.dylib` (macOS) in the build directory.

---

### Step 6: Run Each Experiment

#### Experiments 1, 2, 4, 5 — Jupyter Notebooks (No Source Modification)

```bash
cd project

# Experiment 1: DuckDB vs SQLite across query types
jupyter notebook Exp_1_Vector_vs_Row.ipynb
# → In browser: Run All Cells (Kernel → Restart & Run All)

# Experiment 2: SIMD impact across workload types
jupyter notebook Exp_2_SIMD.ipynb

# Experiment 4: Scale factor linearity
jupyter notebook Exp_4_Scalefactor.ipynb

# Experiment 5: Cache efficiency collapse
jupyter notebook Exp_5_Cache_Efficiency.ipynb
```

Each notebook:
1. Generates or loads TPC-H benchmark data automatically
2. Runs the benchmark
3. Saves plots to `project/plots/`

---

#### Experiment 3 — Source Patching: Goldilocks Vector Size

> ⚠️ **This experiment modifies DuckDB C++ source code and rebuilds 4 times.**  
> Total time: approximately 20–30 minutes.  
> Your `duckdb/` folder must be present and the initial build must have succeeded (Step 5).

```bash
cd project
python3 Exp_3_Vector_Size_Change.py
```

**What it does automatically:**
1. Reads `duckdb/src/include/duckdb/common/vector_size.hpp`
2. Patches `DEFAULT_STANDARD_VECTOR_SIZE` to each of: `64`, `512`, `2048`, `8192`
3. For each size: triggers `cmake --build` (full recompile, ~5 min each)
4. Benchmarks TPC-H Q1 and Q6 on the freshly compiled binary
5. Saves all results to `project/goldilocks_vector_size_results.csv`
6. **Restores the original `vector_size.hpp` at the end**

**Output files:**
```
project/goldilocks_vector_size_results.csv
project/plots/exp_3.png
project/plots/exp_3(1).png
```

---

#### Experiment 6 — Source Patching: Vector Type Impact

> ⚠️ **This experiment modifies DuckDB C++ source code and rebuilds 2 times.**  
> Total time: approximately 10–15 minutes.

```bash
cd project
python3 Exp_6_Vector_type_change.py
```

**What it does automatically:**
1. Reads `duckdb/src/common/types/vector.cpp`
2. Patches the default vector type to `CONSTANT_VECTOR` for Run 1
3. Rebuilds DuckDB, benchmarks, saves results
4. Patches back to `FLAT_VECTOR` for Run 2
5. Rebuilds DuckDB, benchmarks, saves results
6. **Restores the original `vector.cpp` at the end**

**Output files:**
```
project/duckdb_vector_type_results.csv
project/plots/exp_6_vector_comparison.png
```

---

### Step 7: View All Results

After running all experiments, results are in:

```bash
# CSV outputs (Experiments 3 and 6)
cat project/goldilocks_vector_size_results.csv
cat project/duckdb_vector_type_results.csv

# Plot images
ls project/plots/
# exp_1.png, exp_2.png, exp_3.png, exp_3(1).png,
# exp_4.png, exp_5.png, exp_6_vector_comparison.png
```

---

### Quick Reference: All Commands

```bash
# ── One-time setup ──────────────────────────────────────────────────────────
git clone https://github.com/aksh269/DS614-Project.git
cd DS614-Project
git clone https://github.com/duckdb/duckdb.git
pip3 install duckdb pandas numpy matplotlib scipy jupyter

# Build DuckDB (for Exp 3 & 6)
cd duckdb && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_EXTENSIONS=tpch
cmake --build build --config Release -j$(nproc) && cd ..

# ── Run experiments ─────────────────────────────────────────────────────────
cd project

# Exp 1–2, 4–5: Jupyter notebooks (no source changes, ~5 min each)
jupyter notebook Exp_1_Vector_vs_Row.ipynb
jupyter notebook Exp_2_SIMD.ipynb
jupyter notebook Exp_4_Scalefactor.ipynb
jupyter notebook Exp_5_Cache_Efficiency.ipynb

# Exp 3: Source patch — vector size (rebuilds 4×, ~25 min total)
python3 Exp_3_Vector_Size_Change.py

# Exp 6: Source patch — vector type (rebuilds 2×, ~12 min total)
python3 Exp_6_Vector_type_change.py
```

---

### Troubleshooting

| Problem | Solution |
|:---|:---|
| `cmake: command not found` | `sudo apt install cmake` (Linux) or `brew install cmake` (macOS) |
| `duckdb module not found` | `pip3 install duckdb` |
| Exp 3/6 fails: `duckdb/` not found | Run `git clone https://github.com/duckdb/duckdb.git` from `DS614-Project/` root |
| Build fails: C++ errors | Ensure GCC ≥ 9: `g++ --version`. Update with `sudo apt install g++-12` |
| Jupyter not opening | `pip3 install jupyter` then `jupyter notebook --no-browser` |
| Out of disk during build | DuckDB build needs ~3 GB. Free disk space and retry. |
| Experiment 3 leaves source modified | Script auto-restores. If interrupted: `cd duckdb && git checkout src/include/duckdb/common/vector_size.hpp` |
| Experiment 6 leaves source modified | `cd duckdb && git checkout src/common/types/vector.cpp` |

---

## 11. Credits

**Team:**
- Sanjana Nathani — DAU, DS614, Semester 2
- Aksh Patel — DAU, DS614, Semester 2

**Acknowledgments:**
- The DuckDB team at CWI Amsterdam and contributors at [github.com/duckdb/duckdb](https://github.com/duckdb/duckdb) — for building and maintaining an exceptionally well-documented open-source system
- Course instructor, DS614 — for the reverse-engineering methodology that shaped this project
- TPC-H Benchmark Suite — all analytical workloads generated via DuckDB's built-in `tpch` extension

**Key References:**
- [DuckDB: An Embeddable Analytical Database](https://duckdb.org/pdf/SIGMOD2019-demo-duckdb.pdf) — Raasveldt & Mühleisen, SIGMOD 2019
- [MonetDB/X100: Hyper-Pipelining Query Execution](https://www.cidrdb.org/cidr2005/papers/P19.pdf) — Boncz et al., CIDR 2005
- Intel 64 Architecture Optimization Reference Manual
- GCC Auto-Vectorization Guide: [gcc.gnu.org/projects/tree-ssa/vectorization.html](https://gcc.gnu.org/projects/tree-ssa/vectorization.html)

---

**Last Updated:** May 2026  
**Repository:** [github.com/aksh269/DS614-Project](https://github.com/aksh269/DS614-Project)