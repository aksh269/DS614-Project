# 🦆 Systems Engineering Project: DuckDB Vectorisation

**Team:** The Data Engineers
**Team Members:** Sanjana Nathani, Aksh Patel
**System Analyzed:** DuckDB (In-Process Analytical Database)
**Codebase:** [github.com/duckdb/thedataengineers](https://github.com/aksh269/DS614-Project)

---

## Executive Summary

Modern analytical workloads demand processing billions of rows with sub-second latency — a regime where classical row-oriented databases fundamentally collapse. DuckDB solves this by embedding a **columnar, vectorized execution engine** directly in-process, eliminating network overhead, maximizing CPU cache utilization, and exploiting SIMD hardware parallelism.

This report is not a documentation summary. It is a reverse-engineering dissection: we traced DuckDB's actual execution path through source code, identified the three architectural decisions that define its performance profile, empirically validated and broke those assumptions through live experiments, and mapped every finding to core systems concepts taught in class.

**Central thesis:** DuckDB achieves state-of-the-art analytical throughput by structuring data so precisely that the CPU hardware is never starved — and our experiments prove exactly where this holds, and where it collapses.

---

## 1. Execution Understanding: Tracing the Query Path End-to-End

We traced a standard analytical aggregation pipeline to demonstrate complete system understanding:

```sql
SELECT SUM(l_quantity) FROM lineitem GROUP BY l_returnflag;
```

DuckDB does not pass individual rows between operators. Instead, it pushes fixed-size batches of columnar data called `DataChunk`s through a pipeline DAG — the **vectorized push-based iteration model**.

### Step 1 — Query Submission and Parsing

**Component:** `ClientContext`
**File:** `src/main/client_context.cpp` → `ClientContext::Query()`

The raw SQL enters the system here. `ClientContext` passes it to the Parser, which generates a Logical AST. No execution occurs at this stage — DuckDB cleanly separates parsing, planning, and execution into distinct phases, enabling optimizer intervention at the logical level before any physical decisions are locked in.

### Step 2 — Physical Plan Generation

**Component:** `PhysicalPlanGenerator`
**File:** `src/execution/physical_plan_generator.cpp` → `PhysicalPlanGenerator::CreatePlan()`

The logical plan is lowered into a physical execution DAG. For our query, the planner instantiates `PhysicalHashAggregate` on top of `PhysicalTableScan`. This is the moment the DAG is crystallized. The alternative — a Volcano-style pull model as used in PostgreSQL — incurs a virtual function call overhead per row. DuckDB's push model eliminates this cost entirely by batching thousands of rows into a single operator invocation.

### Step 3 — Pipeline Task Execution

**Component:** `Executor` and `Pipeline`
**File:** `src/execution/executor.cpp` → `Executor::ExecuteTask()`

Execution is decomposed into `Pipeline` objects — linear chains of operators with no branching. The `Executor` schedules these pipelines as parallel tasks. Operators pull `DataChunk`s from their source and push results upward through the chain, keeping the CPU continuously fed with data.

### Step 4 — Data Fetching (The Vectorized Core)

**Component:** `DataChunk`
**File:** `src/common/types/data_chunk.cpp` → `DataChunk::Fetch()`

This is the architectural heart of DuckDB. Data is not read as tuples. It is read as contiguous column arrays wrapped in a `DataChunk` — a struct containing one `Vector` per column, each holding up to `STANDARD_VECTOR_SIZE` (default 2048) values. The memory layout is column-major: all values for `l_quantity` are contiguous before any values for `l_returnflag` appear. This is precisely what enables cache-line efficiency and SIMD acceleration.

### Step 5 — Operator Execution on the DataChunk

**Component:** `PhysicalHashAggregate`
**File:** `src/execution/operator/aggregate/physical_hash_aggregate.cpp` → `PhysicalHashAggregate::Execute()`

The aggregation operator receives an entire `DataChunk` and tight-loops over the raw C++ arrays to compute `SUM(l_quantity)`. This loop structure — operating on flat, contiguous memory — is what allows the compiler to emit AVX/SSE SIMD instructions, and what keeps the data inside the L1/L2 CPU cache for the entire duration of the computation.

---

## 2. Key Design Decisions

We identified three core design decisions that fundamentally define DuckDB's architecture and performance profile. For each, we point to the exact code, the problem it solves, the trade-off it introduces, and critically — what the alternative design would have been and why DuckDB rejected it.

### Design Decision 1 — Vectorized DataChunk Processing (vs. Tuple-at-a-Time)

**Implementation:** `src/common/types/data_chunk.cpp` and `src/common/types/vector.cpp`

**Problem solved:** Traditional tuple-at-a-time systems (the Volcano model) invoke a virtual `next()` call for every single row. On a 100 million row scan, that is 100 million function calls, 100 million branch predictions, and 100 million cache misses as the CPU jumps between operator stacks. The CPU spends more time on overhead than on actual computation.

**What DuckDB does instead:** It batches 2048 rows into a `DataChunk` and passes the entire batch to each operator in a single call. The operator then tight-loops over 2048 values — a pattern the CPU branch predictor handles perfectly, and one that keeps data resident in L1 cache throughout the computation.

**Alternative rejected:** The Volcano iterator model (PostgreSQL, SQLite). It is simpler to implement and excellent for OLTP workloads with small result sets where the overhead-per-row is negligible. DuckDB rejected it because analytical queries scan millions of rows, making per-row overhead the dominant cost.

**Trade-off introduced:** Memory pressure increases. Entire chunks must be allocated simultaneously. For workloads requiring heavy random access rather than sequential scans, cache thrashing can negate the batching advantage entirely — as our Experiment 5 demonstrates empirically.

![Vector vs Row Behavior](project/plots/exp_1.png)

### Design Decision 2 — Hardcoded Standard Vector Size (STANDARD_VECTOR_SIZE = 2048)

**Implementation:** `src/include/duckdb/common/vector_size.hpp` — `#define STANDARD_VECTOR_SIZE 2048`

**Problem solved:** Batch size is not a tunable parameter at runtime — it is compiled in. This guarantees that the working set for every operator (2048 integers × 8 bytes = 16 KB per column) fits comfortably within modern L1/L2 CPU caches (typically 32–256 KB), eliminating slow DRAM fetches during computation.

**Alternative rejected:** Dynamically sized batches (adaptive vectorization). While theoretically more flexible, dynamic sizing introduces branch logic to determine chunk boundaries, complicates memory allocation, and makes SIMD code generation significantly harder for the compiler. The static guarantee is a deliberate performance-correctness trade-off.

**Trade-off introduced:** A single static size cannot perfectly fit every hardware architecture or query type. A vector of 2048 64-bit floats occupies 16 KB — perfect for a modern L1 cache. But workloads with dozens of wide columns may overflow cache even at this size. Our Experiment 3 directly tests this boundary by recompiling DuckDB at sizes 64, 512, 2048, and 8192.

![Vector Sizing Performance Curves](project/plots/exp_3(1).png)
![Vector Sizing Bar Chart](project/plots/exp_3.png)

### Design Decision 3 — Exploiting SIMD Hardware Acceleration (with a Known Failure Mode)

**Implementation:** Arithmetic expression paths in `src/execution/expression_executor.cpp`

**Problem solved:** Modern CPUs support Single Instruction Multiple Data (SIMD) — executing the same arithmetic operation on 4, 8, or 16 values simultaneously in one clock cycle using AVX/SSE registers. By storing column data in flat, contiguous C++ arrays, DuckDB allows the compiler to automatically vectorize tight loops into SIMD instructions, multiplying effective arithmetic throughput without any explicit hardware code.

**Alternative rejected:** Scalar per-value computation. This is what SQLite does — it evaluates each expression one value at a time. For arithmetic-heavy analytical queries, this is 4–16x slower than SIMD-accelerated paths, as validated in our Experiment 2.

**Trade-off introduced — and this is critical:** SIMD pipelines are fragile. They strictly require contiguous, uniform memory layouts. The moment a query introduces string pattern matching (`LIKE`), branching `CASE` logic, or NULL-conditional paths, SIMD vectorization degrades or breaks entirely. DuckDB must fall back to scalar execution for these paths. **Our Experiment 1 exposes this directly: on the `STRING LIKE` benchmark, DuckDB's speedup over SQLite drops to its lowest point across all tested query types — and in some hardware configurations, SQLite's simpler execution model can be competitive on string-heavy workloads.** This is not in DuckDB's documentation. We found it empirically.

![SIMD Speedup Validations](project/plots/exp_2.png)

---

## 3. Concept Mapping

We explicitly mapped DuckDB's architecture to four core systems concepts from class:

| Course Concept | Implementation in DuckDB | Code Evidence |
|:---|:---|:---|
| **Execution: Directed Acyclic Graphs (DAG)** | DuckDB parses every SQL query into a strict DAG of `PhysicalOperator` nodes. Data flows from scan leaf nodes up to the query root through `Pipeline` states. There are no cycles — operators are stateless consumers of their child's output. | `src/execution/physical_plan_generator.cpp` |
| **Execution: Vectorization / Batching** | Rather than Volcano-style tuple iteration or MapReduce-style shuffle, DuckDB uses a vectorized push model, transmitting `DataChunk`s of 2048 rows between operators. This is the columnar equivalent of micro-batching. | `src/common/types/data_chunk.cpp` |
| **Storage: Cache Locality and Memory Placement** | DuckDB's columnar in-memory layout stores all values of a column contiguously before any values of the next column. This maximizes spatial locality for sequential scans, fitting operator working sets inside L1/L2 cache. `STANDARD_VECTOR_SIZE` is explicitly chosen to respect cache size boundaries. | `src/include/duckdb/common/vector_size.hpp` |
| **Hardware Acceleration: SIMD** | Expression evaluation loops operate on raw C++ arrays with no pointer aliasing, enabling the compiler to emit AVX/SSE vector instructions automatically. On arithmetic-heavy queries, this delivers 4–16x throughput over scalar execution — and degrades predictably on string and branching workloads. | `src/execution/expression_executor.cpp` |

---

## 4. Mandatory Experiment: Modifying the Core System (Vector Size Goldilocks Analysis)

This is the centerpiece experiment. We did not run DuckDB as a black box — we recompiled it from source with a patched internal constant to observe how the system behaves outside its designed operating parameters.

### Hypothesis

We hypothesized that `STANDARD_VECTOR_SIZE = 2048` exists in a deliberate Goldilocks zone. Too small and DuckDB degrades toward row-at-a-time behavior. Too large and the DataChunk working set spills out of CPU cache into slow DRAM, destroying the throughput advantage.

### Methodology

**Script:** `Exp_3_Vector_Size_Change.py`

1. Located the constant definition in `src/include/duckdb/common/vector_size.hpp` (`#define DEFAULT_STANDARD_VECTOR_SIZE 2048U`)
2. Programmatically patched the header file using Python regex substitution to target values `[64, 512, 2048, 8192]`
3. Triggered a full clean CMake rebuild for each size: `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_EXTENSIONS=tpch`
4. Benchmarked TPC-H Q1 (heavy aggregation) and Q6 (filtered arithmetic) across 3 runs each at every vector size
5. Restored the original header after all experiments concluded

### Results

| Vector Size | Q1 Avg Time | Q6 Avg Time | Behavior Explanation |
|:---:|:---:|:---:|:---|
| 64 | Slowest | Slowest | Operator functions called 32× more frequently — degrades toward row-at-a-time overhead |
| 512 | Moderate | Moderate | Below optimal — not fully utilizing cache capacity |
| 2048 | Fastest | Fastest | Goldilocks zone — working set fits L1/L2 cache |
| 8192 | Slow | Slow | DataChunk exceeds cache boundary — CPU stalls waiting on DRAM |

### Systems Explanation

Moving from 2048 down to 64 caused DuckDB to behave like a row-based system. Each operator function was invoked 32 times more frequently, saturating the CPU with function call and scheduling overhead rather than computation. The tight-loop SIMD advantage evaporates when loops only process 64 elements before returning control.

Moving from 2048 up to 8192 caused the `DataChunk` allocation (8192 values × 8 bytes × number of active columns) to structurally exceed L1/L2 cache capacity. The CPU's prefetcher could no longer anticipate the next cache line, and every vector access began triggering expensive DRAM page fetches. This is the exact cache thrashing scenario that `STANDARD_VECTOR_SIZE` was designed to prevent.

![Vector Sizing Performance Curves](project/plots/exp_3.png)
![Vector Sizing Bar Chart](project/plots/exp_3(1).png)

---

## 5. Failure Analysis: System Behavior Under Stress

### Failure Scenario 1 — What Happens When Data Size Increases Significantly?

**Script:** `Exp_4_Scale_Factor.py`

We incrementally expanded the TPC-H Scale Factor from SF=0.01 (millions of rows) to SF=1.0 (hundreds of millions of rows) and measured execution time for a constant aggregation query.

**Observation:** Execution time scaled perfectly linearly with data volume. There were no scaling cliffs, no sudden throughput collapses, no non-linear degradation.

**Why this happens — and why it is not obvious:** Many in-memory analytical systems attempt to hold and manipulate large singular data structures (hash tables, sort buffers) that grow with data size. When these structures exceed memory capacity, performance collapses non-linearly. DuckDB avoids this by processing data strictly batch-by-batch through its `Pipeline` DAG. Each `DataChunk` is independent — the system simply processes a higher count of identical-sized chunks through the same fixed pipeline. As long as disk and memory have sufficient total capacity, the pipeline throughput is constant per chunk, producing linear scaling.

![Scale Factor Linearity](project/plots/exp_4.png)

### Failure Scenario 2 — What Structural Assumptions Does DuckDB Rely On, and What Breaks Them?

**Script:** `Exp_5_Cache_Efficiency.py`

We stressed the execution engine by comparing three access patterns: sequential column scans, random-access joins (using `random() < 0.01` subquery sampling to force non-deterministic join order), and hash aggregation.

**Observation:** Throughput collapsed from **10,672 MB/s** on sequential scans to **151 MB/s** on random-access joins — a 70× degradation. DuckDB's vectorized engine, optimized for the sequential case, was effectively neutralized.

**Why this is a structural failure, not a bug:** Vectorization is architecturally inseparable from cache locality. Sequential column scans allow the CPU hardware prefetcher to predict and load the next cache line before it is needed. The processor is never idle. Random memory access via join permutations defeats prefetching entirely — each memory lookup lands on an unpredictable address, triggering a cache miss that stalls the CPU for 100–300 clock cycles waiting on DRAM.

The SIMD loops in `src/execution/expression_executor.cpp` are structurally designed for this sequential case. When data is scattered across memory, the loops are starved of input — the CPU executes the instruction, but the data is not in cache, so it waits. The engine does not crash. It simply stops being fast. This is the core architectural assumption of DuckDB: **data access is overwhelmingly sequential**. The moment that assumption breaks, the vectorization advantage evaporates.

**Additional finding from Experiment 1 (String LIKE):** The SIMD degradation is also visible in query type comparisons. On arithmetic aggregation, DuckDB achieves its maximum speedup over SQLite. On `STRING LIKE` pattern matching (`WHERE c_name LIKE 'Customer%000%'`), the speedup is dramatically lower. String operations do not fit the uniform, branch-free loop structure that SIMD requires. DuckDB must fall back to scalar string matching, partially closing the gap with SQLite's simpler execution model. This is a concrete, empirically observed failure mode of the SIMD design decision — not merely a theoretical trade-off.

![Cache Locality Collapse](project/plots/exp_5.png)

---

## 6. Summary: What We Learned from Reverse-Engineering DuckDB

| Finding | Evidence |
|:---|:---|
| DuckDB's performance is architectural, not accidental | Traced end-to-end from `ClientContext::Query()` to `PhysicalHashAggregate::Execute()` |
| `STANDARD_VECTOR_SIZE = 2048` is a carefully chosen hardware constant | Validated by recompiling at 64, 512, 2048, 8192 — performance degrades at both extremes |
| SIMD acceleration has a concrete failure mode: string and branching workloads | Observed in Experiment 1 — LIKE speedup is the lowest across all query types |
| Linear scaling is a deliberate consequence of batch-independent pipeline design | Validated across SF 0.01 to SF 1.0 with no non-linear degradation |
| Random access joins structurally break the vectorization assumption | Throughput collapsed from 10,672 MB/s to 151 MB/s in Experiment 5 |

> **Key Principle Validated:** DuckDB does not win through raw hardware power. It wins by making the CPU's job trivially predictable — fixed-size batches, contiguous memory, branch-free loops. Remove any one of those properties, and the advantage shrinks or disappears entirely. That is not a weakness. It is an explicit, well-reasoned design contract.