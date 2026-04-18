# 🦆 Systems Engineering Project: DuckDB Vectorisation

**Team:** The Data Engineers 
**Team Members:** Sanjana Nathani, Aksh Patel
**System Analyzed:** DuckDB (In-Process Analytical Database)

---

## Executive Summary
DuckDB solves the problem of high-performance analytical (OLAP) processing in an embedded, zero-management database. Unlike classical row-oriented database systems (e.g., SQLite, PostgreSQL) that suffer from massive function-call overhead during analytical queries, DuckDB utilizes a **columnar, vectorized execution engine** to maximize CPU efficiency, keep data within CPU cache lines, and exploit data-parallel SIMD hardware instructions. 

This report reverse-engineers the system to trace its execution path, analyzes critical design decisions impacting performance, and empirically validates the system's behavior through custom experiments.

---

## 1. Execution Understanding: Tracing the Query Path

To demonstrate complete system understanding, we traced a standard analytical aggregation pipeline: `SELECT SUM(l_quantity) FROM lineitem GROUP BY l_returnflag;`.

DuckDB does not pass individual rows between operators. Instead, it pushes fixed-size batches of columnar data called `DataChunks`. This forms a vectorized push-based iteration model.

### Complete Execution Trace (End-to-End)

1. **Query Submission & Parsing** 
   * **Component:** `ClientContext`
   * **Source:** `src/main/client_context.cpp` -> `ClientContext::Query()`
   * **Action:** The query enters the system and is passed to the Parser to generate a Logical AST.
2. **Physical Plan Generation** 
   * **Component:** `PhysicalPlanGenerator`
   * **Source:** `src/execution/physical_plan_generator.cpp` -> `PhysicalPlanGenerator::CreatePlan()`
   * **Action:** The logical plan is converted into a physical execution DAG. For our query, it instantiates a `PhysicalHashAggregate` operator on top of a `PhysicalTableScan`.
3. **Pipeline Task Execution (Push Model)**
   * **Component:** `Executor` & `Pipeline`
   * **Source:** `src/execution/executor.cpp` -> `Executor::ExecuteTask()`
   * **Action:** Execution is broken into tasks. Operators pull chunks from their children and push results upwards.
4. **Data Fetching (The Vectorized Core)**
   * **Component:** `DataChunk`
   * **Source:** `src/common/types/data_chunk.cpp` -> `DataChunk::Fetch()`
   * **Action:** Data is read not as tuples, but as contiguous column arrays wrapped in a `DataChunk` object.
5. **Operator Processing**
   * **Component:** `PhysicalHashAggregate`
   * **Source:** `src/execution/operator/aggregate/physical_hash_aggregate.cpp` -> `PhysicalHashAggregate::Execute()`
   * **Action:** The aggregation logic processes the entire `DataChunk` at once, tight-looping over the vectors to compute the sum, maximizing L1 cache hits and allowing the C++ compiler to generate SIMD instructions.

---

## 2. Key Design Decisions

By dissecting DuckDB’s source, we identified three core design decisions that fundamentally define its architecture and performance profile.

### Design Decision 1: Vectorized `DataChunk` Processing
* **Implementation Code:** `src/common/types/data_chunk.cpp` and `src/common/types/vector.cpp`
* **Problem Solved:** Traditional tuple-at-a-time systems invoke `next()` for *every single row*, stalling the CPU on instruction fetching and destroying cache locality. Vectorization batches thousands of values into tight loops.
* **Trade-off Introduced:** While iteration overhead plummets, memory pressure increases as entire chunks must be allocated in active RAM simultaneously. If workloads require massive random access rather than sequential scans, vectorization yields diminishing returns due to cache thrashing.
  <br>*(See empirical validation below: DuckDB outperforms row-based SQLite massively on aggregations)*
  
  ![Vector vs Row Behavior](project/plots/exp_1.png)

### Design Decision 2: Hardcoded Standard Vector Size (`STANDARD_VECTOR_SIZE`)
* **Implementation Code:** `src/include/duckdb/common/vector_size.hpp` (`#define STANDARD_VECTOR_SIZE`)
* **Problem Solved:** Rather than infinitely scaling batch sizes, DuckDB hard-caps `DataChunks` (default `2048`). This guarantees that the working set sizes for operators strictly fit into modern L1/L2 CPU caches, preventing slow RAM fetches.
* **Trade-off Introduced:** A single static size cannot perfectly fit every hardware architecture or query type. Heavy filtering workloads might prefer smaller vectors, while simple scans might prefer larger vectors.
  <br>*(See explicitly tested trade-offs in the Mandatory Experiment section).*

### Design Decision 3: Exploiting SIMD Hardware Acceleration
* **Implementation Code:** Arithmetic paths in `src/execution/expression_executor.cpp`.
* **Problem Solved:** Modern CPUs can execute Single Instruction Multiple Data (SIMD). By aligning data in contiguous vectors, DuckDB allows the hardware to calculate operations (like `+`, `-`, `<`, `>`) on multiple values in a single CPU clock cycle, massively accelerating compute-heavy queries.
* **Trade-off Introduced:** SIMD pipelines are fragile. They rigidly assume contiguous layout. As soon as queries rely heavily on string matching, branching `CASE` logic, or scattered memory lookups, SIMD vectorization breaks down, resulting in expensive fallback logic.

  ![SIMD Speedup Validations](project/plots/exp_2.png)

---

## 3. Concept Mapping

We explicitly mapped the system architecture to four core academic concepts:

| Course Concept | Implementation in DuckDB |
| :--- | :--- |
| **1. Execution: Directed Acyclic Graphs (DAG)** | DuckDB's execution Engine parses SQL into a strict DAG of `PhysicalOperators`. Data flows dimensionally from scan leaf nodes up to the query root through Pipeline states. |
| **2. Execution: Vectorization / Batching** | Instead of standard MapReduce or Volcano iterator tuples, DuckDB uses a vectorized Volcano model, transmitting `DataChunks` between layers. |
| **3. Storage: Cache Locality & Memory Placement** | DuckDB’s columnar in-memory representation prioritizes sequential cache locality. Vectors are sized specifically to exploit CPU L1 data bounds context. |
| **4. Hardware Acceleration (SIMD)** | Code is specifically authored with loops operating on raw C++ arrays specifically so the compiler can output AVX/SSE vector instructions. |

---

## 4. Mandatory Experiment: Modifying the Core System

We actively altered the system's compiled mechanics to prove our understanding of its internal execution boundaries. 

### Hypothesis & Methodology
We hypothesized that DuckDB's `STANDARD_VECTOR_SIZE` exists in a "Goldilocks Zone". 
1. We wrote a Python test harness `Exp_3_Vector_Size_Change.py`.
2. We actively intercepted and patched `src/include/duckdb/common/vector_size.hpp`.
3. We iteratively rebuilt the DuckDB C++ core setting the limits to `[64, 512, 2048, 8192]`.
4. We benchmarked a TPC-H Aggregation query across each newly compiled system.

### Results & Explanations
* **Observation:** Performance severely degraded at extremes (`64` and `8192`), while plateauing efficiently in the middle.
* **Systems Explanation:** Moving from 2048 down to 64 caused DuckDB to act like a row-based system: operator functions were called 32x more frequently, saturating the CPU with function-pointer overhead. 
* Moving from 2048 up to 8192 caused the `DataChunk` memory allocation to structurally exceed the L1/L2 CPU cache lines. The CPU spent its cycles idle, waiting on slow DDR memory page fetches.

![Vector Sizing Performance Curves](project/plots/exp_3.png)
![Vector Sizing Bar Chart](project/plots/exp_3(1).png)

---

## 5. Failure Analysis (Behavior under stress)

We modeled the system boundaries to perform a deeper failure/stress analysis.

### Failure Scenario 1: What happens when data size increases significantly?
* **Observation:** When incrementally expanding the TPC-H SF (Scale Factor), DuckDB's throughput time scaled perfectly linearly.
* **Why this happens:** Due to its component design, DuckDB operates strictly batch-by-batch. Unlike in-memory databases that attempt to manipulate massive singular data structures (causing scaling cliffs), DuckDB simply pushes a higher integer count of `DataChunks` through the same isolated DAG pipeline. As long as system disk/memory has total capacity, analytical query pipelines scale perfectly linearly without choking.

![Scale Factor Linearity](project/plots/exp_4.png)

### Failure Scenario 2: What structural assumptions does this system rely on?
* **Observation:** DuckDB strongly assumes that data access is sequential. When we stressed the execution engine by forcing it to calculate Random Access Joins, throughput collapsed from `10,672 MB/s` to `151 MB/s`.
* **Why this happens:** Vectorization is intrinsically tied to **Cache Locality**. Sequential scans perfectly predict and prefetch the next L1 cache block. Randomly accessing memory via harsh Join permutations causes continuous `Cache Misses`. The engine "fails" to be performant because the vectorized loops are constantly starved of data blocking the CPU, breaking the core architectural assumption.

![Cache Locality Collapse](project/plots/exp_5.png)

---

> **Key Takeaway from reverse-engineering:** DuckDB achieves state-of-the-art analytical speed not by magic, but by aggressively structuring data (via fixed-size vectors) so the CPU hardware itself is fundamentally uninterrupted.
