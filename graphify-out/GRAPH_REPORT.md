# Graph Report - .  (2026-07-06)

## Corpus Check
- Corpus is ~5,293 words - fits in a single context window. You may not need a graph.

## Summary
- 206 nodes · 279 edges · 9 communities
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 11 edges (avg confidence: 0.9)
- Token cost: 5,200 input · 3,400 output

## Community Hubs (Navigation)
- [[_COMMUNITY_ComputeUnit Internals|ComputeUnit Internals]]
- [[_COMMUNITY_Core Orchestration|Core Orchestration]]
- [[_COMMUNITY_ComputeUnit C-API|ComputeUnit C-API]]
- [[_COMMUNITY_Core C-API|Core C-API]]
- [[_COMMUNITY_Kernel Definition|Kernel Definition]]
- [[_COMMUNITY_Buffer Operations|Buffer Operations]]
- [[_COMMUNITY_Buffer Synchronization|Buffer Synchronization]]
- [[_COMMUNITY_Project Docs & Build|Project Docs & Build]]
- [[_COMMUNITY_Pre-commit Script|Pre-commit Script]]

## God Nodes (most connected - your core abstractions)
1. `ComputeUnit` - 50 edges
2. `Core` - 41 edges
3. `Kernel` - 13 edges
4. `string` - 8 edges
5. `KernelTiming` - 7 edges
6. `ulong` - 6 edges
7. `writeBuffer()` - 6 edges
8. `initialize()` - 6 edges
9. `writeBuffer()` - 6 edges
10. `readBuffer()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Multi-GPU workload splitting (fraction/index/last)` --rationale_for--> `run()`  [INFERRED]
  OCLW_Core.cpp → OCLW_CU.cpp
- `Core class` --shares_data_with--> `Kernel`  [INFERRED]
  OCLW_Core.hpp → OCLW_Kernel.hpp
- `ComputeUnit class` --shares_data_with--> `Kernel`  [INFERRED]
  OCLW_CU.hpp → OCLW_Kernel.hpp
- `Core class` --shares_data_with--> `ComputeUnit class`  [INFERRED]
  OCLW_Core.hpp → OCLW_CU.hpp
- `setProfiling()` --calls--> `setProfiling()`  [EXTRACTED]
  OCLW_Core.cpp → OCLW_CU.cpp

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Multi-GPU Workload Splitting Pipeline** — oclw_core_buildcomputeunits, oclw_core_buildcomputeunitfordevice, oclw_cu_computeunit, oclw_cu_run, concept_multi_gpu_workload_splitting [INFERRED 0.85]
- **Build & Format Toolchain** — cmakelists, build, readme, pre_commit_config, scripts_ensure_blank_lines [INFERRED 0.75]
- **Kernel Profiling Pipeline** — oclw_cu_run, oclw_cu_setprofiling, oclw_cu_buildqueue, oclw_cu_printprofilingresults, oclw_cu_getkerneltimings, concept_profiling_accumulation [INFERRED 0.85]

## Communities (9 total, 0 thin omitted)

### Community 0 - "ComputeUnit Internals"
Cohesion: 0.04
Nodes (45): CommandQueue, Context, addKernel, addSource, buildContext, buildProgram, buildQueue, clearKernels (+37 more)

### Community 1 - "Core Orchestration"
Cohesion: 0.05
Nodes (39): ComputeUnit, addJobToDevice, addKernel, addSourceFile, buildComputeUnitForDevice, buildComputeUnits, buildDevices, buildDevicesUsageMap (+31 more)

### Community 2 - "ComputeUnit C-API"
Cohesion: 0.10
Nodes (19): OpenCL fast-relaxed-math build options, printProfilingResults(), setProfiling(), addKernel(), buildContext(), buildProgram(), buildQueue(), Device (+11 more)

### Community 3 - "Core C-API"
Cohesion: 0.10
Nodes (21): Multi-GPU workload splitting (fraction/index/last), Per-kernel profiling time accumulation, addKernel(), addSourceFile(), buildDevices(), buildDevicesUsageMap(), buildPlatforms(), Kernel (+13 more)

### Community 4 - "Kernel Definition"
Cohesion: 0.12
Nodes (19): cl_ulong, addJobToDevice(), buildComputeUnitForDevice(), buildComputeUnits(), Core class, Device, Core(), removeJobFromDevice() (+11 more)

### Community 5 - "Buffer Operations"
Cohesion: 0.24
Nodes (15): Buffer, addArgument(), allocateBuffer(), fillBuffer(), map, string, T, ulong (+7 more)

### Community 6 - "Buffer Synchronization"
Cohesion: 0.36
Nodes (11): mutex, addArgument(), allocateBuffer(), fillBuffer(), string, T, ulong, vector (+3 more)

### Community 7 - "Project Docs & Build"
Cohesion: 0.25
Nodes (5): Apache 2.0 + Commons Clause license model, CMake presets build configuration, Code organization standard (class layout + separators), Readability via temporary variables, build.sh script

### Community 8 - "Pre-commit Script"
Cohesion: 0.43
Nodes (5): in_macro_body(), is_blank(), main(), process_file(), Check if line i is inside a multi-line macro (previous line ends with \\).

## Knowledge Gaps
- **103 isolated node(s):** `Device`, `uint`, `KernelTiming`, `ulong`, `kernelName` (+98 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ComputeUnit class` connect `Kernel Definition` to `Buffer Operations`?**
  _High betweenness centrality (0.283) - this node is a cross-community bridge._
- **Why does `ComputeUnit` connect `ComputeUnit Internals` to `Buffer Operations`?**
  _High betweenness centrality (0.263) - this node is a cross-community bridge._
- **Why does `KernelTiming` connect `Buffer Operations` to `Kernel Definition`?**
  _High betweenness centrality (0.223) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Kernel` (e.g. with `Core class` and `ComputeUnit class`) actually correct?**
  _`Kernel` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Device`, `uint`, `KernelTiming` to the rest of the system?**
  _110 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `ComputeUnit Internals` be split into smaller, more focused modules?**
  _Cohesion score 0.044444444444444446 - nodes in this community are weakly interconnected._
- **Should `Core Orchestration` be split into smaller, more focused modules?**
  _Cohesion score 0.05128205128205128 - nodes in this community are weakly interconnected._