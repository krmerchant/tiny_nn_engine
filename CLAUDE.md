# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TinyNNEngine** is a TensorRT-like inference-only engine targeting ONNX models, written in C++ with CUDA kernels. The primary demo target is a 2-layer MLP (MNIST digit classification). This project is for JHU's Intro to GPU course.

## Generating the ONNX Model

```bash
cd tools/
python3 generate_mnist_onnx.py --output mnist_fc.onnx --data-dir ./data --epochs 10
```

Downloads MNIST, trains to ≥97% accuracy, and writes `tools/mnist_fc.onnx`.

## Build System

The project will use CMake with CUDA support. Expected build commands once scaffolded:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Run the main demo:
```bash
./build/tinyinfer_demo mnist_fc.onnx path/to/mnist/
```

## Planned Directory Layout

```
include/tinyinfer/      # Public headers
src/                    # C++ implementation files
src/ops/                # Op wrappers (.cpp) and CUDA kernels (.cu)
```

## Architecture

### Execution Flow

1. `Model::load("file.onnx")` → ONNX parser reads protobuf, uploads initializer tensors to GPU, produces a `Graph`
2. `Executor(model, cfg)` → allocates intermediate buffers lazily on a persistent CUDA stream
3. `executor.run({input})` → iterates the topologically-sorted `Graph::nodes`, resolves tensors from a `value_map_`, dispatches kernels through `OpRegistry`

### Key Modules

| Module | Files | Role |
|--------|-------|------|
| **Tensor** | `include/tinyinfer/tensor.h`, `src/tensor.cpp` | GPU buffer abstraction; owns device memory via `shared_ptr<float>` with `cudaFree` deleter; provides H2D/D2H copy, `argmax`, shape metadata |
| **Graph** | `include/tinyinfer/graph.h`, `src/graph.cpp` | Topologically-sorted `vector<Node>` + name-keyed weight map (`initializers`); mirrors ONNX value naming directly |
| **ONNX Parser** | `src/onnx_parser.cpp` | Internal (no public header); parses `onnx::ModelProto` via protobuf; maps op strings → `OpType` enum; uploads initializers to GPU |
| **Model** | `include/tinyinfer/model.h`, `src/model.cpp` | Public ONNX load entry point; owns the `Graph`; exposes `print_graph()` |
| **OpRegistry** | `include/tinyinfer/op_registry.h`, `src/op_registry.cpp` | Singleton factory: `OpType` → `OpKernel`; each op `.cpp` self-registers at static-init time |
| **CUDA Kernels** | `src/ops/gemm_kernel.cu`, `src/ops/relu_kernel.cu`, `src/ops/softmax_kernel.cu` | Pure CUDA; Gemm uses shared-memory tiling (tiled SGEMM); Softmax uses row-wise log-sum-exp with `__syncthreads` |
| **Op Wrappers** | `src/ops/gemm_op.cpp`, `src/ops/relu_op.cpp`, `src/ops/softmax_op.cpp` | Thin `OpKernel` subclasses; resolve tensor pointers from `KernelContext`, launch `.cu` kernels |
| **Executor** | `include/tinyinfer/executor.h`, `src/executor.cpp` | Drives the inference loop; manages CUDA stream and lazy intermediate buffer allocation |
| **Profiler** | `include/tinyinfer/profiler.h`, `src/profiler.cpp` | Wraps `cudaEvent_t` pairs around kernel launches; reports per-op timing after `cudaStreamSynchronize` |
| **MNISTLoader** | `include/tinyinfer/mnist_loader.h`, `src/mnist_loader.cpp` | Reads IDX binary format; normalizes pixels to `[0,1]`; CPU-only |
| **Evaluator** | `include/tinyinfer/evaluator.h`, `src/evaluator.cpp` | Stateless; batches through test set, calls `argmax(1)` on outputs, accumulates accuracy + 10×10 confusion matrix |

### Target ONNX Graph (MNIST MLP)

```
input(1×784) → [Gemm] → fc1_out(1×128) → [ReLU] → relu_out(1×128) → [Gemm] → fc2_out(1×10) → [Softmax] → output(1×10)
```

Initializers (`graph_.initializers`): `fc1.weight (128×784)`, `fc1.bias (128)`, `fc2.weight (10×128)`, `fc2.bias (10)`

### Design Decisions

- **Value map pattern**: activations are keyed by ONNX tensor name in `value_map_` (avoids a translation layer between ONNX names and internal IDs)
- **Self-registering ops**: each op `.cpp` registers itself into `OpRegistry` at static-init time — adding a new op requires no changes to core files
- **Shared-ptr with custom deleter**: `Tensor` uses `shared_ptr<float>` with `cudaFree` as deleter for automatic GPU memory management
- **`tinyinfer::internal` namespace**: anything not part of the public API lives in `tinyinfer::internal` (e.g. `parse_onnx` in `src/onnx_parser.h`); code under `include/tinyinfer/` should not expose `internal` symbols

## Coding Conventions

- **Naming**: `snake_case` for all identifiers (variables, functions, methods, member variables, namespaces, file names) except class and struct names, which use `UpperCamelCase` (e.g. `class OpRegistry`, `struct KernelContext`, but `void launch_kernel(...)`, `int batch_size`)

## Success Criteria

- ≥ 97% accuracy on MNIST test set (10,000 samples)
- Per-op profiling table (Gemm, ReLU, Gemm, Softmax)
- Latency comparison vs. PyTorch CPU and CUDA baselines
