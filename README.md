# OpenCLWrapper

A C++ wrapper library for OpenCL that simplifies GPU kernel management and execution.

## Building

### Prerequisites

- CMake 3.14+
- C++17 compiler (gcc, clang, or MSVC)
- Qt 6 (static or shared)
- OpenCL SDK

### Quick Start

```bash
# 1. Create CMakeUserPresets.json (machine-specific, git-ignored)
#    See the template below for the required structure.

# 2. Build
./build.sh              # default: static preset
./build.sh shared       # shared-library preset
```

### CMakeUserPresets.json

Create `CMakeUserPresets.json` in the project root (git-ignored). It should pin your Qt6 kit paths:

```json
{
  "version": 3,
  "cmakeMinimumRequired": { "major": 3, "minor": 3.21 },
  "configurePresets": [
    {
      "name": "static",
      "displayName": "Static Qt6 (self-contained binary)",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "CMAKE_PREFIX_PATH": "/path/to/Qt-6.8.2-static",
        "CMAKE_BUILD_TYPE": "Release"
      }
    },
    {
      "name": "shared",
      "displayName": "Shared Qt6",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "CMAKE_PREFIX_PATH": "/path/to/Qt-6.8.2-shared",
        "CMAKE_BUILD_TYPE": "Release"
      }
    }
  ],
  "buildPresets": [
    { "name": "static", "configurePreset": "static" },
    { "name": "shared", "configurePreset": "shared" }
  ]
}
```

Replace `/path/to/Qt-6.8.2-static` and `/path/to/Qt-6.8.2-shared` with your actual Qt installation paths.

### Manual CMake

You can also build without presets:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/Qt-6.8.2-static -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

## Architecture & Design Principles

### Readability: Use Temporary Variables

**NEVER** inline complex expressions (`.data()`, `const_cast`, array indexing, struct member access) inside function call arguments. Extract them into named temporary variables first.

### Tests for New Features

Every new feature **MUST** have tests added:
- Unit tests for the new functionality in isolation
- Integration tests exercising the feature through the full pipeline

## Code Organization Standard

### `.hpp` files — Class Layout

- Access specifier order: `public:` → `protected:` → `private:` (one section each, never duplicate)
- Within each access specifier:
  1. Types / Aliases / Nested types
  2. Constructors / Destructors
  3. Methods grouped by purpose with `//-- Section Name --//` headers
  4. Members grouped by purpose with `//-- Section Name --//` headers

### `.cpp` files

- Method implementations follow the **exact same order** as declarations in the `.hpp`
- Every method separated by:
  `//===================================================================================================================//`
- Section headers `//-- Section Name --//` for complex files (matching `.hpp` sections), placed between separator lines
- Static member initializations at the top (before methods)
- Template instantiations at the bottom

## License

Apache 2.0 with Commons Clause Restriction. See [LICENSE.md](LICENSE.md) for details.
