# OpenCLWrapper Project

OpenCLWrapper is a C++ wrapper library for OpenCL that simplifies GPU kernel management and execution.

## Building

```bash
mkdir -p build && cd build
cmake ..
make
```

## Architecture & Design Principles

### Readability: Use Temporary Variables

**NEVER** inline complex expressions (`.data()`, `const_cast`, array indexing, struct member access) inside function call arguments. Extract them into named temporary variables first.

### Tests for New Features

Every new feature **MUST** have tests added:
- Unit tests for the new functionality in isolation
- Integration tests exercising the feature through the full pipeline

## Code Organization Standard

### `.hpp` files - Class Layout

- Access specifier order: `public:` -> `protected:` -> `private:` (one section each, never duplicate)
- Within each access specifier:
  1. Types / Aliases / Nested types
  2. Constructors / Destructors
  3. Methods grouped by purpose with `//-- Section Name --//` headers
  4. Members grouped by purpose with `//-- Section Name --//` headers

### `.cpp` files

- Method implementations follow the **exact same order** as declarations in the `.hpp`
- Every method separated by:
  ```
  //===================================================================================================================//
  ```
- Section headers `//-- Section Name --//` for complex files (matching `.hpp` sections), placed between separator lines:
  ```
  //===================================================================================================================//
  //-- Section Name --//
  //===================================================================================================================//
  ```
- Static member initializations at the top (before methods)
- Template instantiations at the bottom

## Commit Messages

Use [conventional commits](https://www.conventionalcommits.org/) with the format `type(scope): description`:

| Prefix   | Use for                                   |
|----------|-------------------------------------------|
| `feat`   | New features                              |
| `fix`    | Bug fixes                                 |
| `refactor` | Code refactoring                        |
| `perf`   | Performance improvements                  |
| `test`   | Adding or updating tests                  |
| `docs`   | Documentation changes                     |
| `chore`  | Maintenance tasks (build, deps, etc.)     |
| `build`  | Changes to build system (CMake, etc.)     |

- Subject line: imperative mood, lowercase type, no trailing period.
- Body: explain **why** the change was made.
- Never add a `Co-Authored-By` or any co-author trailer.
