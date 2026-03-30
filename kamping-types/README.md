# kamping-types

A standalone C++17 header-only library that maps C++ types to MPI datatypes.

`kamping-types` is extracted from [KaMPIng](https://github.com/kamping-site/kamping) and can be consumed independently — without the KaMPIng communicator layer.

## Quick Start

```cmake
include(FetchContent)
FetchContent_Declare(
    kamping
    GIT_REPOSITORY https://github.com/kamping-site/kamping.git
    GIT_TAG main
)
FetchContent_MakeAvailable(kamping)

# Link only to the type module, not the full KaMPIng library
target_link_libraries(myapp PRIVATE kamping::types)
```

Then include what you need:

```cpp
#include "kamping/types/mpi_type_traits.hpp"
#include "kamping/types/scoped_datatype.hpp"
#include "kamping/types/struct_type.hpp"

// Obtain an MPI_Datatype for a builtin — no commit required
MPI_Datatype int_type = kamping::types::mpi_type_traits<int>::data_type(); // MPI_INT

// Commit and RAII-manage a contiguous type for float[4]
kamping::types::ScopedDatatype arr_type{kamping::types::mpi_type_traits<float[4]>::data_type()};
MPI_Send(data, 1, arr_type.data_type(), dest, tag, MPI_COMM_WORLD);
// type is freed when arr_type goes out of scope
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `KAMPING_TYPES_BUILD_EXAMPLES` | `ON` (top-level), `OFF` (subdirectory) | Build the example program |
| `KAMPING_TYPES_ENABLE_REFLECTION` | `OFF` | Enable struct reflection via Boost.PFR for arbitrary struct types |

## Headers

| Header | Contents |
|--------|----------|
| `kamping/types/builtin_types.hpp` | `TypeCategory`, `builtin_type<T>`, `is_builtin_type_v<T>` |
| `kamping/types/mpi_type_traits.hpp` | `type_dispatcher<T>()`, `mpi_type_traits<T>`, `has_static_type_v<T>` |
| `kamping/types/contiguous_type.hpp` | `contiguous_type<T,N>`, `byte_serialized<T>` |
| `kamping/types/struct_type.hpp` | `kamping_tag`, `struct_type<T>` |
| `kamping/types/scoped_datatype.hpp` | `ScopedDatatype` — RAII commit/free wrapper |
| `kamping/types/kabool.hpp` | `kabool` — bool wrapper safe for MPI containers |

## Type Dispatch Rules

`type_dispatcher<T>()` maps C++ types to type traits according to these rules:

| C++ type | Result |
|----------|--------|
| MPI builtin (`int`, `double`, `std::complex<float>`, …) | `builtin_type<T>` — named MPI type, no commit |
| Enum | dispatches to underlying type |
| `T[N]`, `std::array<T,N>` | `contiguous_type<T,N>` — must be committed |
| Everything else | `no_matching_type` — specialize `mpi_type_traits<T>` |

Use `has_static_type_v<T>` to check at compile time whether a type is handled.

## Extending for Custom Types

Specialize `mpi_type_traits<T>` to support your own types:

```cpp
struct Point { float x, y, z; };

namespace kamping::types {
// Option 1: use struct_type (requires std::pair/std::tuple, or Boost.PFR reflection)
template <>
struct mpi_type_traits<std::pair<int, double>> : struct_type<std::pair<int, double>> {};

// Option 2: build the type manually
template <>
struct mpi_type_traits<Point> {
    static constexpr bool has_to_be_committed = true;
    static MPI_Datatype data_type() {
        MPI_Datatype type;
        MPI_Type_contiguous(3, MPI_FLOAT, &type);
        return type;
    }
};
} // namespace kamping::types
```

## When Using Full KaMPIng

When linking against `kamping::kamping` instead of `kamping::types`, you additionally get:

- `type_dispatcher<T>()` — also maps trivially-copyable types to `byte_serialized<T>`
- `mpi_datatype<T>()` — returns a committed, environment-managed `MPI_Datatype`
- `include/kamping/types/utility.hpp` — `mpi_type_traits` for `std::pair`
- `include/kamping/types/tuple.hpp` — `mpi_type_traits` for `std::tuple`
