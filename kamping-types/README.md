# kamping-types

A standalone C++17 header-only library that maps C++ types to MPI datatypes and reduction operations.

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
| `kamping/types/reduce_ops.hpp` | `kamping::ops::` functors, `mpi_operation_traits<Op,T>`, `ScopedOp`, `with_operation_functor` |

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

## Reduction Operations

`kamping/types/reduce_ops.hpp` provides a C++ functor vocabulary that maps to MPI's builtin reduction ops.

### Builtin Functor Types (`kamping::ops::`)

| Functor | MPI constant | Identity |
|---------|-------------|---------|
| `ops::max<T>` | `MPI_MAX` | `std::numeric_limits<T>::lowest()` |
| `ops::min<T>` | `MPI_MIN` | `std::numeric_limits<T>::max()` |
| `ops::plus<T>` | `MPI_SUM` | `0` |
| `ops::multiplies<T>` | `MPI_PROD` | `1` |
| `ops::logical_and<T>` | `MPI_LAND` | `true` |
| `ops::logical_or<T>` | `MPI_LOR` | `false` |
| `ops::logical_xor<T>` | `MPI_LXOR` | `false` |
| `ops::bit_and<T>` | `MPI_BAND` | `~T{0}` |
| `ops::bit_or<T>` | `MPI_BOR` | `T{0}` |
| `ops::bit_xor<T>` | `MPI_BXOR` | `T{0}` |

All functors default to `T = void`, enabling deduced argument types.

### Querying the MPI_Op at Compile Time

`mpi_operation_traits<Op, T>` maps a (functor, element type) pair to its builtin `MPI_Op`:

```cpp
#include "kamping/types/reduce_ops.hpp"

using T = mpi_operation_traits<kamping::ops::plus<>, int>;
static_assert(T::is_builtin);           // true — maps to MPI_SUM
MPI_Op op = T::op();                    // MPI_SUM
int identity = T::identity;             // 0
```

`is_builtin` is `false` for functors not covered by a predefined MPI operation (e.g., `std::minus<>`).

### RAII Handle: `ScopedOp`

`ScopedOp` wraps an `MPI_Op` and calls `MPI_Op_free` on destruction only when it owns the op (user-defined ops created via `MPI_Op_create`). Predefined constants are wrapped non-owning.

Analogous to `ScopedDatatype` for `MPI_Datatype`.

```cpp
MPI_Op raw_op;
MPI_Op_create(&my_reduce_fn, /*commute=*/1, &raw_op);
kamping::types::ScopedOp scoped{raw_op, /*owns=*/true};
// MPI_Op_free called automatically when scoped goes out of scope
```

### Runtime Dispatch: `with_operation_functor`

`with_operation_functor` maps a runtime `MPI_Op` handle to its C++ functor and invokes a callable:

```cpp
kamping::types::with_operation_functor(MPI_SUM, [](auto op) {
    // op is kamping::ops::plus<>{}
});
```

Unknown ops dispatch to `kamping::ops::null<>{}`.

### Commutativity Tags

User-defined operations can be tagged:

```cpp
kamping::ops::commutative      // ops::internal::commutative_tag
kamping::ops::non_commutative  // ops::internal::non_commutative_tag
```

## When Using Full KaMPIng

When linking against `kamping::kamping` instead of `kamping::types`, you additionally get:

- `type_dispatcher<T>()` — also maps trivially-copyable types to `byte_serialized<T>`
- `mpi_datatype<T>()` — returns a committed, environment-managed `MPI_Datatype`
- `include/kamping/types/utility.hpp` — `mpi_type_traits` for `std::pair`
- `include/kamping/types/tuple.hpp` — `mpi_type_traits` for `std::tuple`
