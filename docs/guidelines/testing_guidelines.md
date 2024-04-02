Testing Guidelines {#testing_guidelines}
============
In this document, we describe how to write tests for the KaMPIng library.
We use [GoogleTest] for writing tests. Write *many* unit tests for your code. Check for nonsensical input and edge cases.

Our general philosophy is:
> If there is no unit test for it, it does not exist.

# Naming and organizing tests
Consider the following class:

```cpp
// file: include/kamping/foo.hpp
class Foo {
public:
    int bar();
    int baz(int x);
}
```

The corresponding test file should be named `tests/foo_test.cpp`. Inside this file, add tests using the following naming scheme: `TEST(FooTest, <member function of Foo>_<description>)`, where `<description>` could for example be: `basics`, `invalid_parameters`, or many others.

The tests for `Foo` may look like this:

```cpp
// file: tests/foo_test.cpp
TEST(FooTest, bar_basics) {
    // ...
}
TEST(Test_Helpers, mpi_datatype_enum) {
    // ...
}
TEST(FooTest, baz_basics) {
    // ...
}
TEST(FooTest, baz_negative_input) {
    // ...
}
TEST(FooTest, baz_invalid_parameter) {
    // ...
}
```

Do not forget to document what your tests are checking.

# Registering tests
If you have written your unit tests, you need to build the tests and register them with `ctest` so they may be executed automatically by our CI or by running `ctest` or `make test` in the build directory.

We provide several CMake helper functions to ease test registration. These helpers are implemented in the CMake modules `KampingTestHelper` and `KaTestrophe`, but usually you should only have to use the former.

To simplify building a specific test by relying on command line completion (e.g. by using `make test_<TAB>`), the test target names should start with the prefix `test_*`.

To register a test `tests/foo_test.cpp` which should not be executed using `mpiexec` use `kamping_register_test(...)` and add the following line to `tests/CMakeLists.txt`

```cmake
kamping_register_test(test_foo FILES foo_test.cpp)
```
This links the test with KaMPIng (which transitively links MPI), adds the test target and registers the test with `ctest`.

If your test `test/foo_mpi_test.cpp` should be executed in parallel using MPI use `kamping_register_mpi_test(...)` like so:

```cmake
kamping_register_mpi_test(test_foo_mpi FILES foo_mpi_test.cpp CORES 1 2 4 8)
```
The test will be executed using 1, 2, 4 and 8 MPI ranks separately.

For a detailed description of the functions available for registering tests, see the reference below.

# Compilation failure tests

If you want to test that a certain piece of code does not compile, you can create a compilation failure test.
This is one of the only possibilities to test that a templated class cannot be instantiated with a specific template parameter or that a `static_assert` fails.

KaMPIng provides a CMake helper for creating compilation failure tests.
Let's look at the example of `mpi_datatype<Datatype>()` which should not compile for types for which there is no equivalent MPI datatype, e.g. `void` or a pointer.

First, we have to create a simple `cpp` file containing the code to be tested.
To reduce the amount of duplicated code, the CMake macro for compilation failures supports sections.
In each compilation, only one of sections (`POINTER`, `VOID`) is enabled.
To test that there is no bug in the remainder of the file, we also define a `else` section for which compilation should succeed.
Our CMake helper will automatically compile the file without any sections macros defined to check this.

This is the content of our compilation failure test for `mpi_datatype<Datatype>()`.

```cpp
using namespace ::kamping;

int main(int argc, char** argv) {
#if defined(POINTER)
    // Calling mpi_datatype with a pointer type should not compile.
    auto result = mpi_datatype<int*>();
// [...] more sections, e.g. function
#elif defined(VOID)
    // Calling mpi_datatype with a void type should not compile.
    auto result = mpi_datatype<void>();
#else
// If none of the above sections is active, this file will compile successfully.
#endif
}
```

And the CMake code to register our tests:

```cmake
kamping_register_compilation_failure_test(
    test_mpi_datatype_unsupported_types
    FILES test_mpi_datatype_unsupported_types.cpp
    SECTIONS "POINTER" "VOID"
)
```

If you want to link libraries other than KaMPIng, use `katestrophe_add_compilation_failure_test`.

# Reference

## KampingTestHelper
The CMake module `KaTestrophe` provides convenience wrappers for registering tests for KaMPIng. 

```cmake
# Convenience wrapper for adding tests for KaMPIng
# This creates the target, links googletest and kamping (which includes MPI as transitive depenency), enables warnings and registers the test.
# The test is executed directly, wihout using `mpiexec`
#
# TARGET_NAME the target name
# FILES the files of the target
#
kamping_register_test(target FILES [filename ...])
```

```cmake
# Convenience wrapper for adding tests for KaMPIng which rely on MPI
# This creates the target, links googletest, kamping and MPI, enables warnings and registers the tests.
# The test is executed using `mpiexec`, using the number of MPI ranks specified.
#
# TARGET_NAME the target name
# FILES the files of the target
# CORES the number of MPI ranks to run the test for
#
kamping_register_mpi_test(target FILES [filename ...] CORES [Integer ...])
```

```cmake
# Convenience wrapper for registering a set of tests that should fail to compile and require KaMPIng to be linked.
#
# TARGET prefix for the targets to be built
# FILES the list of files to include in the target
# SECTIONS sections of the compilation test to build
#
kamping_register_compilation_failure_test(<target name> FILES [filename ... ] SECTIONS [section ...])
```

## KaTestrophe
The CMake module `KaTestrophe` provides general functions for registering unit tests with `ctest`, which should be executed using `mpiexec`. They are agnostic to the library under test and do not depend on KaMPIng. You should only have to use them if you want fine control over what is linked to your test and how it is executed.

```cmake
# Adds an executable target with the specified files FILES and links gtest and the MPI gtest runner
#
# KATESTROPHE_TARGET target name
# FILES the files to include in the target
katestrophe_add_test_executable(target FILES [filename ...])
```

```cmake
# Registers an executable target KATESTROPHE_TEST_TARGET as a test to be executed with ctest.
# The test is executed using `mpiexec`, using the number of MPI ranks specified.
#
# KATESTROPHE_TEST_TARGET target name
# CORES the number of MPI ranks to run the test with
#
katestrophe_add_mpi_test(target CORES [Integer ...])
```

```cmake
# Registers a set of tests which should fail to compile.
#
# TARGET prefix for the targets to be built
# FILES the list of files to include in the target
# SECTIONS sections of the compilation test to build
# LIBRARIES libraries to link via target_link_libraries(...)
#
katestrophe_add_compilation_failure_test(
  TARGET target_prefix
  FILES [filename ...]
  SECTIONS [section ...]
  LIBRARIES [library ...]
)
```

[GoogleTest]: https://google.github.io/googletest/
