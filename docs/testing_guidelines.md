Testing Guidelines {#testing_guidelines}
============
In this document, we describe how to write tests for the KaMPI.ng library.
We use [GoogleTest] for writing tests. Write *many* unit tests for your code. Check for nonsensical input and edge cases.

Our general philosophy is:
> If there is no unit test for it, it does not exist.

# Naming and organizing tests
Name your unit test files `test_corresponding_filename_of_code.cpp`. Inside this file, add tests using the following naming scheme: `TEST(Test_ClassName, function_<description>)`, where `<description>` could for example be: `basics`, `invalid_parameters`, `typedefs_and_using` or many others.

The tests for the global helper function `mpi_datatype<T>()` are in `tests/test_mpi_datatype.cpp` and for example include the following tests:

```cpp
TEST(Test_Helpers, mpi_datatype_basics) {
// ...
TEST(Test_Helpers, mpi_datatype_typedefs_and_using) {
// ...
TEST(Test_Helpers, mpi_datatype_size_t) {
// ...
TEST(Test_Helpers, mpi_datatype_enum) {
// ...
```
As `mpi_datatype<T>()` is not part of any class, we use the organizatorial unit `Helpers` as the class name.

# Registering tests
If you have written your unit tests, you need to build the tests and register them with `ctest` so they may be executed automatically by our CI.

We provide several CMake helper functions to ease test registration. These helpers are implemented in the CMake modules `KampingTestHelper` and `KaTestrophe`.

To register a test `tests/my_unit_test.cpp` which does not rely on MPI use `kamping_register_test(...)` and add the following line to `tests/CMakeLists.txt`

```cmake
kamping_register_test(my_unit_test FILES my_unit_test.cpp)
```
This links the test with KaMPI.ng, adds the test target and registers the test with `ctest`.

If your test `test/my_mpi_unit_test.cpp` should be executed in parallel using MPI use `kamping_register_mpi_test(...)` like so:

```cmake
kamping_register_mpi_test(my_mpi_unit_test FILES my_mpi_unit_test.cpp CORES 1 2 4 8)
```
The test will be executed using 1, 2, 4 and 8 MPI ranks separately.


# Compilation failure tests

```cmake
```

# Reference

## KaTestrophe

```cmake
# Adds an executable target with the specified files FILES and links gtest and the MPI gtest runner
#
# KATESTROPHE_TARGET target name
# FILES the files to include in the target
katestrophe_add_test_executable(target FILES [filename ...])
```

```cmake
# Registers an executable target KATESTROPHE_TEST_TARGET as a test to be executed with ctest
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
## KampingTestHelper

```cmake
# Convenience wrapper for adding tests for KaMPI.ng
# this creates the target, links googletest and kamping, enables warnings and registers the test
#
# TARGET_NAME the target name
# FILES the files of the target
#
kamping_register_test(target FILES [filename ...])
```

```cmake
# Convenience wrapper for adding tests for KaMPI.ng which rely on MPI
# this creates the target, links googletest, kamping and MPI, enables warnings and registers the tests
#
# TARGET_NAME the target name
# FILES the files of the target
# CORES the number of MPI ranks to run the test for
#
kamping_register_mpi_test(target FILES [filename ...] CORES [Integer ...])
```

[GoogleTest]: https://google.github.io/googletest/
