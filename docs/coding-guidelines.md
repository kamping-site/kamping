Coding Guidelines {#coding_guidelines}
============
# General
We are writing a header-only library.
Using the provided `.clang-format` library is mandatory. The CI will reject non-conforming pull-requests.


# Scoping and Naming
* We are working in the `kamping` (KaMPI.ng) namespace to avoid polluting the user's namespace with trivial names as `in`, `out`, or `root`.
* Classes and structs start with an Upper case letter and are using CamelCase.
* Start variables, attributes, functions, and members with a lower case letter. Use `snake_case`, that is, separate words by an underscore (\_). Write acronyms in lower case letters, e.g., `partitioned_msa` and `generate_mpi_failure`. This also applies to KaMPI.ng -> `kamping`.
* `struct`s have only trivial methods if at all, everything more complicated has to be a `class`. `struct`s are always forbidden to have private members or functions.
* `std::pair` leads to hard to read code. Use named pairs (`struct`s) instead.
* Private attributes and member functions *start* with an underscore, e.g., `_name` or `_clear_cache()`.
* Getters and setters are overloaded methods, for example `author.name()` is a getter and `author.name("Turing")` the corresponding setter.
* For methods which return a boolean, consider prefixing the name with `is`, e.g., `is_valid()` instead of `valid()`
* Methods which return a number with non-obvious unit should have the unit as a postfix, for example `length_px()` instead of `length()`.
* Declare methods and attributes in the following order: `public`, `protected`, `private`. Inside these regions: `using` (typedefs) and (scoped) `enum`s, `struct`s and `class`es, attributes, methods.
* Use `.cpp` and `.hpp` as file endings. File names are lowercase and separate word using an underscore. For example `sparse_all2all.hpp`.
* Name local variables with full names. If a name would get too long, write a comment explaining the full meaning of the variable. Use acronyms very sparingly. Some allowed acronyms: `len` for `length (of)`, `num` for `number (of)`, `mpi`.
* Use informative names for templated types. "T1" etc. is only allowed for very general code. Template names are in CamelCase (like classes), e.g. `GeneratorFunction`.
* Types defined with `using` inherit the naming rules of the type they are aliasing.

# Writing Tests
For details on how to write tests see the [Testing Guidelines](testing_guidelines.md).

# Rules for functions
TODO \@Demian \@Matthias: Rules for API
* For internal functions: Return output only values via the `return` statement, even multiple return values.
* For internal functions: Return in-output values via a reference argument.
* Mark the parameters as `const` where possible.
* Mark member functions as `const` where possible. Use `mutable` for caches to be able to keep getters `const`. 
* For now, we avoid inheritance until somebody needs it. We can then think about rules for using it.
* Use templated methods instead of passing a function pointer etc. as this allows the compiler to inline the function. Use `static_assert` to check the signature of the function being passed as an argument.
* Implicit conversions are forbidden, constructors with only one parameter have to be marked as `explicit`.
* Is one of the following functions declared, either implement, `default`, or `delete` all the others: Constructor, copy constructor, move constructor, copy assignment operator, move assignment operator, destructor. If you don't need some of them and don't want to think about if they would be easy to implement, `delete` them. Either way, write a comment explaining either why you think they should not be implemented, are hard to implement, or that you don't need them and didn't want to bother implementing them.

# Header files and includes
* Use `#pragma once` instead of include guards, as it is available on all important compilers.
* Include all used headers, don't rely on transitive inclusions. Maybe use `include-what-you-use` to check.
* The `#include` statements will be automatically sorted by clang-format inside blocks separated by a whitespace. Use the following, whitespace-separated blocks: 1. System header, 2. STL header, 3. External library header, 4. Project header.
* Do not use `using namespace` in header files as this would then also apply to all files including this header file.
* Use the `kamping/` prefix when including our own header files.

# General
* Use east side `const`.
* Use the more concise `size_t` over `std::size_t`. Do not put `using size_t = std::size_t` into your code.
* Use `nullptr` instead of `NULL` or `0`.
* Use `using` instead of `typedef`.
* Use `sizeof` instead of constants.
* Use C++ style casts; if possible, use `asserting_cast<destType>(srcValve)` or `throwing_cast<destType>(srcValue)`.
* Use `//` instead of `/* */` for comments, even multi-line comments.
* Add Doxygen documentation for each (!) function and member variable, even private ones. Describe what this function does, which parameters it takes and what the return value is. Also describe all assumptions you make. This rule applies to all functions, even trivial ones.
* Start comments describing TODOs with `// TODO ...`. This allows grepping for TODOs.
* Use almost always `auto`.
* Use scoped `enum`s instead of unscoped `enum`s.
* Prefer smart pointers over raw pointers.
* Avoid `std::bind`, use lambda functions instead as they result in better readability and allow the compiler to inline better.
* Use the subset of `C++` which compiles in `gcc10`, `clang10` and `icc19`.
* Use the `KASSERT()` macro with the appropriate assertion level to validate the internal state of your code.
* Use the `KTHROW()` macro to validate user-supplied data.

# Warnings
Code *should* compile with `clang` and `gcc` (not `icc`) without warning with the warning flags given below. If you want to submit code which throws warnings, at least two other persons have to agree. Possible reasons for this are: False-positive warnings.

```
# TODO: Use CheckCXXCompilerFlag for this?
list(
    APPEND WARNING_FLAGS
    "-Wall"
    "-Wextra"
    "-Wconversion"
    "-Wnon-virtual-dtor"
    "-Woverloaded-virtual"
    "-Wshadow"
    "-Wsign-conversion"
    "-Wundef"
    "-Wunreachable-code"
    "-Wunused"
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    list(
        APPEND WARNING_FLAGS
        "-Wcast-align"
        "-Wnull-dereference"
        "-Wpedantic"
    )
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    list(
        APPEND WARNING_FLAGS
        "-Wcast-align"
        "-Wnull-dereference"
        "-Wpedantic"
        "-Wnoexcept"
        "-Wsuggest-attribute=const"
        "-Wsuggest-attribute=noreturn"
        "-Wsuggest-override"
    )
endif()

# OFF by default.
if(WARNINGS_ARE_ERRORS)
  list(
    APPEND WARNING_FLAGS
    "-Werror"
    )
endif()
```

# Tooling and Workflow
* Use "modern" CMake as build system
* Use Doxygen for documentation. TODO \@Florian: Which style?
* Use `git submodule` to include dependencies. TODO: Explain in the README, how to work with git submodules.
* Write *many* unit tests for your code. If there is no unit test for it, it does not exist. Also check for nonsensical inputs and edge cases.
