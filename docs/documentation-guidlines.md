# KaMPI.ng Documentation Guidelines

In this document, we describe the documentation guidelines for the KaMPI.ng library.
We use [Doxygen] and add the documentation directly in the source code.
When it comes to the scope of documentation, we use the following as rule of thumb:

> If you ask yourself, "Should I add documentation to this code?", the answer is always "Yes!"



## General
First, let us take a look at the general guidelines for documentation.
[Doxygen] uses (normal) C++-comments and special commands within these comments to extract and generate the documentation.

Similar to all documents, the documentation has to be written in English.
All documentation should consist of full sentences---do *not* use bullet points.
The documentation is written for others that may not be as familiar with the code as you are, so keep that in mind when you think that a detailed documentation may not be necessary.

### Comment Syntax
Since multi-line comments are not allowed in the [coding guidelines] of this project, we use single-line comments to mark the documentation.
Instead of using plain single-line C++-comments, e.g., `//`, three slashes are required `///` to start a [Doxygen]-comment block.
Note that at least *two* consecutive lines of such comments have to be present to start a comment block that is recognized by [Doxygen].

```
/// I am a multi-line comment block that is used for documentation.
/// Look at me, I am another line.
```

If you want to write an inline/single line comment, after a member of a class or struct, an additional marker `<` is required.
```
class Foo {

  int bar ///< Documentation for bar.

};
```

### Keywords

The most important keywords that will be used throughout the documentation are `@brief`, `@param`, `@tparam`, and `@return`.

Whenever you add documentation to a new part of the code, a brief summary is necessary.
To this end, the `@brief` keyword is used.
A brief description is mandatory for all classes, structs, enums, members, and functions.

```
///
/// @brief This is a small example of how to document code using Doxygen in KaMPI.ng.
class RunningExample {

};
```

For members a brief inline documentation is sufficient.

```
///
/// @brief This is a small example of how to document code using Doxygen in KaMPI.ng.
class RunningExample {

    bool is_used; ///< Marks if the example is used during construction.

};
```

The brief documentation helps to get a quick understanding of the functionality usage of the documented code.
However, sometimes it is necessary to add a more detailed description.
To this end, no keyword is required, instead start a new paragraph below the `@brief` description.
The detailed description is optional and only required if the brief description does not fully explain the functionality or if some additional details require an explanation.

```
/// @brief This is a small example of how to document code using Doxygen in KaMPI.ng.
///
/// Since there are a lot of different ways to document code using Doxygen, we use this class to show how to do so in
/// KaMPI.ng. We will add more and more stuff to this example, until every realisitc case is covered.
class RunningExample {

    bool is_used; ///< Marks if the example is used during construction.

};
```

If we have (template) parameters, they have to be documented, too.
We use the keywords `@tparam` and `@param` to do so.
Note that the `@tparam` keyword can also be used for classes and structs (which is not shown in the example).
The `@tparam` keyword always has to appear before the `@param` keyword.

```
/// @brief This is a small example of how to document code using Doxygen in KaMPI.ng.
///
/// Since there are a lot of different ways to document code using Doxygen, we use this class to show how to do so in
/// KaMPI.ng. We will add more and more stuff to this example, until every realisitc case is covered.
class RunningExample {

    bool is_used; ///< Marks if the example is used during construction.

    /// @brief Example function showing the usage of two more keywords while evaluating it.
    /// @tparam Type Type of the example passed to the function.
    /// @param example Example that is evaluated.
    template <typename Type>
    void evaluate_example(Type const& example) {
        //
        // Do stuff here
        //
    }
};
```

When documenting parameters it is also possible to specify whether the parameter is an input or output parameter via `@param[in]` and `@param[out]` (or both `@param[in,out]`).
Since most parameters are usually input parameters, only `[out]` and `[in,out]` has to be specified explicitly.
Types of parameters do not have to be specified explicitly.
Note that in the example above we write `@param example [...]` instead of `@param Type const& examplet [...]`.

#### Syntax Highlighting and References

#### Using Bibtex

### Scope

In the table below, we give a brief overview which keywords have to be used as part of the documentation in which parts of the code.
Here *Yes* means that the keyword has to be used, *(Yes)* means that the keyword has to be used if applicable, *Optional* means that the keyword can be used if deemed necessary, and a dash (*-*) means that this keyword is not applicable here.

| Part of the Code/Required | `@brief` | detailed description | `@tparam` |
|---------------------------|----------|----------------------|-----------|
| namespace                 |          |                      |           |
| class                     | Yes      | Yes                  | (Yes)     |
| function                  | Yes      | Optional             | (Yes)     |
| member                    | Yes      | Optional             | -         |
| enum                      | Yes      | Optional             | -         |
| struct                    | Yes      | Yes                  | (Yes)     |
|                           |          |                      |           |

## Additional (Non-Code) Material
Since a good documentation does not only document each class and function of the library, we also want to include additional resources, e.g., examples and guides, within out documentation.

[Doxygen]: https://www.doxygen.nl/
[coding guidelines]: coding-guidelines.md
