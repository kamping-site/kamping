Documentation Guidelines {#documentation_guidelines}
============

In this document, we describe the documentation guidelines for the KaMPIng library.
We use [Doxygen] and add the documentation directly in the source code.
When it comes to the scope of documentation, we use the following as rule of thumb:

> If you ask yourself, "Should I add documentation to this code?", the answer is always "Yes!"



# General
First, let us take a look at the general guidelines for documentation.
[Doxygen] uses (normal) C++-comments and special commands within these comments to extract and generate the documentation.

Similar to all documents, the documentation has to be written in English.
All documentation should consist of full sentences---do *not* use bullet points.
The documentation is written for others that may not be as familiar with the code as you are, so keep that in mind when you think that a detailed documentation may not be necessary.

# Comment Syntax
Since multi-line comments are not allowed in the [coding guidelines] of this project, we use single-line comments to mark the documentation.
Instead of using plain single-line C++-comments, e.g., `//`, three slashes are required `///` to start a [Doxygen]-comment block.
Note that at least *two* consecutive lines of such comments have to be present to start a comment block that is recognized by [Doxygen].

```cpp
/// I am a multi-line comment block that is used for documentation.
/// Look at me, I am another line.
```

If you want to write an inline/single line comment, after a member of a class or struct, an additional marker `<` is required.

```cpp
class Foo {

  int bar ///< Documentation for bar.

};
```

# Keywords

The most important keywords that will be used throughout the documentation are `@brief`, `@param`, `@tparam`, and `@return`.

Whenever you add documentation to a new part of the code, a brief summary is necessary.
To this end, the `@brief` keyword is used.
A brief description is mandatory for all classes, structs, enums, members, and functions.

```cpp
///
/// @brief This is a small example of how to document code using Doxygen in KaMPIng.
class RunningExample {

};
```

For members a brief inline documentation is sufficient.
Note that both public *and* private members and functions (see [example](@ref example-of-minimal-documentation-style)) have to be documented.

```cpp
///
/// @brief This is a small example of how to document code using Doxygen in KaMPIng.
class RunningExample {

public:
    bool is_used; ///< Marks if the example is used during construction.

private:
    int examples_in_use_; ///< Counts the number of examples that are currently in use.

};
```

The brief documentation helps to get a quick understanding of the functionality usage of the documented code.
However, sometimes it is necessary to add a more detailed description.
To this end, no keyword is required, instead start a new paragraph below the `@brief` description.
The detailed description is mandatory for classes and structs and optional for everything else, where it is only required if the brief description does not fully explain the functionality or if some additional details require an explanation.

```cpp
/// @brief This is a small example of how to document code using Doxygen in KaMPIng.
///
/// Since there are a lot of different ways to document code using Doxygen, we use this class to show how to do so in
/// KaMPIng. We will add more and more stuff to this example, until every realisitc case is covered.
class RunningExample {

public:
    bool is_used; ///< Marks if the example is used during construction.

private:
    int examples_in_use_; ///< Counts the number of examples that are currently in use.

};
```

If we have (template) parameters, they have to be documented, too.
We use the keywords `@tparam` and `@param` to do so.
Note that the `@tparam` keyword can also be used for classes and structs (which is not shown in the example).
The `@tparam` keyword always has to appear before the `@param` keyword.

```cpp
/// @brief This is a small example of how to document code using Doxygen in KaMPIng.
///
/// Since there are a lot of different ways to document code using Doxygen, we use this class to show how to do so in
/// KaMPIng. We will add more and more stuff to this example, until every realisitc case is covered.
class RunningExample {

public:
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

private:
    int examples_in_use_; ///< Counts the number of examples that are currently in use.

};
```

When documenting parameters it is also possible to specify whether the parameter is an input or output parameter via `@param[in]` and `@param[out]` (or both `@param[in,out]`).
Since most parameters are usually input parameters, only `[out]` and `[in,out]` has to be specified explicitly.
Types of parameters do not have to be specified explicitly, i.e., we write `@param example [...]` instead of `@param Type const& example [...]`.

We can document the return value of functions using the `@return` keyword.
If multiple cases of return values are documented, `@return` can be used for each case to improve readability of the documentation in the code.
They are automatically concatenated in the documentation.
With this keyword, we finally have everything we need to obtain the minimal documentation that is required for everything in the KaMPIng code base (as shown in the example below).
We say *minimal* because the documentation can be improved by using special formatting, cross-references, and many more features, which we discuss further [below](@ref formatting-and-references).

# Example of Minimal Documentation Style {#example-of-minimal-documentation-style}

```cpp
/// @brief This is a small example of how to document code using Doxygen in KaMPIng.
///
/// Since there are a lot of different ways to document code using Doxygen, we use this class to show how to do so in
/// KaMPIng. We will add more and more stuff to this example, until every realisitc case is covered.
class RunningExample {

public:
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

private:
    int examples_in_use_; ///< Counts the number of examples that are currently in use.

    /// @brief Checks if an example is currently somewhere and should not be reused.
    /// @param example Example for which it is checked whether it is currently used.
    /// @return \c true if the example is used somewhere.
    /// @return \c false otherwise.
    bool is_currently_used(RunningExample const& example) {
        //
        // Do stuff here
        //
    }
};
```

# Global Functions
Doxygen considers all files as private per default.
To make the (required) documentation of global functions appear in the generated documentation, all files containing global functions must contain a `\\\ @file` at the top of the file without any label.
Additionally, a `\\\ @brief Description` of the file's content should be given.

```cpp
\\\ @file
\\\ @brief Global function used in this example.

\\\ @brief Example of a global function
void example() {
    // Do stuff here
}
```

# tl;dr

In the example above, we also have added the first formatting commands with `\c`, which sets the following word in typewriter font.
Before we take a look at the formatting that should be used in the KaMPIng documentation, we give a short overview what parts of the documentation are required.
In the table below, we give a brief overview which keywords have to be used as part of the documentation in which parts of the code.
Here *Yes* means that the keyword has to be used, *(Yes)* means that the keyword has to be used if applicable, *Optional* means that the keyword can be used if deemed necessary, and a dash (*-*) means that this keyword is not applicable here.

| Part of the Code/Required        | `@brief` | detailed description | `@tparam` | `@param` | `@return` | `@file` including `@brief` |
|----------------------------------|----------|----------------------|-----------|----------|-----------|----------------------------|
| class                            | Yes      | Yes                  | (Yes)     | -        | -         | -                          |
| public **and** private functions | Yes      | Optional             | (Yes)     | (Yes)    | (Yes)     | -                          |
| global functions                 | Yes      | Optional             | (Yes)     | (Yes)    | (Yes)     | Yes                        |
| public **and** private members   | Yes      | Optional             | -         | -        | -         | -                          |
| enum                             | Yes      | Optional             | -         | -        | -         | -                          |
| struct                           | Yes      | Yes                  | (Yes)     | -        | -         | -                          |


# TODOs
While all code should be finalized (finished and polished) before it is merged into the main branch of the KaMPIng repository, sometimes there are open *todos* that cannot yet be fixed or may be fixed later as part of a bigger update.
To better keep up with this type of todo, we want to create a list of these them.
Fortunately, [Doxygen] provides an easy aggregation of all *todos* in the code.
To this end, we have to mark all todos using the `@todo` command, where we summarize the todo. Small steps needed to complete it can be described using `\\ TODO`.
This way, we can easily keep track of open todos in our documentation.

```cpp
/// @brief An example function that does stuff.
void example_function() {
    if (dummy_value > 0) {
        /// @todo Throw an exception instead of simply aborting as soon as our exception handling is included.
        // TODO throw exception here and
        // TODO handle it in all places where this function is called.
        std::abort();
    }
    // do stuff here
}

```

# Formatting and References {#formatting-and-references}

As mentioned above, the documentation can be formatted with different commands.
Due to better readability of the unprocessed documentation, we use a backslash `\` to enable formatting commands similar to how LaTeX works.
(`@` would also work but is strongly discouraged in KaMPIng.)

- **Typwriter Font** `\c`

  We use typewriter font to highlight language specific concepts like boolean types `true` and `false`, character types like `signed char` and `unsigned char`, `nullptr`, or fixed size integers like `uint16_t` or `int64_t`.
  Note that [Doxygen] is unaware of any C++-Standard-Library-Classes so if such classes (e.g., `std::vector` or `std::array`) are used, they should also be formatted using typewriter font.
- **References to Other Parts of the Code** `\ref` or done automatically

  Sometimes it may be useful to refer to other parts of the documentation, e.g., if there is an alternative to a function that does similar things the user might be looking for or if there is more relevant documentation at another part of the code.
  Fortunately, [Doxygen] has a well-working [automatic link generation].
  For most documented parts of the code, links will be added automatically.
  More precisely, links are auto-generated for classes that contain at least one non-lower case character.
  If a class should not fulfill this requirement  check the [coding guidelines] because that should not happen or use `\ref` to manually add a reference to the class.
  Links to (member) functions are also added automatically.
  Please refer to the [automatic link generation] guide for more information on the required formatting if the generation should not work directly.
  The same holds for members.

  One important rule when it comes to adding references, do *not* simply add a reference to another part of the documentation without describing what to find there if it is not clear from context (see [this discussion](https://meta.stackexchange.com/q/225370) for an explanation).
- **Formulas and Math Font Support** `\f$ Formular \f$`

  The documentation supports LaTeX syntax to generate good looking formulas and enable math font support.
  For example `\f$O(n\lg\frac{n}{k})\f$` renders a nice looking asymptotic bound.
- **Citing** `\cite <BibTeX Key>`

  Resources can also be cited using BibTeX.
  To this end, the reference has to be placed in the global [literature.bib] file.
  Please read the [following section](@ref references) on how to format the BibTeX entry.
  Then, it can be cited by using the `\cite <BibTeX key>` command.

# Structuring the Documentation
[Doxygen] can be used to [group] different parts of the documentation independent of the structure of the source code.
We want to keep the number of groups small but still use this feature to make different aspects of KaMPIng easily discoverable in the documentation.
Documentation can be added to a group via the `@addtogroup` command, e.g.,

```cpp
/// @addtogroup <label>
/// @{

// ...

/// @}
```
All groups have to be defined somewhere.
To this end, we use the main file of our documentation [main.dox], see [below](@ref main-page).
Here, we can define the group using the following commands.

```
@defgroup <label> <Printed Name>
@brief Brief description of the group. Can be followed by a detailed description (separated by empty line).
```

# Additional (Non-Code) Material
Since a good documentation does not only document each class and function of the library, we also want to include additional resources, e.g., examples and guides, within out documentation.

## Main Page {#main-page}
The main page (starting page) of the documentation is [main.dox].
Here, general information and links to additional resources should be added.
Additionally, we can add a brief description of all groups here.

## Guides and Examples

Guides and examples should be added in form of Markdown files and linked in [main.dox].
See this file for an example.

## References {#references}
All references have to be added as BibTeX entry in [literature.bib].
There are two cases: either the reference is available in [dblp] or not.
If its available in [dblp], please just copy the *condensed* BibTeX entry to [literature.bib] and add the *DOI* if available.

```bibtex
@article{DBLP:journals/siamcomp/KnuthMP77,
  author    = {Donald E. Knuth and
               James H. Morris Jr. and
               Vaughan R. Pratt},
  title     = {Fast Pattern Matching in Strings},
  journal   = {{SIAM} J. Comput.},
  volume    = {6},
  number    = {2},
  pages     = {323--350},
  year      = {1977},
  doi       = {10.1137/0206024},
}
```
In case that the reference is not available in the [dbpl], try to emulate the BibTeX style as close a possible and mark the entry an *nondblp*.
```bibtex
@article{NODBLP:journals/siamcomp/KnuthMP77,
  author    = {Donald E. Knuth and
               James H. Morris Jr. and
               Vaughan R. Pratt},
  title     = {Fast Pattern Matching in Strings},
  journal   = {{SIAM} J. Comput.},
  volume    = {6},
  number    = {2},
  pages     = {323--350},
  year      = {1977},
  doi       = {10.1137/0206024},
}
```

[Doxygen]: https://www.doxygen.nl/
[automatic link generation]: https://www.doxygen.nl/manual/autolink.html#linkfunc
[coding guidelines]: coding_guidelines.md
[dblp]: https://dblp.uni-trier.de/
[group]: https://www.doxygen.nl/manual/grouping.html
[literature.bib]: literature.bib
[main.dox]: main.dox
