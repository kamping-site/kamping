# KaMPIng (Karlsruhe MPI next generation)
> An MPI wrapper which makes using MPI feel like real C++

# Project Goals
## Main Goals
   - ban `man MPI_*` from your command line history
   - zero-overhead whenever possible
     - if not possible this must be clear to the user
   - easy to use when simple communication is required, but powerful enough for finely tuned behavior
   - accumulate knowledge/algorithms of our group in a single place
   - ensure that the library outlives multiple generations of our group
## Platform
   - compiles with GCC, Clang, Intel Compiler on SuperMUC, bwUniCluster, HoReKa
   - C++17
     - we do not want to rely on C++20 features, because Intel does not support it yet
   - use *modern* CMake to enable easy inclusion in projects
## Design Goals 
   - a user should be able to use the library by reading only a minimal amount of documentation
     - code completion and the compiler should give enough assistance
     - compile-time checks for incorrect usage
     - avoid long parameter lists
   - useful defaults
     - if I don't need a tag, I do not have to give one
     - if it is clear that the root is 0, I don't want to care about setting it explicitly
   - allow inclusion of more complex operations/algorithms
     - message queue, sparse-all-to-all, ...
     - architecture must be general enough
