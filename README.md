# KaMPI.ng: Karlsruhe MPI next generation

![KaMPI.ng logo](./docs/images/logo.svg)

This is KaMPI.ng, an MPI wrapper which makes using MPI feel like real C++

KaMPI.ng is developed at the [Algorithm Engineering Group](https://algo2.iti.kit.edu) at Karlsruhe Institute of Technology.

## Goals
   - ban `man MPI_*` from your command line history
   - zero-overhead whenever possible
     - if not possible this must be clear to the user
   - easy to use when simple communication is required, but powerful enough for finely tuned behavior
     - useful defaults
     - compile-time checks for incorrect usage
   - accumulate knowledge/algorithms of our group in a single place
   - ensure that the library outlives multiple generations of our group

## Platform
   - compiles with GCC, Clang and ICC on SuperMUC-NG, bwUniCluster, HoreKa
   - C++17
     - we do not want to rely on C++20 features, because Intel does not support it yet
   - easy inclusion into other projects by using modern CMake
   
## LICENSE

KaMPIng is released under the GNU Lesser General Public License. See [COPYING](COPYING) and [COPYING.LESSER](COPYING.LESSER) for details
