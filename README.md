# KaMPIng: Karlsruhe MPI next generation :camping:

![KaMPIng logo](./docs/images/logo.svg)

This is KaMPIng [kampɪŋ], a (near) zero-overhead MPI wrapper for modern C++.

It covers the whole range of abstraction levels from low-level MPI calls to
convenient STL-style bindings, where most parameters are inferred from a small
subset of the full parameter set. This allows for both rapid prototyping and
fine-tuning of distributed code with predictable runtime behavior and memory
management.

Using template-metaprogramming, only code paths required for computing missing
parameters are generated at compile time, which results in (near) zero-overhead
bindings.

KaMPIng is developed at the [Algorithm Engineering
Group](https://algo2.iti.kit.edu/english/index.php) at Karlsruhe Institute of
Technology.

## Features :sparkles:
### Named Parameters :speech_balloon:
Using plain MPI, operations like `MPI_Allgatherv` often lead to verbose and error-prone boilerplate code:

``` c++
std::vector<T> v = ...; // Fill with data
int size;
MPI_Comm_size(comm, &size);
int n = static_cast<int>(v.size());
std::vector<int> rc(size), rd(size);
MPI_Allgather(&n, 1, MPI_INT, rc.data(), 1, MPI_INT, comm);
std::exclusive_scan(rc.begin(), rc.end(), rd.begin(), 0);
int n_glob = rc.back() + rd.back();
std::vector<T> v_glob(v_global_size);
MPI_Allgatherv(v.data(), v_size, MPI_TYPE, v_glob.data(), rc.data(), rd.data(), MPI_TYPE, comm);

```
In contrast, KaMPIng introduces a streamlined syntax with inspiration from Python's named parameters. For example, the `allgatherv` operation becomes more intuitive and concise:

```c++
std::vector<T> v_glob = comm.allgatherv(send_buf(v));
```
Empowered by named parameters, KaMPIng allows users to name and pass parameters in arbitrary order, computing default values only for the missing ones. This not only improves readability but also streamlines the code, providing a user-friendly and efficient way of writing MPI applications.

### Controlling memory allocation :floppy_disk:
KaMPIng's *resize policies* allow for fine-grained control over when allocation happens.
``` c++
// easy to use with sane defaults
std::vector<int> v = comm.recv<int>(source(kamping::rank::any));

// flexible memory control
std::vector<int> v_out;
v_out.resize(enough_memory_to_fit);
comm.recv<int>(recv_buf<kamping::no_resize>(v_out), recv_count(i_know_already_know_that), source(kamping::rank::any));
```

### STL support :books:
- KaMPIng works with everything that is a `std::contiguous_range`, everywhere.
- Builtin C++ types are automatically mapped to their corresponding MPI types. 
- All internally used containers can be altered via template parameters.
### Expandability :jigsaw:
- Don't like the performance of your MPI implementation's reduce algorithm? Just override it using our plugin architecture.
- Add additional functionality to communicator objects, without altering any application code.
- Easy to integrate with existing MPI code.
- Flexible core library for a new toolbox :toolbox: of distributed datastructures and algorithms

### And much more ... :arrow_upper_right:
- Easy non-blocking communication via request pools.
- Compile time and runtime error checking (which can be completely deactivated).
- Collective hierarchical timers to speed up your evaluation workflow.
- ...

Dive into the documentation or tests to find out more ...

### (Near) zero overhead - for development and performance :chart_with_upwards_trend:
Using template-metaprogramming, KaMPIng only generates the code paths required for computing missing parameters. 
The following shows a complete implementation of distributed sample sort with KaMPIng. 

```c++
void sort(MPI_Comm comm_, std::vector<T>& data, size_t seed) {
    Communicator<> comm(comm_);
    size_t const   oversampling_ratio = 16 * static_cast<size_t>(std::log2(comm.size())) + 1;
    std::vector<T> local_samples(oversampling_ratio);
    std::sample(data.begin(), data.end(), local_samples.begin(), oversampling_ratio, std::mt19937{seed});
    auto global_samples = comm.allgather(send_buf(local_samples)).extract_recv_buffer();
    std::sort(global_samples.begin(), global_samples.end());
    for (size_t i = 0; i < comm.size() - 1; i++) {
        global_samples[i] = global_samples[oversampling_ratio * (i + 1)];
    }
    global_samples.resize(num_splitters);
    std::vector<std::vector<T>> buckets(global_samples.size() + 1);
    for (auto& element: data) {
        auto const bound = std::upper_bound(global_samples.begin(), global_samples.end(), element);
        buckets[static_cast<size_t>(bound - global_samples.begin())].push_back(element);
    }
    data.clear();
    std::vector<int> scounts;
    for (auto& bucket: buckets) {
        data.insert(data.end(), bucket.begin(), bucket.end());
        scounts.push_back(static_cast<int>(bucket.size()));
    }
    data = comm.alltoallv(send_buf(data), send_counts(scounts)).extract_recv_buffer();
    std::sort(data.begin(), data.end());
}
```
While a lot more concise than the [./examples/applications/sample-sort/mpi.hpp]((verbose) plain MPI implementation), it introduces no additional run time overhead.

## Platform :desktop_computer:
- intensively tested with GCC and Clang and OpenMPI
- requires a C++17 ready compiler
- easy integration into other projects using modern CMake
   
## LICENSE
KaMPIng is released under the GNU Lesser General Public License. See [COPYING](COPYING) and [COPYING.LESSER](COPYING.LESSER) for details
