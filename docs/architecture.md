# Passing Parameters to Wrapped MPI Functions

A central concept of KaMPIng is the DataBuffer. It encapsulates all data (or raw memory) which is passed to the underlying MPI call such as ...
A call to a named parameter instantiates a DataBuffer.
Let us examine DataBuffers in action. In the following we see the type signature of `MPI_Alltoallv` as defined in MPI-4.0 which takes a `sendbuf` of variable
size on each PE i and sends messages of size sendcounts[j] from PE i to PE j where the messages are received into `recvbuf`.
MPI requires us to provide additional information such as the number of elements to receive (`recvcounts[]`) or a possible displacements (`sdispls`, `rdispls`) etc.
```cpp
MPI_Alltoallv(const void *sendbuf, const int sendcounts[],
              const int sdispls[], MPI_Datatype sendtype, void *recvbuf,
              const int recvcounts[], const int rdispls[],
              MPI_Datatype recvtype, MPI_Comm comm)
```

In KaMPIng all such information is encapsulated into a DataBuffer. Furthermore, the user does not need to provide all information required by the MPI function signature.
Instead KaMPIng can compute information not provided by the user in many cases. For `Alltoallv` for example, the user is free to omit the `recvcounts` parameter as these can be deduced from the `sendcounts` given on each PE at the cost of additional communication (one call to `MPI_Alltoall`) provided that `sendtype` and `recvtype` refer to the same type.

Hence, the call to `MPI_Alltoallv` may look like:
```cpp
  std::vector<double> send_buffer(comm.size(), 42);
  std::vector<int> counts(comm.size(), 1);
  std::vector<int> recv_buffer;
  std::vector<int> recv_counts;
  {
    // case A: received data is stored in recv_buffer
    recv_buffer.reserve(\*sufficiently large size*\);
    comm.alltoallv(send_buf(data), send_counts(counts), recv_buf_out(recv_buffer), recv_counts_out();
  }
  {
    // case B: received data (and the associated recv_counts) are stored in a std::tuple like result object and be retrieved from there.
    recv_counts.reserve(comm.size());
    auto result = comm.alltoallv(send_buf(data),
                                 send_counts(counts),
                                 recv_buf_out(),
                                 recv_counts_out());
  }
```

Here, `send_buf(data)`, `send_counts(counts)`, `recv_buf_out(...)`, `recv_counts_out()` construct DataBuffers encapsulating the send buffer, send counts, recv buffer and receive counts, respectively.
After having established the general usage of DataBuffers, we now have a closer look on their internals and how they determine the behaviour of wrapped MPI calls.

A DataBuffer has multiple (orthogonal) properties with which the caller can control the behaviour of this specific parameter inside the wrapped MPI call:
- type: <in, out, (inout)>
- ownership: <owning, non-owning>
- resize-policy: <no-resize, grow-only, resize-to-fit>

## Parameter Type

Let us start with the type property. Named parameters like `send_counts(...)`, `recv_counts(...)` , ... generate DataBuffers with type property *in*.
This signals that they wrap memory containing meaningful *input* data which can be used by the MPI call directly, e.g. the receive counts wrapped by `recv_counts(...)`.
The corresponding `*_out` named parameters are *output* parameters.
They do not contain meaningful data yet will be filled with data during the MPI call itself as for `recv_buf_out()` or beforehand by KaMPIng for parameters such as `recv_counts_out()` etc.

To sum this up there are three ways to pass parameters to KaMPIng.
1. **in** parameter: the caller directly provides the parameter required by the MPI call such as the sendbuf, sendcounts, recvcounts etc.
2. **out** parameter: the caller does not know the parameter, asks kamping to compute/infer the parameter and return the value to the user.
3. **omitted** parameter: the caller does not know the parameter, asks kamping to internally compute/infer the parameter but is not interested in its value. Therefore it is discared once the wrapped MPI call has completed.

**Note**: Depending on the wrapped MPI call some parameters have to be provided by the user (as **in** parameters) such as `send_buf` in almost all MPI functions or `send_counts` in `MPI_Alltoallv` as KaMPIng cannot infer these values.

Hence, passing an `*_out` parameter signals KaMPIng, that the user does not know this parameter but is interested in its value and wants to obtain the data computed during the wrapped MPI call afterwards.
The computed/infered data of an out parameter is either returned by value via an result object or written directly into a memory location specified by the caller.
The parameter `recv_buf_out()` is a special case as KaMPIng assume that one is always interest in the received data.
Therefore, even if `recv_buf_out()` is not given, one will obtain the received data via a result object.

TODO briefly mention inout

## Ownership
This property simply determines whether a DataBuffer owns its underlying data following the corresponding C++ ownership concept and is most important for out parameters.
A DataBuffer *owns* its underlying memory/container if its has been passed to the named parameter as an lvalue as in `send_buf(data)` or `recv_buf_out(recv_buffer)`.
Otherwise a DataBuffer is *non-owning*, e.g. for `recv_buf_out()`, `recv_buf_out(std::move(recv_buffer))`.
Note that a named (out) parameter without associated underlying memory/container (such as `recv_buf_out()`) signifies that the caller asks KaMPIng to allocate the memory to hold the computed/infered values.
The user can further specify the container/allocator to use for the memory allocation. TODO add reference

The ownership of an out parameter is important as it specifies how the computed data will be returned to the caller.
Owning out parameters are moved to a result object which is returned by value.
Non-owning out parameters write their data directly to their associated memory location and will therefore not be part of the result object.

```cpp
  // owning out parameters:
  auto result = comm.alltoallv(send_buf(data), send_counts(counts), recv_buf_out(), recv_counts_out());
  auto recv_buffer = result.extract_recv_buffer();
  auto recv_counts = result.extract_recv_counts();
  // or retrieve via structured bindings, i.e. auto [recv_buffer, recv_counts ] = comm.alltoallv(...);

  // non-owning out parameters:
  comm.alltoallv(send_buf(data), send_counts(counts), recv_buf_out(recv_buffer), recv_counts_out(recv_counts));
```

TODO refer to documentation about Result object (or write here?)

## Resize Policy
Resize policies are only important for out parameters. They control if/how the underlying memory/container are resized if the provided memory is not large enough to hold the computed values.
- `no-resize`: the underlying container are not resized and the caller has to ensure that it is large enough to hold all data.
- `grow-only`: KaMPIng will resize the underlying container if its initial size is too small. However, a container's size will never be reduced.
- `resize-to-fit`: KaMPIng will resize the underlying container to have exactly the size required to hold all data.

The default resize policy is `no-resize` (except for empty named parameters).
For `grow-only` or `resize-to-fit` KaMPIng requires the container to posess a member function `resize(unsigned integer)` taking an (unsigned) integer specify the requested size.
Note that KaMPIng can resize `recv_buf` only if the corresponding `recv_type` matches a `C++` type.

```cpp
  std::vector<T>    recv_buffer;                // will be resized by KaMPIng
  std::vector<int>  recv_counts(comm.size());   // has to be large enough to hold all recv counts
  comm.alltoallv(send_buf(data), send_counts(counts),
                 recv_buf_out<grow-only>(recv_buffer),
                 recv_counts_out<no-resize>(recv_counts));
```


