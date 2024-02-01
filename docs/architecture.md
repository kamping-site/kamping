# Passing Parameters in KaMPIng

A central concept of KaMPIng is the named parameters approach allowing the caller to name and pass parameters in arbitrary order and even omit certain parameters.
KaMPIng allows the user a fine-grained control over the parameter set which will be explained in greater detail in this document.

Internally, this functionality is empowered by the DataBuffer class, which encapsulates all data (or raw memory) which is passed via a named parameter to KaMPIng.
A call to a named parameter instantiates a DataBuffer object.
In this document we will have a closer look on the relevant properties of the DataBuffer and how it can be used to control the wrapped MPI call in conjuction with named parameters.

To illustrate the named parameter approach, we first look at the function signature of `MPI_Alltoallv` as defined in MPI-4.0 which takes a `sendbuf` of variable
size on each PE i and sends messages of size sendcounts[j] from PE i to PE j where the messages are received into `recvbuf`.
Apart from these three parameters, MPI requires additional information such as the number of elements to receive (`recvcounts[]`) or a possible displacements (`sdispls`, `rdispls`) etc.
```cpp
MPI_Alltoallv(const void *sendbuf, const int sendcounts[],
              const int sdispls[], MPI_Datatype sendtype, void *recvbuf,
              const int recvcounts[], const int rdispls[],
              MPI_Datatype recvtype, MPI_Comm comm)
```

In KaMPIng all these parameters to an MPI function are represented by ***named parameters***:
- send_buf(...), send_counts(...)/send_counts_out(...), recv_counts()/recv_counts_out(), ... TODO refer to complet list

These named parameters either serve as *input (in)* or *output (out)* parameters.
Via a named in parameter the caller can provide input data to the wrapped MPI call such as `send_buf(buf)` or `send_counts(counts)` with buf and counts accomodating the data to be sent and send counts, respectively.
Using named out parameter the caller ask KaMPIng to internally compute/infer this parameter and output its value. Named output parameters are created via the respective `*_out()` suffix.
The data requested via out parameters is then either directly written to a memory location passed within the named parameter call or returned in a `std::tuple`-like *result* object.

One special case it the receive buffer. Although being an out parameter, it does not need to be explicityl given as KaMPIng assumes that a always wants to obtain this buffer.

```cpp
  std::vector<T> data = ...;          // initialize data to send
  std::vector<int> send_counts = ...; // initialize send counts
  
  auto recv_buf = comm.alltoallv(send_buf(data), send_counts(counts));
```

MPI parameters are encapsulated into a DataBuffer. Furthermore, the user does not need to provide all information required by the MPI function signature.
Instead KaMPIng can compute information not provided by the user in many cases. For example, for `Alltoallv`, the user is free to omit all parameters apart from `sendbuf` and `sendcounts` as all other parameters can be infered at the cost of additional local computation or communication (one call to `MPI_Alltoall`) provided that `sendtype` and `recvtype` refer to the same type.

Hence, the call to `MPI_Alltoallv` may look like:
```cpp
  std::vector<T> data = ...;          // initialize data to send
  std::vector<int> send_counts = ...; // initialize send counts
  
  auto recv_buf = comm.alltoallv(send_buf(data), send_counts(counts));
```

Here, `send_buf(data)` and `send_counts(counts)` construct DataBuffers encapsulating the data to send and the send counts. The received values are returned by value.

However, the user might be interested in the `recv counts` parameter and wants KaMPIng to also output these values resulting in:
```cpp
  std::vector<T> data = ...;          // initialize data to send
  std::vector<int> send_counts = ...; // initialize send counts
  
  auto result = comm.alltoallv(send_buf(data), send_counts(counts), recv_counts_out());
  std::vector<T>   recv_buf    = result.extract_recv_counts();
  std::vector<int> recv_counts = result.extract_recv_buf();
  // or via structured bindings as auto [recv_buf, recv_counts] = comm.alltoallv(...);
```
Now, KaMPIng returns a `std::tuple` like result object containing the received elements and the recv counts.
The actual data can be explicitly extracted
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


