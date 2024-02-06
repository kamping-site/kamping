# Passing Parameters in KaMPIng

A core feature of KaMPIng is the named parameters concept allowing the caller to name and pass parameters in arbitrary order and even omit certain parameters.
With this approach, KaMPIng allows the user a fine-grained control over the parameter set which will be explained in greater detail in this document.

To illustrate the concept, we first look at the function signature of `MPI_Alltoallv` as defined in MPI-4.0 which takes a `sendbuf` of variable
size on each PE i and sends messages of size `sendcounts[j]` from PE i to PE j where the messages are received into `recvbuf`.
Apart from these three parameters, MPI requires additional information such as the number of elements to receive (`recvcounts[]`) or a possible displacements (`sdispls`, `rdispls`) etc.
```cpp
MPI_Alltoallv(const void *sendbuf, const int sendcounts[],
              const int sdispls[], MPI_Datatype sendtype, void *recvbuf,
              const int recvcounts[], const int rdispls[],
              MPI_Datatype recvtype, MPI_Comm comm)
```

In KaMPIng all these parameters to an MPI function are represented by ***named parameters***:
- send_buf(...), send_counts(...)/send_counts_out(...), recv_counts()/recv_counts_out(), ... TODO refer to complet list

KaMPIng realizes named parameters as factory functions which construct parameter objects inplace, but it's easier to think of this concept in terms of `name(buffer)`.

The named parameters either serve as *in(put)* or *out(put)* parameters.
Via a named *in* parameter the caller can provide input data to the wrapped MPI call such as for `send_buf(buf)` or `send_counts(counts)` with `buf` and `counts` accomodating the data to be sent and send counts, respectively.
Using named out parameter the caller asks KaMPIng to internally compute/infer this parameter and output its value. Named output parameters are created via the respective `*_out()` suffix.
The data requested via out parameters is then either directly written to a memory location passed within the named parameter call or returned in a `std::tuple`-like *result* object (depending on the ownership property, [see](named_parameters.md#Ownership)).

One special case it the receive buffer. Although being an out parameter, it does not need to be explicitly given as KaMPIng assumes that a caller always wants to obtain this buffer.

See the following examples for an illustration of the different options

```cpp
  std::vector<T> data = ...;          // initialize data to send
  std::vector<int> send_counts = ...; // initialize send counts
  
  {
    // only (implicit) out parameter: recv buffer
    auto recv_buf = comm.alltoallv(send_buf(data), send_counts(counts));
  }
  {
    // out parameters: recv counts and recv buffer
    auto result = comm.alltoallv(send_buf(data), send_counts(counts), recv_counts_out());
    std::vector<T>   recv_buf    = result.extract_recv_counts();
    std::vector<int> recv_counts = result.extract_recv_buf();
    // or via structured bindings as auto [recv_buf, recv_counts] = comm.alltoallv(...);
  }
  {
    // out parameter data is written to a specific memory location: recv counts and recv buffer
    std::vector<int> recv_counts = ...; // allocate enough memory to hold recv counts
    std::vector<T>   recv_buffer = ...; // allocate enough memory to hold recv elements
    comm.alltoallv(send_buf(data),
                   send_counts(counts),
                   recv_counts_out(recv_counts),
                   recv_buf_out(recv_buffer));
  }
  {
    // out parameter data is written to a specific memory location: recv counts and recv buffer
    // potential resizing is done by KaMPIng
    std::vector<int> recv_counts;
    std::vector<T>   recv_buffer;
    comm.alltoallv(send_buf(data),
                   send_counts(counts),
                   recv_counts_out<resize-to-fit>(recv_counts),
                   recv_buf_out<resize-to-fit>(recv_buffer));
  }
```

To summarize, there are three ways to pass parameters to KaMPIng:
1. **in** parameter: the caller directly provides the parameter required by the MPI call such as the sendbuf, sendcounts, recvcounts etc.
2. **out** parameter: the caller does not know the parameter, asks KaMPIng to compute/infer the parameter and return the value to the caller.
3. **omitted** parameter: the caller does not know the parameter, asks KaMPIng to internally compute/infer the parameter but is not interested in its value. Therefore it is discared once the wrapped MPI call has completed.

**Note**: Depending on the wrapped MPI call some parameters have to be provided by the user (as **in** parameters) such as `send_buf` in almost all MPI functions or `send_counts` in `MPI_Alltoallv` as KaMPIng cannot infer these values.
KaMPIng will show an error at compile time if any required parameter is missing.


Internally, a call to a named parameter factory function (`send_buf(...), send_counts(...), ...`) instantiates an object of the *DataBuffer* class encapsulating
all passed data/memory and user request regarding potential output strategies.

## Named Parameters and DataBuffers
To fully benefit from KaMPIng's parameter flexibility we briefly present the most important concepts of the DataBuffer class.
A DataBuffer wraps an underlying container. KaMPIng requires this container to have contiguous memory, expose its `value_type` and has a member function `data()` returning a pointer to the start of its memory and a member function `size()`.

Futhermore, DataBuffer has multiple (orthogonal) properties with which the caller can control the behavior of corresponding parameter inside the wrapped MPI call:
- type: <in, out, (inout)>
- ownership: <owning, non-owning>
- resize-policy: <no-resize, grow-only, resize-to-fit>

### Parameter Type

Let us start with the type property. Named parameters like `send_counts(...)`, `recv_counts(...)` , ... generate DataBuffers with type property *in*.
This signals that they wrap a container with meaningful *input* data which can be used by the MPI call directly, e.g. the send counts wrapped by `send_counts(...)`.

The corresponding `*_out` named parameters instantiates DataBuffer objects with type property *out*.
They do not contain meaningful data yet and will be filled with values during the wrapped MPI call.

DataBuffers with type *inout* correspond to named parameters like `sendrecv_buf(...)` used in `MPI_Bcast` where data is sent on one root rank (in parameter) and received (out parameter) on all other ranks.

### Ownership
This property simply determines whether a DataBuffer owns its underlying data following the corresponding C++ ownership concept and is most important for out parameters.
A DataBuffer *references* (*non*-owns) its container if the latter has been passed to the named parameter as an lvalue as in `send_buf(data)` or `recv_buf_out(recv_buffer)`.
Otherwise a DataBuffer is *owning*, e.g. for `recv_buf_out()`, `recv_buf_out(std::move(recv_buffer))`.

Note that a named (out) parameter without associated underlying container (such as `recv_buf_out()`) implies that the caller asks KaMPIng to allocate the memory to hold the computed/infered values.
This results in an owning container, which is allocated by KaMPIng and ownership is transferred to the caller upon return.

Furthermore, the ownership of an out parameter is important as it specifies how the computed data will be returned to the caller.
Owning out parameters are moved to a result object which is returned by value.
Non-owning out parameters write their data directly to their associated underlying container.
Therefore, the DataBuffer object corresponding to this named parameter will not be part of the result object.
The following code provides some example for owning and non-owning out parameters:

```cpp
  // owning out parameters:
  {
    auto result = comm.alltoallv(send_buf(data),
                                 send_counts(counts),
                                 recv_buf_out(),
                                 recv_counts_out());
    auto recv_buffer = result.extract_recv_buffer();
    auto recv_counts = result.extract_recv_counts();
    // auto send_counts = result.extract_send_counts() // compilation error: this cannot be extracted
                                                       // since it is not specified as an out parameter.
  }
  
  // owning out parameters with structured bindings:
  {
    auto [recv_buffer, recv_counts] = comm.alltoallv(send_buf(data),
                                                     send_counts(counts),
                                                     recv_buf_out(),
                                                     recv_counts_out());
  }

  // non-owning out parameters:
  {
    std::vector<T> recv_buffer = ...; // allocate sufficient memory
    std::vector<int> recv_counts;     // allocate sufficient memory
    comm.alltoallv(send_buf(data),
                   send_counts(counts),
                   recv_buf_out(recv_buffer),
                   recv_counts_out(recv_counts));
  }
```

TODO refer to documentation about Result object.

### Resize Policy
Resize policies are only important for out parameters. They control if/how the underlying memory/container are resized if the provided memory is not large enough to hold the computed values.
- `no-resize`: the underlying container are not resized and the caller has to ensure that it is large enough to hold all data.
- `grow-only`: KaMPIng will resize the underlying container if its initial size is too small. However, a container's size will never be reduced.
- `resize-to-fit`: KaMPIng will resize the underlying container to have exactly the size required to hold all data.

The default resize policy is `no-resize` (except for empty named parameters).
For `grow-only` or `resize-to-fit` KaMPIng requires the container to posess a member function `resize(unsigned integer)` taking an (unsigned) integer specify the requested size.
Note that KaMPIng can resize `recv_buf` only if the corresponding `recv_type` matches a `C++` type. Otherwise the user must ensure that the container is large enough and only `no-resize` is allowed.

```cpp
  std::vector<T>    recv_buffer;                
  std::vector<int>  recv_counts(comm.size());   
  
  comm.alltoallv(send_buf(data), send_counts(counts),
                 recv_buf_out<grow-only>(recv_buffer),    // will be resized by KaMPIng
                 recv_counts_out<no-resize>(recv_counts)  // has to be large enough to hold all recv counts
                );
```
