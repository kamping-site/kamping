Using the Communicator {#user_guidelines_communicator}
============
KaMPI.ng is an MPI wrapper. It provides access to the most used MPI calls using a simple C++ interface.
This guide assumes basic knowledge of MPI, as we use a lot of MPI terminology.
We can find most of KaMPI.ng's functionality in the kamping::Communicator. 
This class not only provide important information regarding the used \c MPI_Communicator and also access to the wrapped MPI functions.

KaMPI.ng can be used as wrapper for
- collective_communication (TODO GROUP) and
- @TODO point-to-point.

We can assess these functions as follows:
```cpp
using namespace kmp = kamping
kmp::Communicator comm;
// Do Stuff
comm.mpi_function(Named Parameters);
```
As we can see in the example above, KaMPI.ng makes use of named parameters (TODO GROUP).
In the following sections, we describe the different parameters provided by KaMPI.ng and KaMPI.ng's behavior depending on the passed parameters.

# Default Values
The main idea behind the kamping::Communicator class is to provide access to MPI operations for an \c MPI_Communicator.
The communicator, its size, the rank of each PE, a default root, and possibly more information are provided by a communicator object.
All default values provided by the kamping::Communicator can be found in its documentation page.
The default values can obviously be changed and overwritten on a per-operation basis.

# Named Parameters
The naming scheme of the parameters is similar to the names used in MPI.
We can distinguish two types of buffers in KaMPI.ng.
The first type takes a \c Container as parameter.
This container has to cover a consecutive part of memory.
The following containers are part of this type of parameter.

- kamping::send_buf() and kamping::recv_buf()
- kamping::send_counts() and kamping::recv_counts()
- kamping::send_displs() and kamping::recv_displs()
- kamping::recv_counts_out()
- kamping::send_displs_out() and kamping::recv_displs_out()

Additionally, there are parameters that do not require an additional container.
These parameters can be used to specify a non-default root for a \c gather operation or the operation for a \c reduce operation.
- kamping::root()
- kamping::op()

## Parameters and Containers
The parameters that require a container also do some memory management, i.e., here KaMPI.ng makes sure that the container has size of the resulting elements after the operation finishes.
For example, if we use an \c std::vector as container for a kamping::recv_buffer(), the container will have size equal to the number of received elements on all PEs.
If there is any PE that does not receive any data, the container will have size 0.

There is, however, one big exception to this rule.
If we do not want KaMPI.ng to manage the memory and do so ourselves (maybe because we have a better understanding of the required memory overall), we can do so.
To this end, we make use of kamping::Span
