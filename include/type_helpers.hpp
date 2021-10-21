#pragma once
#include <mpi.h>

namespace MPIWrapper {

template <size_t n>
MPI_Datatype mpi_custom_continuous_type() {
    static MPI_Datatype type = MPI_DATATYPE_NULL;
    if (type == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(n, MPI_CHAR, &type);
        MPI_Type_commit(&type);
    }
    return type;
}
template <typename T>
MPI_Datatype get_mpi_type() {
  return mpi_custom_continuous_type<sizeof(T)>();
}


}  // namespace MPIWrapper
