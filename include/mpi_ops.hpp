#include <mpi.h>

#include <algorithm>
#include <type_traits>

template <int is_commutative, typename Op, typename T>
struct CustomFunction {
  CustomFunction() {
    MPI_Op_create(CustomFunction<is_commutative, Op, T>::execute,
                  is_commutative, &mpi_op);
  }
  static void execute(void* invec, void* inoutvec, int* len,
                      MPI_Datatype* /*datatype*/) {
    T* invec_ = static_cast<T*>(invec);
    T* inoutvec_ = static_cast<T*>(inoutvec);
    Op op{};
    std::transform(invec_, invec_ + *len, inoutvec_, inoutvec_, op);
  }
  ~CustomFunction() { MPI_Op_free(&mpi_op); }
  MPI_Op& get_mpi_op() { return mpi_op; }
  MPI_Op mpi_op;
  Op op;
};
