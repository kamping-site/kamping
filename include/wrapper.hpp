#pragma once

#include <limits>
#include <vector>

#include <mpi.h>

#include "mpi_ops.hpp"
#include "type_helpers.hpp"

// questions/problems:
// - collectives with asymmetric send/recv numbers who does memory allocations
// - support for serialization out of the box?
// - support for large-size mpi (64-bit bit send_counts) out of the box?
// - point-to-point how to handle different send modes?
// ...
//
// Improvements/Adjustments
// -> memory management: Return Container as template parameter (default is
// std::vector)
//    -> Return Container as class template parameter
// -> (contiguous) Iterators instead of pointers
//
// Requirements:
//  memory allocation * additional return of info about sizes for vectorized
//  variants * additional info about send sizes to input
// Two approaches to solve the above requirements:
// 1. function overloads -> combinatorial explosion in interface
// 2. method chaining + state machine:
//  some calls of gatherv
//    MPIContext ctx;
//    std::vector<..> data = ctx.gatherv(send_buf).set_root(3).call();
//    std::vector<..> data =
//    ctx.gatherv(send_buf).set_root(3).use_recv_counts(counts_buffer).call();
//    ctx.gatherv(send_buf).set_recv_buffer(recv_buffer).call();
//    auto [recv_buf, recv_counts] = ctx.gatherv(send_buf).return_counts().call();
// requires state machine per collective operations

namespace MPIWrapper {

struct Rank {
  int rank;
};

template <template <typename> typename DefaultContainer = std::vector>
class MPIContext {
 public:
  enum class SendMode { normal, buffered, synchronous };
  explicit MPIContext(MPI_Comm comm) : comm_{comm} {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
  }

  // *********************************
  // * reduce
  // *********************************
  template <typename T, typename Op>
  void reduce(T* buffer, std::size_t size, Op op, int root) {
    // boost.mpi like mpi-function detection
    // if constexpr (is_same_v<Op, MPI_Op>) {
    //  MPI_Reduce(&t, 1, get_mpi_type<T>(), op, root, comm_);
    // } else {
    //  CustomFunction fun<true, Op, T>{};
    //  MPI_Reduce(&t, 1, get_mpi_type<T>(), fun.get_mpi_op(), root, comm_);
    // }
  }
  template <typename T, typename Op>
  T reduce(T t, Op op, int root) {
    return T{};
  }
  template <typename T, typename Op>
  void reduce(std::vector<T>& buffer, Op op, int root) {
    // like above
    return T{};
  }
  // *********************************
  // * allreduces (-> similiar to reduce?)
  // *********************************

  // *********************************
  // gathers
  // *********************************
  template <typename T>
  void gather(T t, T* send_buffer, int root) const {}
  template <typename T>
  std::vector<T> gather(T t, int root) const {
    return {};
  }
  template <typename T>
  void gatherv(T* send_buffer, T* recv_buffer, std::size_t size,
               int root) const {}
  template <typename T,
            template <typename> typename Container = DefaultContainer>
  Container<T> gatherv(T* send_buffer, std::size_t size, int root,
                       std::vector<std::size_t>&) const {
    return {};
  }
  template <typename T>
  std::vector<T> gatherv(std::vector<T>& send_buf) const {
    return {};
  }
  // *********************************
  // * allgathers
  // *********************************
  template <typename T>
  void allgather(T t, T* send_buf) const {}
  template <typename T>
  std::vector<T> allgather(T t) const {
    return {};
  }
  template <typename T>
  void allgatherv(T t, T* send_buf) const {}
  template <typename T>
  void allgatherv(T* send_buffer, T* recv_buffer, std::size_t size) const {}
  template <typename T>
  std::vector<T> allgatherv(T* send_buffer, std::size_t size) const {
    return {};
  }
  // *********************************
  // * (sparse) alltoalls
  // *********************************
  template <typename T>
  void all_to_all(T* send_buffer, T* recv_buffer,
                  std::size_t* send_counts) const {}
  template <typename T>
  std::vector<T> all_to_all(T* send_buffer, std::size_t* send_counts) const {}
  template <typename T>
  std::vector<T> all_to_all(std::vector<T>& send_buf) const {}
  template <typename SendMessage, typename RecvMessages, typename Config>
  std::vector<RecvMessages> sparse_all_to_all(
      const std::vector<SendMessage>& send_msgs, const Config& conig) const {
    return {};
  }
  // *********************************
  // * scans
  // *********************************

  // *********************************
  // * broadcast
  //   -> memory management is a bit special as of
  // *********************************
  template <typename T>
  void broadcast(T* send_recv_buffer, std::size_t nb_elems,
                 int root = 0) const {
    broadcast_impl(this, send_recv_buffer, nb_elems, root);
  }
  template <typename T>
  void broadcast(T& t, int root = 0) const {
    broadcast(&t, 1, root);
  }
  template <typename T>
  void broadcast(std::vector<T>& ts, int root = 0) const {
    broadcast(ts.data(), ts.size(), root);
  }
  template <typename T>
  std::vector<T> broadcast_(const std::vector<T>& ts = {}, int root = 0) const {
    return broadcast__impl(comm_, ts, root);
  }
  template <typename T>
  void broadcast(std::vector<T>& ts, std::size_t size, int root = 0) const {
    if (size == 0) {
      // do allocation for ts
      // involves broadcast of root's size
    }
    assert(rank() != root || size <= ts.size());
    broadcast(ts.data(), ts.size(), root);
  }
  // *********************************
  // * barriers
  // *********************************
  void barrier() const {}
  // *********************************
  // * point-to-point
  // *********************************
  template <typename T>
  void send(const T* send_buffer, std::size_t size, int recipient, int tag,
            SendMode mode) {}
  template <typename T>
  void recv(const T* recv_buffer, int sender, int tag) {}
  template <typename T>
  void send(const std::vector<T>& send_buffer, int recipient, int tag,
            SendMode mode) {}
  template <typename T>
  std::vector<T> recv(int sender, int tag) {}
  // missing -> asynchronous variants
  //
  //
  //

  MPI_Comm get_comm() const { return comm_; }
  int rank() const { return rank_; }
  int size() const { return size_; }
  unsigned int rank_unsigned() const {
    return static_cast<unsigned int>(rank_);
  }
  unsigned int size_unsigned() const {
    return static_cast<unsigned int>(size_);
  }

 private:
  void big_type_handling() const {}
  MPI_Comm comm_;
  int rank_;
  int size_;
  static constexpr std::size_t mpi_size_limit = std::numeric_limits<int>::max();
};
}  // namespace MPIWrapper

namespace MPIWrapper::details {

template <typename T>
void broadcast_impl(MPI_Comm comm, T* ts, std::size_t nb_elems, int root = 0) {
  // Additional steps;
  // if (nb_elems > mpi_size_limit) { broadcast_big(...); }
  // if (!is_trivially_copyable_v<T>) { broadcast_serialization(...); }
  MPI_Bcast(ts, static_cast<int>(nb_elems), get_mpi_type<T>(), root, comm);
}
template <typename T>
std::vector<T> broadcast__impl(MPI_Comm comm, const std::vector<T>& ts = {},
                               int root = 0) {
  std::size_t size = broadcast(ts.size(), root);
  std::vector<T> ts_prime;
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  if (rank == root)
    ts_prime = ts;
  else
    ts_prime.resize(size);
  broadcast(ts_prime.data(), size, root);
  return ts_prime;
}

};  // namespace MPIWrapper::details
