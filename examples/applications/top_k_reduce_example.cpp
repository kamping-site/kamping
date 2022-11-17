#include <algorithm>
#include <cstddef>
#include <optional>
#include <ostream>

#include <mpi.h>

#include "kamping/collectives/reduce.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"

template <size_t K, typename ValueType>
class TopK {
public:
    explicit TopK() {}
    ValueType& operator[](size_t i) {
        return elements[i];
    }
    ValueType const& operator[](size_t i) const {
        return elements[i];
    }

private:
    std::array<ValueType, K> elements;
};

template <size_t K, typename ValueType>
TopK<K, ValueType> merge(TopK<K, ValueType> const& lhs, TopK<K, ValueType> const& rhs) {
    size_t          lhs_current = 0;
    size_t          rhs_current = 0;
    TopK<K, size_t> merged;
    for (size_t i = 0; i < K; i++) {
        if (lhs[lhs_current] < rhs[rhs_current]) {
            merged[i] = lhs[lhs_current];
            lhs_current++;
        } else {
            merged[i] = rhs[rhs_current];
            rhs_current++;
        }
    }
    return merged;
}

template <size_t K, typename ValueType>
std::ostream& operator<<(std::ostream& os, TopK<K, ValueType> const& top_k) {
    os << "TopK(";
    for (size_t i = 0; i < K; ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << top_k[i];
    }
    os << ")";
    return os;
}

template <size_t K, typename ValueType>
auto kamping_top_k(TopK<K, ValueType> const& local_top_k, kamping::Communicator& comm) {
    using namespace kamping;
    auto result = comm.reduce(send_buf(local_top_k), op(merge<K, size_t>, ops::commutative)).extract_recv_buffer();
    if (comm.is_root()) {
        return std::make_optional(result[0]);
    } else {
        return std::optional<TopK<K, ValueType>>{};
    }
}

template <size_t K, typename ValueType>
std::optional<TopK<K, ValueType>> mpi_top_k(TopK<K, ValueType> const& local_top_k, MPI_Comm comm) {
    // create a custom datatype
    MPI_Datatype topK_type;
    // to make it truly generic we rely on KaMPIng
    MPI_Type_contiguous(K, kamping::mpi_datatype<ValueType>(), &topK_type);
    MPI_Type_commit(&topK_type);

    // create a custom reduce operation
    MPI_Op             topK_merge_op;
    MPI_User_function* merge_op = [](void* invec, void* inoutvec, int* len, MPI_Datatype*) {
        TopK<K, ValueType>* invec_    = static_cast<TopK<K, ValueType>*>(invec);
        TopK<K, ValueType>* inoutvec_ = static_cast<TopK<K, ValueType>*>(inoutvec);
        std::transform(invec_, invec_ + *len, inoutvec_, inoutvec_, merge<K, size_t>);
    };
    MPI_Op_create(merge_op, true, &topK_merge_op);

    // the actual MPI call
    TopK<K, ValueType> global_top_k;
    MPI_Reduce(&local_top_k, &global_top_k, 1, topK_type, topK_merge_op, 0, comm);

    // cleanup
    MPI_Op_free(&topK_merge_op);
    MPI_Type_free(&topK_type);

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        return std::make_optional(std::move(global_top_k));
    } else {
        return std::optional<TopK<K, ValueType>>{};
    }
}

int main(int argc, char* argv[]) {
    namespace kmp = kamping;
    kmp::Environment  env(argc, argv);
    kmp::Communicator comm;

    constexpr size_t K = 3;
    TopK<K, size_t>  input;
    for (size_t i = 0; i < K; ++i) {
        input[i] = comm.rank() + i * comm.size();
    }
    std::cout << "[R" << comm.rank() << "] local_input=" << input << std::endl;

    auto kamping_result = kamping_top_k(input, comm);
    if (comm.is_root()) {
        std::cout << "global_result_kamping=" << kamping_result.value() << std::endl;
    }

    auto mpi_result = mpi_top_k(input, MPI_COMM_WORLD);
    int  rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "global_result_mpi=" << mpi_result.value() << std::endl;
    }
}
