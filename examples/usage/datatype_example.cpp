#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/p2p/send.hpp"

void print_datatype(MPI_Datatype datatype) {
    int num_integers, num_addresses, num_datatypes, combiner;
    MPI_Type_get_envelope(datatype, &num_integers, &num_addresses, &num_datatypes, &combiner);
    switch (combiner) {
        case MPI_COMBINER_NAMED: {
            char name[MPI_MAX_OBJECT_NAME];
            int  str_len;
            MPI_Type_get_name(datatype, name, &str_len);
            std::cout << "MPI_COMBINER_NAMED: " << name << std::endl;
            break;
        }
        case MPI_COMBINER_STRUCT: {
            std::cout << "MPI_COMBINER_STRUCT: " << std::endl;
            std::vector<int>          integers(static_cast<size_t>(num_integers));
            std::vector<MPI_Aint>     addresses(static_cast<size_t>(num_addresses));
            std::vector<MPI_Datatype> datatypes(static_cast<size_t>(num_datatypes));
            MPI_Type_get_contents(
                datatype,
                num_integers,
                num_addresses,
                num_datatypes,
                integers.data(),
                addresses.data(),
                datatypes.data()
            );
            for (size_t i = 0; i < static_cast<size_t>(integers[0]); i++) {
                std::cout << "blocklength=" << integers[i + 1] << ", displacement=" << addresses[i];
                print_datatype(datatypes[i]);
            }
            break;
        }
        case MPI_COMBINER_CONTIGUOUS: {
            std::cout << "MPI_COMBINER_CONTIGUOUS: " << std::endl;
            int          count;
            MPI_Datatype t;
            MPI_Type_get_contents(datatype, num_integers, num_addresses, num_datatypes, &count, nullptr, &t);
            std::cout << "count=" << count << " ";
            print_datatype(t);
            break;
        }
        default:
            std::cout << "Unknown combiner" << std::endl;
    }
}

int MPI_Type_commit(MPI_Datatype* type) {
    std::cout << "MPI_Type_commit" << std::endl;
    print_datatype(*type);
    return PMPI_Type_commit(type);
}
int MPI_Type_free(MPI_Datatype* type) {
    std::cout << "MPI_Type_free" << std::endl;
    // printdatatype(*type);
    return PMPI_Type_free(type);
}

template <typename T1, typename T2>
struct MyPair {
    T1 first;
    T2 second;
};
struct Foo {
    int                  a;
    double               b;
    MyPair<float, float> p;
};
// Explicit specialization of mpi_type_traits for MyPair using the automic type constructor which uses reflection to
// create the type using MPI_Type_create_struct.
namespace kamping {
template <>
struct mpi_type_traits<std::tuple<int, float, double>> : struct_type<std::tuple<int, float, double>> {};
} // namespace kamping

int main() {
    using namespace kamping;
    kamping::Environment e;
    Communicator         comm;
    std::cout << std::boolalpha;
    MyPair<double, bool> p = {1.0, true};
    comm.send(destination(rank::null), send_buf(p));
    std::tuple<int, float, double> t = {1, 2.0f, 3.0};
    comm.send(destination(rank::null), send_buf(t));
    // using a pair directly does not work because std::pair is not a trivially copyable type, so the automatic byte
    // serialization is not enabled. This could be fixed by providing a specialization of mpi_type_traits for
    // std::pair<double, bool>. std::pair<double, bool>        p2 = {2.0, false}; comm.send(destination(rank::null),
    // send_buf(p2));
    Foo f = {1, 2.0, {3.0, 4.0}};
    comm.send(destination(rank::null), send_buf(f));
}
