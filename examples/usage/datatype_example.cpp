#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/p2p/send.hpp"

void printdatatype(MPI_Datatype datatype) {
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
                printdatatype(datatypes[i]);
            }
            break;
        }
        case MPI_COMBINER_CONTIGUOUS: {
            std::cout << "MPI_COMBINER_CONTIGUOUS: " << std::endl;
            int          count;
            MPI_Datatype t;
            MPI_Type_get_contents(datatype, num_integers, num_addresses, num_datatypes, &count, nullptr, &t);
            std::cout << "count=" << count << " ";
            printdatatype(t);
            break;
        }
        default:
            std::cout << "Unknown combiner" << std::endl;
    }
}

int MPI_Type_commit(MPI_Datatype* type) {
    std::cout << "MPI_Type_commit" << std::endl;
    printdatatype(*type);
    return PMPI_Type_commit(type);
}
int MPI_Type_free(MPI_Datatype* type) {
    std::cout << "MPI_Type_free" << std::endl;
    // printdatatype(*type);
    return PMPI_Type_free(type);
}

struct Foo {
    int                     a;
    double                  b;
    std::pair<float, float> p;
};

int main() {
    using namespace kamping;

    kamping::Environment e;
    Communicator         comm;
    std::cout << std::boolalpha;
    std::pair<double, bool> p = {1.0, true};
    comm.send(destination(rank::null), send_buf(p));
    std::tuple<int, float, double> t = {1, 2.0f, 3.0};
    comm.send(destination(rank::null), send_buf(t));
    // std::pair<double, bool>        p2 = {2.0, false};
    // comm.send(destination(rank::null), send_buf(p2));
    Foo f = {1, 2.0, {3.0, 4.0}};
    comm.send(destination(rank::null), send_buf(f));
}
