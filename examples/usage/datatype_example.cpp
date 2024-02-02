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
            std::cout << "MPI_COMBINER_STRUCT" << std::endl;
            std::vector<int>          integers(num_integers);
            std::vector<MPI_Aint>     addresses(num_addresses);
            std::vector<MPI_Datatype> datatypes(num_datatypes);
            MPI_Type_get_contents(
                datatype,
                num_integers,
                num_addresses,
                num_datatypes,
                integers.data(),
                addresses.data(),
                datatypes.data()
            );
            for (int i = 0; i < integers[0]; i++) {
                std::cout << "blocklength=" << integers[i + 1] << ", displacement=" << addresses[i];
                printdatatype(datatypes[i]);
            }
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
    printdatatype(*type);
    return PMPI_Type_free(type);
}

int main() {
    using namespace kamping;

    kamping::Environment           e;
    Communicator                   comm;
    std::pair<double, bool>        p  = {1.0, true};
    std::pair<double, bool>        p2 = {2.0, false};
    std::tuple<int, float, double> t  = {1, 2.0f, 3.0};
    comm.send(destination(rank::null), send_buf(p));
    comm.send(destination(rank::null), send_buf(t));
    comm.send(destination(rank::null), send_buf(p2));
}
