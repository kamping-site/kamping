#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>

#include <kamping/communicator.hpp>
#include <mpi.h>

#include "./prefix_doubling.hpp"
#include "kamping/environment.hpp"
#include "kamping/plugin/sort.hpp"

auto load_local_input(
    std::filesystem::path const& path, kamping::Communicator<std::vector, kamping::plugin::SampleSort>& comm
) {
    MPI_File mpi_file;

    MPI_File_open(comm.mpi_communicator(), path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);

    MPI_Offset global_file_size = 0;
    MPI_File_get_size(mpi_file, &global_file_size);

    size_t local_size     = kamping::asserting_cast<size_t>(global_file_size) / comm.size();
    size_t remaining_size = kamping::asserting_cast<size_t>(global_file_size) % comm.size();

    MPI_File_seek(mpi_file, comm.rank_signed() * kamping::asserting_cast<int32_t>(local_size), MPI_SEEK_SET);

    std::vector<uint8_t> result(local_size + (comm.rank() + 1 == comm.size() ? remaining_size : 0));

    MPI_File_read(
        mpi_file,
        result.data(),
        kamping::asserting_cast<int32_t>(result.size()),
        kamping::builtin_type<uint8_t>::data_type(),
        MPI_STATUS_IGNORE
    );

    return result;
}

int main(int argc, char* argv[]) {
    kamping::Environment env(argc, argv);

    kamping::Communicator<std::vector, kamping::plugin::SampleSort> comm;

    if (argc != 2) {
        std::cerr << "Wrong number of parameters" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <path_to_file>" << std::endl;
        kamping::comm_world().abort();
        return 1;
    }

    std::ifstream input_stream(argv[1], std::ios::in);
    if (!input_stream.is_open()) {
        std::cerr << "Could not open file " << argv[1] << std::endl;
        std::cerr << "Usage: " << argv[0] << " <path_to_file>" << std::endl;
        kamping::comm_world().abort();
        return 1;
    }

    auto local_input = load_local_input(argv[1], comm);

    auto suffix_array = prefix_doubling<uint32_t>(std::move(local_input), comm);

    return 0;
}
