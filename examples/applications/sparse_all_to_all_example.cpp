#include <cstddef>

#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameters.hpp"

// A very non-generic implementation of a sparse all-to-all communication that is conceptually similar to the generic
// one provided via kamping::Mailbox
class SparseAllToAll {
    template <typename T, typename Comm>
    std::unordered_map<size_t, std::vector<T>>
    alltoall(Comm& comm, std::unordered_map<size_t, std::vector<T>> const& send_messages, int const _tag) {
        using namespace ::kamping;

        std::unordered_map<size_t, std::vector<T>> recvd_messages;

        // TODO Extract this into a convenience function in kamping::Communicator
        // Define the function to receive the next message if one is available
        auto recv_msg_if_avail =
            [](Comm& comm, std::unordered_map<size_t, std::vector<T>>& recvd_messages, int const _tag) {
                MPI_Status recv_status;

                bool const msg_avail = comm.iprobe(source(rank::any), tag(_tag), status(recv_status));
                if (msg_avail) {
                    auto const _source = recv_status.MPI_SOURCE;
                    KASSERT(
                        recv_status.MPI_TAG == _tag,
                        "Tag of the iprobe-returned status does not match the requested tag.",
                        assert::light
                    );

                    KASSERT(
                        recvd_messages.find(_source) == recvd_messages.end(),
                        "Received a message from a source that has already sent a message.",
                        assert::light
                    );
                    recvd_messages[_source].emplace({});
                    auto const result =
                        comm.recv(source(_source), tag(_tag), recv_buf(recvd_messages[_source]).status(recv_status));
                }
            };

        // Send out all our messages using issend; keep the corresponding requests around for checking if the messages
        // have been received.
        std::vector<kamping::Request> send_msg_requests;
        for (auto const& [dest_rank, messages]: send_messages) {
            auto const result = comm.issend(send_buf(messages), dest(dest_rank), tag(_tag));
            send_msg_requests.emplace_back(std::move(result.extract_request()));
        }

        // TODO Rewrite this, once we have a request pool
        // Receive messages until all messages sent have been received
        while (!kamping::requests::test_all(send_msg_requests)) {
            // If there's a message available, receive it directly into recv_messages
            recv_msg_if_avail(comm, recvd_messages, tag);
        }

        // Enter a ibarrier; this signals to all other PEs that all messages we sent were received.
        auto ibarrier = comm.ibarrier();

        // Continue receiving messages sent to us until all ranks posted to the barrier that their (and thus all)
        // messages have been received.
        while (!ibarrier.test()) {
            recv_msg_if_avail(comm, recvd_messages, tag);
        }

        return recvd_messages;
    }
};
