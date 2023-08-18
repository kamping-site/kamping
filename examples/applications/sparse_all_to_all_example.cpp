#include <cstddef>

#include <mpi.h>

#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameters.hpp"

// A very non-generic implementation of a sparse all-to-all communication that is conceptually similar to the generic
// one provided via kamping::Mailbox
template <typename T, typename Comm>
class SparseAllToAll {
    struct RecvdMessage {
        std::vector<T> data;
        size_t         src_rank;
    };

    std::vector<RecvdMessage>
    alltoall(Comm& comm, std::unordered_map<size_t, std::vector<T>> const& send_messages, int const _tag) {
        using namespace ::kamping;

        std::vector<RecvdMessage> recvd_messages;
        recvd_messages.emplace_back({});

        // Checks if a new message is available for receiving and if so, receives into recvd_messages.
        auto const recv_msg_if_avail = [&comm, &recvd_messages, _tag]() {
            static Status recv_status;
            auto const&   current_msg = recvd_messages.back();

            bool const msg_received =
                comm.try_recv(recv_buf(current_msg.data), tag(_tag), source(rank::any), status(recv_status));
            if (msg_received) {
                current_msg.src_rank = recv_status.source();
                recvd_messages.emplace_back({});
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
            recv_msg_if_avail();
        }

        // Enter a ibarrier; this signals to all other PEs that all messages we sent were received.
        auto ibarrier = comm.ibarrier();

        // Continue receiving messages sent to us until all ranks posted to the barrier that their (and thus all)
        // messages have been received.
        while (!ibarrier.test()) {
            recv_msg_if_avail();
        }

        // Remove the last (empty) message from recvd_messages and return.
        KASSERT(!recvd_messages.empty());
        KASSERT(recvd_messages.back().data.empty());
        recvd_messages.pop_back();
        return recvd_messages;
    }
};
