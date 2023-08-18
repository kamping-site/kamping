// This file is part of KaMPIng.
//
// Copyright 2023 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#pragma once

#include <cstddef>

#include "kamping/communicator.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/p2p/isend.hpp"
#include "kamping/p2p/try_recv.hpp"

namespace kamping {

template <typename Value, typename Comm, typename Tag>
class Mailbox {
public:
    Mailbox(Comm comm, Tag tag) : _comm(comm), _tag(tag) {}

    template <typename Dest, typename SendBuf>
    void post(Dest const&& dest_rank, SendBuf const&& send_buf) {
        // Send out all our messages using issend; keep the corresponding requests around for checking if the messages
        // have been received.
        // TODO Create the tag parameter object once and store it for reuse?
        _comm.issend(dest(dest_rank), send_buf(send_buf), tag(_tag), request(_requests));
    }

    template <typename Callback>
    void recv_all(Callback recv_callback) {
        // Receive messages until all messages sent have been received
        while (!_requests.testall()) {
            this->_recv_msg_if_avail(recv_callback);
        }

        // Enter a ibarrier; this signals to all other PEs that all messages we sent were received.
        // TODO Document that the user shall not use an ibarrier inside his callback function. (Why would they?)
        auto ibarrier = _comm.ibarrier();

        // Continue receiving messages sent to us until all ranks posted to the barrier that their (and thus all)
        // messages have been received.
        while (!ibarrier.test()) {
            this->_recv_msg_if_avail(recv_callback);
        }
    }

private:
    Comm&              _comm;
    Tag                _tag;
    RequestPool        _requests;
    std::vector<Value> _recv_buffer;

    template <typename Callback>
    void _recv_msg_if_avail(Callback recv_callback) {
        Status     recv_status;
        bool const msg_received = _comm.try_recv(recv_buf(_recv_buffer), _tag, source(rank::any), status(recv_status));
        if (msg_received) {
            recv_callback(recv_status.source(), _recv_buffer);
        }
    };
};
} // namespace kamping
