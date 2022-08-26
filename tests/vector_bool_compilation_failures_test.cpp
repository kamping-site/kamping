// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <functional>

#include "helpers_for_testing.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/communicator.hpp"
#include "kamping/named_parameters.hpp"

int main(int /*argc*/, char** /*argv*/) {
    using namespace kamping;
    bool                single_element_bool [[maybe_unused]]   = false;
    kabool              single_element_kabool [[maybe_unused]] = false;
    std::vector<bool>   vector_bool                            = {false, true, false};
    std::vector<kabool> vector_kabool                          = {false, true, false};
    Communicator        comm;

#if defined(SINGLE_BOOL_VEC_BOOL)
    // the recv_buf would be a std::vector<bool>, which is not supported
    // solution: receive into a buffer which is not std::vector or use kabool
    comm.gather(send_buf(single_element_bool));
#elif defined(SINGLE_KABOOL_VEC_BOOL)
    // the recv_buf is a std::vector<bool>, which is not supported
    comm.gather(send_buf(single_element_kabool), recv_buf(vector_bool));
#elif defined(SEND_VEC_BOOL)
    // the send_buf is a std::vector<bool>, which is not supported
    comm.gather(send_buf(vector_bool));
#elif defined(SEND_VEC_KABOOL_RECV_VEC_BOOL)
    // the recv_buf is a std::vector<bool>, which is not supported
    comm.gather(send_buf(vector_kabool), recv_buf(vector_bool));
// If none of the above sections is active, this file will compile successfully.
#endif
}
