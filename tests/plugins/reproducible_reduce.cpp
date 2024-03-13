#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <type_traits>
#include <vector>
#include <memory>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/named_parameters_detail/status_parameters.hpp"
#include "kamping/p2p/isend.hpp"
#include "kamping/collectives/bcast.hpp"
#include "kamping/plugin/plugin_helpers.hpp"

using kamping::BufferResizePolicy;

// Binary Tree Reduce

// MessageBuffer
template <typename T>/*{{{*/
struct MessageBufferEntry {
    size_t index;
    T      value;
};

const uint8_t MAX_MESSAGE_LENGTH    = 4;
int const     MESSAGEBUFFER_MPI_TAG = 1;

template<typename T> class TD;


template <typename T, template <typename...> typename DefaultContainerType>
class MessageBuffer {
using ResultType = kamping::NonBlockingResult<kamping::MPIResult<>, kamping::internal::DataBuffer<kamping::Request,kamping::internal::ParameterType::request, kamping::internal::BufferModifiability::modifiable, kamping::internal::BufferOwnership::owning, kamping::internal::BufferType::out_buffer, kamping::BufferResizePolicy::no_resize, kamping::internal::BufferAllocation::lib_allocated, kamping::internal::default_value_type_tag>>;
public:
    MessageBuffer(kamping::Communicator<DefaultContainerType> const& comm)
        : _inbox(),
          _target_rank(-1),
          _awaited_numbers(0),
          _sent_messages(0),
          _sent_elements(0),
          _send_buffer_clear(true),
          _comm(comm) {
        _outbox.reserve(MAX_MESSAGE_LENGTH + 1);
        _buffer.reserve(MAX_MESSAGE_LENGTH);
    }

    void receive(int const source_rank) {
        _comm.recv(
            kamping::recv_buf<BufferResizePolicy::resize_to_fit>(_buffer),
            kamping::tag(MESSAGEBUFFER_MPI_TAG),
            kamping::source(source_rank),
            kamping::recv_count(sizeof(MessageBufferEntry<T>) * MAX_MESSAGE_LENGTH),
            kamping::status_out()
        );

        // Extract values from the message
        for (auto const entry: _buffer) {
            _inbox[entry.index] = entry.value;
        }
    }

    void flush(void) {
        if (_target_rank == -1 || _outbox.size() == 0)
            return;

        _request = std::make_unique<ResultType>(_comm.isend(
            kamping::send_buf(_outbox),
            kamping::destination(_target_rank),
            kamping::tag(MESSAGEBUFFER_MPI_TAG),
            kamping::request()
        ));
        //TD<decltype(m_request)> a;


        ++_sent_messages;

        _target_rank       = -1;
        _send_buffer_clear = false;
    }

    void wait(void) {
        if (_send_buffer_clear)
            return;

        _request->wait();
        _outbox.clear();
        _send_buffer_clear = true;
    }

    void put(int const target_rank, const size_t index, const T value) {
        if (_outbox.size() >= MAX_MESSAGE_LENGTH || this->_target_rank != _target_rank) {
            flush();
        }
        wait();

        if (_target_rank == -1) {
            _target_rank = target_rank;
        }

        _outbox.emplace_back(index, value);

        if (_outbox.size() >= MAX_MESSAGE_LENGTH)
            flush();
        ++_sent_elements;
    }

    const T get(int const source_rank, const size_t index) {
        // If we have the number in our inbox, directly return it
        if (auto const entry = _inbox.find(index); entry != _inbox.end()) {
            const T value = entry->second;
            _inbox.erase(entry);
            return value;
        }

        // If not, we will wait for a message, but make sure no one is waiting for our results.
        flush();
        wait();
        receive(source_rank);

        KASSERT(_inbox.find(index) != _inbox.end());

        auto const entry = _inbox.find(index);
        const T    value = entry->second;
        _inbox.erase(entry);
        return value;
    }

protected:
    std::array<MessageBufferEntry<T>, MAX_MESSAGE_LENGTH> _entries;
    std::map<uint64_t, T>                                 _inbox;
    int                                                   _target_rank;
    std::vector<MessageBufferEntry<T>>                    _outbox;
    std::vector<MessageBufferEntry<T>>                    _buffer;
    std::unique_ptr<ResultType>                           _request; 
    size_t                                                _awaited_numbers;
    size_t                                                _sent_messages;
    size_t                                                _sent_elements;
    bool                                                  _send_buffer_clear;
    kamping::Communicator<DefaultContainerType> const&    _comm;
};/*}}}*/

template <typename T, template <typename...> typename DefaultContainerType>
class ReproducibleCommunicator {
public:
    using Communicator = kamping::Communicator<DefaultContainerType>;

    template <typename U>
    kamping::Communicator<DefaultContainerType> init_comm(U comm) {
        // TODO: how to simply set communicator without this identity split operation (copied from alltoall plugin)
        auto mycomm = comm.split(0);
        return Communicator(mycomm.disown_mpi_communicator(), mycomm.root_signed(), true);
    }

    template <
        template <typename...> typename = DefaultContainerType,
        template <typename, template <typename...> typename>
        typename... Plugins>
    ReproducibleCommunicator(
        kamping::Communicator<DefaultContainerType, Plugins...> const& comm,
        const std::map<size_t, size_t>                                 start_indices,
        const size_t                                                   region_begin,
        const size_t                                                   region_size
    )
        : _start_indices{start_indices},
          _region_begin{region_begin},
          _region_size{region_size},
          _region_end{region_begin + region_size},
          _global_size{start_indices.end()->second},
          _origin_rank{_global_size == 0 ? 0 : static_cast<int>(_rank_from_index_map(0))},
          _comm{init_comm(comm)},
          _rank_intersecting_elements(_calculate_rank_intersecting_elements()),
          _reduce_buffer(_region_size),
          _message_buffer(_comm) {}

    template <typename... Args>
    const T reproducible_reduce(Args... args) {
        using namespace kamping;
        KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf, op), KAMPING_OPTIONAL_PARAMETERS());

        // get send buffer
        auto&& send_buf =
            internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();
        using send_value_type = typename std::remove_reference_t<decltype(send_buf)>::value_type;

        static_assert(
            std::is_same_v<std::remove_const_t<send_value_type>, T>,
            "send type must be equal to the type used during Communicator initiation"
        );

        auto& operation_param = internal::select_parameter_type<internal::ParameterType::op>(args...);
        auto operation = operation_param.template build_operation<send_value_type>();

        return _perform_reduce(send_buf.data());
    }

private:
    //template <typename F>
    const T _perform_reduce(const T *buffer) {
        for (auto const index: _rank_intersecting_elements) {
            if (_subtree_size(index) > 16) {
                // If we are about to do some considerable amount of work, make sure
                // the send buffer is empty so noone is waiting for our results
                _message_buffer.flush();
            }
            _message_buffer.put(_rank_from_index_map(index), index, _perform_reduce(index, buffer));
        }

        _message_buffer.flush();
        _message_buffer.wait();

        T result;
        if (_comm.rank() == _origin_rank) {
            result = _perform_reduce(0, buffer);
        }

        _comm.bcast_single(kamping::send_recv_buf(result));

        return result;
    }

    //template <typename F>
    const T _perform_reduce(const size_t index, const T *buffer) {
        if ((index & 1) == 1) {
            return buffer[index - _region_begin];
        }
        
        const size_t max_x = (index == 0) ? _global_size - 1
            : std::min(_global_size - 1, index + _subtree_size(index) - 1);
        const size_t max_y = (index == 0) ? static_cast<size_t>(ceil(log2(_global_size)))
            : static_cast<size_t>(log2(_subtree_size(index)));

        const size_t largest_local_index = std::min(max_x, _region_end - 1);
        const auto n_local_elements = largest_local_index + 1 - index;

        size_t elements_in_buffer = n_local_elements;
        T *destination_buffer = _reduce_buffer.data();
        const T *source_buffer = static_cast<const T *>(buffer + (index - _region_begin));


        for (size_t y = 1; y <= max_y; y += 1) {
            size_t elements_written = 0;

            for (size_t x = 0; x + 2 <= elements_in_buffer; x += 2) {
                // TODO: actually apply operation from parameters.
                destination_buffer[elements_written++] = source_buffer[x] + source_buffer[x + 1];
            }
            const size_t remaining_elements = elements_in_buffer - 2 * elements_written;
            KASSERT(0 <= remaining_elements && remaining_elements <= 1);

            if (remaining_elements == 1) {
                destination_buffer[elements_written++] = source_buffer[elements_in_buffer];
            }

            // After first iteration, read only from accumulation buffer
            source_buffer = destination_buffer;
            elements_in_buffer = elements_written;
        }

        KASSERT(elements_in_buffer == 1);
        return destination_buffer[0];
        
    }


    auto _calculate_rank_intersecting_elements(void) const {
        std::vector<size_t> result;

        if (_region_begin == 0 || _region_size == 0) {
            return result;
        }

        size_t index{0};
        while (index < _region_end) {
            KASSERT(_parent(index) < _region_begin);
            result.push_back(index);
            index += _subtree_size(index);
        }

        return result;
    }

    auto _parent(const size_t i) const {
        KASSERT(i != 0);

        // clear least significand set bit
        return i & (i - 1);
    }

    auto _subtree_size(const size_t i) const {
        auto const largest_child_index{i | (i - 1)};
        return largest_child_index + 1 - i;
    }

    auto _rank_from_index_map(const size_t index) {
        // Get an iterator to the start index that is greater than index
        auto it = _start_indices.upper_bound(index);
        assert(it != _start_indices.begin());
        --it;

        return kamping::asserting_cast<int>(it->second);
    }

    const std::map<size_t, size_t>                    _start_indices;
    const size_t                                      _region_begin, _region_size, _region_end, _global_size;
    const int _origin_rank;
    const kamping::Communicator<DefaultContainerType> _comm;
    const std::vector<size_t>                         _rank_intersecting_elements;
    std::vector<T> _reduce_buffer;
    MessageBuffer<T, DefaultContainerType> _message_buffer;
};

// Plugin Code
template <typename Comm, template <typename...> typename DefaultContainerType>
class ReproducibleReducePlugin
    : public kamping::plugin::PluginBase<Comm, DefaultContainerType, ReproducibleReducePlugin> {
public:
    template <typename T, typename... Args>
    auto make_reproducible_comm(Args... args) {
        using namespace kamping;

        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(recv_displs, send_counts),
            KAMPING_OPTIONAL_PARAMETERS()
        );

        using default_recv_displs_type = decltype(kamping::recv_displs_out(alloc_new<DefaultContainerType<int>>));
        auto&& recv_displs =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_displs, default_recv_displs_type>(
                std::tuple(),
                args...
            )
                .template construct_buffer_or_rebind<DefaultContainerType>();
        using recv_displs_type = typename std::remove_reference_t<decltype(recv_displs)>::value_type;
        static_assert(std::is_same_v<std::remove_const_t<recv_displs_type>, int>, "Recv displs must be of type int");

auto const& send_counts = internal::select_parameter_type<internal::ParameterType::send_counts>(args...)
                                      .template construct_buffer_or_rebind<DefaultContainerType>();
        // This is the value type (i.e. of the underlying container)
        // using sendcounts_type = typename std::remove_reference_t<decltype(send_counts)>::value_type;

        auto comm = this->to_communicator();
        KASSERT(send_counts.size() == comm.size(), "send_counts must be of same size as communicator");
        KASSERT(recv_displs.size() == comm.size(), "recv_displs must be of same size as communicator");

        auto const global_array_length = static_cast<size_t>(
            std::reduce(send_counts.data(), send_counts.data() + send_counts.size(), 0, std::plus<>())
        );

        // Construct index map which maps global array indices to PEs
        std::map<size_t, size_t> start_indices;
        for (size_t p = 0; p < comm.size(); ++p) {
            KASSERT(send_counts.data()[p] >= 0, "send_count for rank " << p << " is negative");
            KASSERT(recv_displs.data()[p] >= 0, "displacement for rank " << p << " is negative");

            if (send_counts.data()[p] == 0)
                continue;

            start_indices[asserting_cast<size_t>(recv_displs.data()[p])] = p;
        }
        start_indices[global_array_length] = comm.size(); // guardian element

        KASSERT(start_indices.find(0) != start_indices.end(), "recv_displs does not have entry for index 0");
        // Verify correctness of index map
        for (auto it = start_indices.begin(); it != start_indices.end(); ++it) {
            auto const next = std::next(it);
            if (next == start_indices.end())
                break;

            auto const rank              = it->second;
            auto const region_start      = it->first;
            auto const region_end        = region_start + asserting_cast<size_t>(send_counts.data()[rank]);
            auto const next_rank         = next->second;
            auto const next_region_start = next->first;

            KASSERT(
                region_end == next_region_start,
                "Region of rank " << rank << " ends at index " << region_end << ", but next region of rank "
                                  << next_rank << " starts at index " << next_region_start
            );
        }

        return ReproducibleCommunicator<T, DefaultContainerType>(
            this->to_communicator(),
            start_indices,
            asserting_cast<size_t>(recv_displs.data()[comm.rank()]),
            asserting_cast<size_t>(send_counts.data()[comm.rank()])
        );
    }
};

TEST(ReproducibleReduceTest, PluginInit) {
    kamping::Communicator<std::vector, ReproducibleReducePlugin> comm;

    double const        epsilon = std::numeric_limits<double>::epsilon();
    std::vector<double> test_array{1, 1 + epsilon, 2 + epsilon, epsilon, 8, 9};

    int              values_per_rank = test_array.size() / comm.size();
    std::vector<int> send_counts(comm.size(), values_per_rank);
    std::vector<int> recv_displs;

    for (size_t i = 0; i < comm.size(); i++) {
        recv_displs.push_back(i);
        i += kamping::asserting_cast<size_t>(send_counts[i]);
    }

    auto reproducible_comm =
        comm.make_reproducible_comm<double>(kamping::recv_displs(recv_displs), kamping::send_counts(send_counts));

    auto v = reproducible_comm.reproducible_reduce(
            kamping::send_buf(test_array), 
            kamping::op(std::plus<double>()));

    EXPECT_EQ(v, std::reduce(test_array.begin(), test_array.end(), std::plus<>()));


}
