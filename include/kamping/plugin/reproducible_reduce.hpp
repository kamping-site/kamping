#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/data_buffer.hpp"
#include "kamping/named_parameter_check.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/named_parameters_detail/status_parameters.hpp"
#include "kamping/p2p/isend.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/plugin/plugin_helpers.hpp"

namespace kamping::plugin {

// Binary Tree Reduce
namespace reproducible_reduce {

/// @brief Encapsulates a single intermediate result (value) and its index
/// @tparam T Type of the stored value.
template <typename T>
struct MessageBufferEntry {
    /// @brief Global index according to reduction order
    size_t index;
    /// @brief Intermediate value during calculation
    T value;
};

constexpr uint8_t MAX_MESSAGE_LENGTH    = 4;
constexpr int     MESSAGEBUFFER_MPI_TAG = 0xb586772;

/// @brief Responsible for storing and communicating intermediate results between PEs.
/// @tparam T Type of the stored values.
/// @tparam Communicator Type of the underlying communicator.
template <typename T, typename Communicator>
class MessageBuffer {
    // TODO: how to shorten this result type
    using ResultType = NonBlockingResult<
        MPIResult<>,
        internal::DataBuffer<
            Request,
            internal::ParameterType,
            internal::ParameterType::request,
            internal::BufferModifiability::modifiable,
            internal::BufferOwnership::owning,
            internal::BufferType::out_buffer,
            BufferResizePolicy::no_resize,
            internal::BufferAllocation::lib_allocated,
            internal::default_value_type_tag>>;

public:
    /// @brief Construct a new message buffer utilizing the given communicator \p comm
    /// @param comm Underlying communicator used to send the messages.
    MessageBuffer(Communicator const& comm)
        : _entries(),
          _inbox(),
          _target_rank(),
          _outbox(),
          _buffer(),
          _request(nullptr),
          _awaited_numbers(0),
          _sent_messages(0),
          _sent_elements(0),
          _send_buffer_clear(true),
          _comm(comm) {
        _outbox.reserve(MAX_MESSAGE_LENGTH + 1);
        _buffer.reserve(MAX_MESSAGE_LENGTH + 1);
    }

    /// @brief Receive a message from another PE and store its contents.
    ///
    /// @param source_rank Rank of the sender.
    void receive(int const source_rank) {
        _comm.recv(
            recv_buf<BufferResizePolicy::resize_to_fit>(_buffer),
            tag(MESSAGEBUFFER_MPI_TAG),
            source(source_rank),
            recv_count(MAX_MESSAGE_LENGTH * sizeof(MessageBufferEntry<T>))
        );

        // Extract values from the message
        for (auto const entry: _buffer) {
            _inbox[entry.index] = entry.value;
        }
    }

    /// @brief Asynchronously send locally stored intermediate results.
    ///
    /// If there are none, no message is dispatched.
    void flush(void) {
        if (!_target_rank.has_value() || _outbox.empty())
            return;

        _request = std::make_unique<ResultType>(send());
        ++_sent_messages;

        _target_rank.reset();
        _send_buffer_clear = false;
    }

    /// @brief Wait until the message dispatched by flush() is actually sent and clear any stored values.
    void wait(void) {
        if (_send_buffer_clear) {
            return;
        }

        _request->wait();
        _outbox.clear();
        _send_buffer_clear = true;
    }

    /// @brief Store an intermediate result inside the message buffer for eventual transmission to its destination.
    ///
    /// Triggers a send if
    /// 1. the target rank of the currently stored values does not coincide with \p target_rank or
    /// 2. the message buffer is already full
    /// 3. the message buffer is full after adding \p value
    ///
    /// @param target_rank Rank of the PE which requires the value for further processing.
    /// @param index Global index of the value being sent.
    /// @param value Actual value that must be sent.
    void put(int const target_rank, size_t const index, T const value) {
        bool const outbox_full                        = _outbox.size() >= MAX_MESSAGE_LENGTH;
        bool const buffer_addressed_to_different_rank = _target_rank.has_value() && _target_rank != target_rank;
        if (outbox_full || buffer_addressed_to_different_rank) {
            flush();
        }
        wait();

        // We can now overwrite target rank because either
        // A) it was previously different but flush() has reset it or
        // B) it already has the same value.
        _target_rank = target_rank;

        KASSERT(_outbox.size() < _outbox.capacity());
        KASSERT(_outbox.capacity() > 0);
        MessageBufferEntry<T> entry{index, value};
        _outbox.push_back(entry);

        if (_outbox.size() >= MAX_MESSAGE_LENGTH) {
            flush();
        }
        ++_sent_elements;
    }

    /// @brief Get the intermediate result with the specified \p index from \p source_rank.
    ///
    /// If the value has been received beforehand, it is immediately returned.
    /// Otherwise the method blocks until the message from \p source_rank containing the value arrives.
    ///
    /// @param source_rank Rank of the PE that holds the desired intermediate result.
    /// @param index Global index of the intermediate result.
    T const get(int const source_rank, size_t const index) {
        auto const entry = _inbox.find(index);
        T          value;

        if (entry != _inbox.end()) {
            // If we have the number in our inbox, directly return it.
            value = entry->second;
            _inbox.erase(entry);
        } else {
            // If not, we will wait for a message, but make sure no one is waiting for our results.
            flush();
            wait();
            receive(source_rank);

            auto const new_entry = _inbox.find(index);
            KASSERT(new_entry != _inbox.end());
            value = new_entry->second;
            _inbox.erase(new_entry);
        }

        return value;
    }

private:
    auto send() {
        return _comm.isend(send_buf(_outbox), destination(*_target_rank), tag(MESSAGEBUFFER_MPI_TAG), request());
    }

private:
    std::array<MessageBufferEntry<T>, MAX_MESSAGE_LENGTH> _entries;
    std::map<uint64_t, T>                                 _inbox;
    std::optional<int>                                    _target_rank;
    std::vector<MessageBufferEntry<T>>                    _outbox;
    std::vector<MessageBufferEntry<T>>                    _buffer;
    std::unique_ptr<ResultType>                           _request;
    size_t                                                _awaited_numbers;
    size_t                                                _sent_messages;
    size_t                                                _sent_elements;
    bool                                                  _send_buffer_clear;
    Communicator const&                                   _comm;
};

// Helper functions

/// @brief Get the index of the parent of non-negative index \p i.
inline auto tree_parent(size_t const i) {
    KASSERT(i != 0);

    // Clear least significand set bit
    return i & (i - 1);
}

/// @brief Return the number of indices contained by the subtree with index \p i.
inline auto tree_subtree_size(size_t const i) {
    auto const largest_child_index{i | (i - 1)};
    return largest_child_index + 1 - i;
}

/// @brief Return the rank of the PE that holds the intermediate result with the specified \p index according to a \p
/// start_indices map.
inline auto tree_rank_from_index_map(std::map<size_t, size_t> const& start_indices, size_t const index) {
    // Get an iterator to the start index that is greater than index
    auto it = start_indices.upper_bound(index);
    KASSERT(it != start_indices.begin());
    --it;

    return kamping::asserting_cast<size_t>(it->second);
}

/// @brief Calculate the indices of intermediate results that must be communicated to other PEs.
///
/// @param region_begin Index of the first element assigned to the local rank.
/// @param region_end Index of the first element larger than \p region_begin that is not assigned to the local PE.
inline auto tree_rank_intersecting_elements(size_t const region_begin, size_t const region_end) {
    std::vector<size_t> result;

    size_t const region_size = region_end - region_begin;

    if (region_begin == 0 || region_size == 0) {
        return result;
    }

    size_t index{region_begin};
    while (index < region_end) {
        if (index > 0) {
            KASSERT(tree_parent(index) < region_begin);
        }
        result.push_back(index);
        index += tree_subtree_size(index);
    }

    return result;
}

/// @brief Calculate the logarithm to base 2 of the specified \p value.
///
/// Rounds down:
///
/// @code{.cpp}
/// ( log2l(4) == 2 == log2l(5) )
/// @endcode
inline auto log2l(size_t const value) {
    // See https://stackoverflow.com/a/994623
    size_t       i            = value;
    unsigned int target_value = 0;
    while (i >>= 1)
        ++target_value;

    return target_value;
}

/// @brief Return the number of necessary passes through the array to fully reduce the subtree with the specified \p
/// index.
inline size_t subtree_height(size_t const index) {
    KASSERT(index != 0);

    return log2l(tree_subtree_size(index));
}

/// @brief Return the number of necessary passes through the array to fully reduce a tree with \p global_size elements.
inline size_t tree_height(size_t const global_size) {
    if (global_size == 0) {
        return 0U;
    }

    unsigned int result = log2l(global_size);

    if (global_size > (1UL << result)) {
        return result + 1;
    } else {
        return result;
    }
}

/// @brief Communicator that can reproducibly reduce an array of a fixed size according to a binary tree scheme.
///
/// @tparam T Type of the elements that are to be reduced.
/// @tparam DefaultContainerType Container type of the original communicator.
template <typename T, typename Communicator>
class ReproducibleCommunicator {
public:
    /// @brief Create a new reproducible communicator.
    /// @tparam Comm Type of the communicator.
    /// @param comm Underlying communicator to transport messages.
    /// @param start_indices Map from global array indices onto ranks on which they are held. Must have no gaps, start
    /// at index 0 and contain a sentinel element at the end.
    /// @param region_begin Index of the first element that is held locally.
    /// @param region_size Number of elements assigned to the current rank.
    ReproducibleCommunicator(
        Communicator const&            comm,
        std::map<size_t, size_t> const start_indices,
        size_t const                   region_begin,
        size_t const                   region_size
    )
        : _start_indices{start_indices},
          _region_begin{region_begin},
          _region_size{region_size},
          _region_end{region_begin + region_size},
          _global_size{(--start_indices.end())->first},
          _origin_rank{_global_size == 0 ? 0UL : tree_rank_from_index_map(_start_indices, 0)},
          _comm{comm},
          _rank_intersecting_elements(tree_rank_intersecting_elements(_region_begin, _region_end)),
          _reduce_buffer(_region_size),
          _message_buffer(_comm) {}

    /// @brief Reproducible reduction according to pre-initialized scheme.
    /// The following parameters are required:
    /// - \ref kamping::send_buf() containing the local elements that are reduced. This buffer has to match the size
    /// specified during creation of the \ref ReproducibleCommunicator.
    /// - \ref kamping::op() wrapping the operation to apply to the input.
    ///
    /// @param args All required arguments as described above.
    /// @return Final reduction result obtained by applying the operation in a fixed order to all input elements across
    /// PEs.
    template <typename... Args>
    T const reproducible_reduce(Args... args) {
        KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(send_buf, op), KAMPING_OPTIONAL_PARAMETERS());

        // get send buffer
        auto&& send_buf =
            internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();
        using send_value_type = typename std::remove_reference_t<decltype(send_buf)>::value_type;

        KASSERT(
            send_buf.size() == _region_size,
            "send_buf must have the same size as specified during creation of the reproducible communicator. "
                << "Is " << send_buf.size() << " but should be " << _region_size << " on rank " << _comm.rank()
        );

        static_assert(
            std::is_same_v<std::remove_const_t<send_value_type>, T>,
            "send type must be equal to the type used during Communicator initiation"
        );

        // Get the operation used for the reduction. The signature of the provided function is checked while building.
        auto& operation_param = internal::select_parameter_type<internal::ParameterType::op>(args...);
        // If you want to understand the syntax of the following line, ignore the "template " ;-)
        auto operation = operation_param.template build_operation<send_value_type>();

        return _perform_reduce(send_buf.data(), operation);
    }

private:
    template <typename Func>
    T const _perform_reduce(T const* buffer, Func&& op) {
        for (auto const index: _rank_intersecting_elements) {
            if (tree_subtree_size(index) > 16) {
                // If we are about to do some considerable amount of work, make sure
                // the send buffer is empty so noone is waiting for our results
                _message_buffer.flush();
            }
            auto const target_rank = tree_rank_from_index_map(_start_indices, tree_parent(index));
            T const    value       = _perform_reduce(index, buffer, op);
            _message_buffer.put(asserting_cast<int>(target_rank), index, value);
        }

        _message_buffer.flush();
        _message_buffer.wait();

        T result;
        if (_comm.rank() == _origin_rank) {
            result = _perform_reduce(0, buffer, op);
        }

        _comm.bcast_single(kamping::send_recv_buf(result), kamping::root(_origin_rank));

        return result;
    }

    template <typename Func>
    T const _perform_reduce(size_t const index, T const* buffer, Func&& op) {
        if ((index & 1) == 1) {
            return buffer[index - _region_begin];
        }

        size_t const max_x =
            (index == 0) ? _global_size - 1 : std::min(_global_size - 1, index + tree_subtree_size(index) - 1);
        size_t const max_y = (index == 0) ? tree_height(_global_size) : subtree_height(index);

        KASSERT(max_y < 64, "Unreasonably large max_y");

        size_t const largest_local_index = std::min(max_x, _region_end - 1);
        auto const   n_local_elements    = largest_local_index + 1 - index;

        size_t   elements_in_buffer = n_local_elements;
        T*       destination_buffer = _reduce_buffer.data();
        T const* source_buffer      = static_cast<T const*>(buffer + (index - _region_begin));

        for (size_t y = 1; y <= max_y; y += 1) {
            size_t const stride           = 1UL << (y - 1);
            size_t       elements_written = 0;

            for (size_t x = 0; x + 2 <= elements_in_buffer; x += 2) {
                T const a                              = source_buffer[x];
                T const b                              = source_buffer[x + 1];
                destination_buffer[elements_written++] = op(a, b);
            }
            size_t const remaining_elements = elements_in_buffer - 2 * elements_written;
            KASSERT(remaining_elements <= 1);

            if (remaining_elements == 1) {
                auto const indexA = index + (elements_in_buffer - 1) * stride;
                auto const indexB = indexA + stride;

                T const elementA = source_buffer[elements_in_buffer - 1];
                if (indexB > max_x) {
                    // This element is the last because the subtree ends here
                    destination_buffer[elements_written++] = elementA;
                } else {
                    auto const source_rank = tree_rank_from_index_map(_start_indices, indexB);
                    T          elementB    = _message_buffer.get(asserting_cast<int>(source_rank), indexB);
                    destination_buffer[elements_written++] = op(elementA, elementB);
                }
            }

            // After first iteration, read only from accumulation buffer
            source_buffer      = destination_buffer;
            elements_in_buffer = elements_written;
        }

        KASSERT(elements_in_buffer == 1);
        return destination_buffer[0];
    }

    std::map<size_t, size_t> const _start_indices;
    size_t const                   _region_begin, _region_size, _region_end, _global_size;
    size_t const                   _origin_rank;
    Communicator const&            _comm;
    std::vector<size_t> const      _rank_intersecting_elements;
    std::vector<T>                 _reduce_buffer;
    MessageBuffer<T, Communicator> _message_buffer;
}; // namespace kamping::plugin
} // namespace reproducible_reduce

/// @brief Reproducible reduction of distributed arrays.
///
/// To make a reduction operation reproducible independent of communicator size and operation associativity, the
/// computation order must be fixed. We assign a global index to each element and let a binary tree dictate the
/// computation as seen in the figure below:
///
/// \image html tree_reduction.svg "Reduction of 16 elements distributed over 4 PEs"
///
///
/// The ordering of array elements must not necessarily follow the rank order of PEs.
/// We represent the distribution of array elements as a list of send_counts and displacements for each rank.
/// For the example above, send_counts would be `{4, 4, 4, 4}` since each rank
/// keeps four elements, and the displacement would be `{8, 4, 0, 12}`, since
/// the first element of rank 0 has index 8, the first element of rank 1 has
/// index 4 and so on.
///
/// More background of reproducible reduction is provided
/// [here](https://cme.h-its.org/exelixis/pubs/bachelorChristop.pdf).
///
template <typename Comm, template <typename...> typename DefaultContainerType>
class ReproducibleReducePlugin
    : public kamping::plugin::PluginBase<Comm, DefaultContainerType, ReproducibleReducePlugin> {
public:
    /// @brief Create a communicator with a fixed distribution of a global array that can perform reductions in the same
    /// reduction order.
    ///
    /// The following parameters are required:
    /// - \ref kamping::send_counts() containing the number of elements each rank holds locally.
    /// - \ref kamping::recv_displs() containing the displacement (a.k.a. starting index) for each rank.
    ///
    /// For further details, see documentation of the \ref ReproducibleReducePlugin
    ///
    /// Note that the reduce operation sends messages with the tag `0xb586772`.
    /// During the reduce, no messages shall be sent on the underlying
    /// communicator with this tag to avoid interference and potential
    /// deadlocks.
    ///
    /// @tparam T Type of the elements that are to be reduced.
    /// @param args All required arguments as specified above.
    /// @return A \ref reproducible_reduce::ReproducibleCommunicator
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

        // Assert distribution is the same on all ranks
        for (auto i = 0U; i < send_counts.size(); ++i) {
            KASSERT(
                comm.is_same_on_all_ranks(send_counts.data()[i]),
                "send_counts value for rank " << i << " is not uniform across the cluster",
                assert::light_communication
            );
            KASSERT(
                comm.is_same_on_all_ranks(recv_displs.data()[i]),
                "recv_displs value for rank " << i << " is not uniform across the cluster",
                assert::light_communication
            );
        }

        KASSERT(global_array_length > 0, "The array must not be empty");

        // Construct index map which maps global array indices to PEs
        std::map<size_t, size_t> start_indices;
        for (size_t p = 0; p < comm.size(); ++p) {
            KASSERT(send_counts.data()[p] >= 0, "send_count for rank " << p << " is negative");
            KASSERT(recv_displs.data()[p] >= 0, "displacement for rank " << p << " is negative");

            if (send_counts.data()[p] == 0) {
                continue;
            }

            start_indices[asserting_cast<size_t>(recv_displs.data()[p])] = p;
        }
        start_indices[global_array_length] = comm.size(); // guardian element

        KASSERT(start_indices.begin()->first >= 0UL, "recv_displs must not contain negative displacements");
        KASSERT(start_indices.begin()->first == 0UL, "recv_displs must have entry for index 0");

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

        return reproducible_reduce::ReproducibleCommunicator<T, Comm>(
            this->to_communicator(),
            start_indices,
            asserting_cast<size_t>(recv_displs.data()[comm.rank()]),
            asserting_cast<size_t>(send_counts.data()[comm.rank()])
        );
    }
};
} // namespace kamping::plugin
