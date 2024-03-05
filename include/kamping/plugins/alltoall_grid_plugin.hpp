#include <numeric>

#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/plugins/plugin_helpers.hpp"

namespace kamping::plugin {

/// @brief Descriptor for different levels for message envelopes used in indirect communication.
enum MessageEnvelopeLevel {
    no_envelope,           ///< do not use an envelope at all (if possible)
    source,                ///< only additionally add the source PE in the envelope (if possible)
    source_and_destination ///< add source and destination PE in the envelope
};

namespace grid_plugin_helpers {

/// @brief Mixin for \ref MessageEnvelope to store a source PE.
struct Source {
    /// @brief Get destination PE.
    [[nodiscard]] size_t get_source() const {
        return asserting_cast<size_t>(source);
    }

    /// @brief Get destination PE.
    [[nodiscard]] int get_source_signed() const {
        return source;
    }

    /// @brief Set destination PE.
    void set_source(int value) {
        source = value;
    }
    int source; ///< Rank of source PE.
};

/// @brief Mixin for \ref MessageEnvelope to store a destination PE.
struct Destination {
    /// @brief Get destination PE.
    [[nodiscard]] size_t get_destination() const {
        return asserting_cast<size_t>(destination);
    }

    /// @brief Get destination PE.
    [[nodiscard]] int get_destination_signed() const {
        return destination;
    }

    /// @brief Set destination PE.
    void set_destination(int value) {
        destination = value;
    }
    int destination; ///< Rank of destination PE.
};

/// @brief Augments a plain message with additional information via \tparam Attributes
/// @tparam PayloadType Plain message type.
/// @tparam Attributes source or destination information
template <typename PayloadType, typename... Attributes>
struct MessageEnvelope : public Attributes... {
    using Payload = PayloadType; ///< Underlying Type of message.
    static constexpr bool has_source_information =
        internal::type_list<Attributes...>::template contains<Source>; ///< Indicates whether the envelope contains the
                                                                       ///< source PE.
    static constexpr bool has_destination_information =
        internal::type_list<Attributes...>::template contains<Destination>; ///< Indicates whether the envelope contains
                                                                            ///< the destination PE.
    /// @brief Default constructor.
    MessageEnvelope() = default;

    /// @brief Constructor for wrapping a message.
    MessageEnvelope(PayloadType payload) : _payload{std::move(payload)} {}

    /// @brief Return reference to payload.
    Payload& get_payload() {
        return _payload;
    }

    /// @brief Return const reference to payload.
    Payload const& get_payload() const {
        return _payload;
    }

    /// @brief Output operator.
    friend std::ostream& operator<<(std::ostream& out, MessageEnvelope const& msg) {
        out << "(payload: " << msg.get_payload();
        if constexpr (MessageEnvelope::has_source_information) {
            out << ", source: " << msg.get_source();
        }
        if constexpr (MessageEnvelope::has_destination_information) {
            out << ", destination: " << msg.get_destination();
        }
        out << ")";
        return out;
    }

    Payload _payload; ///< payload
};

/// @brief Select the right MessageEnvelope depending on the provided MsgEnvelopeLevel.
template <MessageEnvelopeLevel level, typename T>
using MessageEnvelopeType = std::conditional_t<
    level == MessageEnvelopeLevel::no_envelope,
    T,
    std::conditional_t<
        level == MessageEnvelopeLevel::source,
        MessageEnvelope<T, Source>,
        MessageEnvelope<T, Source, Destination>>>;

} // namespace grid_plugin_helpers

/// @brief Object returned by \ref plugin::GridCommunicatorPlugin::make_grid_communicator() representing a grid
/// communicator which enables alltoall communication with a latency in `sqrt(p)` where p is the size of the
/// original communicator.
/// @tparam DefaultContainerType Container type of the original communicator.
template <template <typename...> typename DefaultContainerType>
class GridCommunicator {
public:
    using LevelCommunicator = kamping::Communicator<DefaultContainerType>; ///< Type of row and column communicator.

    /// @brief Creates a two dimensional grid by splitting the given communicator of size `p` into a row and a column
    /// communicator each of size about `sqrt(p)`.
    /// @tparam Comm Type of the communicator.
    /// @param comm Communicator to be split into a two dimensioal grid.
    template <
        template <typename...> typename = DefaultContainerType,
        template <typename, template <typename...> typename>
        typename... Plugins>
    GridCommunicator(kamping::Communicator<DefaultContainerType, Plugins...> const& comm)
        : _size_of_orig_comm{comm.size()},
          _rank_in_orig_comm{comm.rank()} {
        // GridCommunicator(kamping::Communicator<DefaultContainerType, Plugins...>& comm) {
        double const sqrt       = std::sqrt(comm.size());
        const size_t floor_sqrt = static_cast<size_t>(std::floor(sqrt));
        const size_t ceil_sqrt  = static_cast<size_t>(std::ceil(sqrt));
        // We want to ensure that #columns + 1 >= #rows >= #columns.
        // Therefore, use floor(sqrt(comm.size())) columns unless we have enough PEs to begin another row when using
        // ceil(sqrt(comm.size()) columns.
        const size_t threshold                   = floor_sqrt * ceil_sqrt;
        _number_columns                          = (comm.size() >= threshold) ? ceil_sqrt : floor_sqrt;
        const size_t num_pe_in_incomplete_column = comm.size() / _number_columns;
        auto [row_num, column_num] = pos_in_complete_grid(comm.rank()); // assume that we have a complete grid,
        _size_complete_rectangle   = _number_columns * num_pe_in_incomplete_column;
        if (comm.rank() >= _size_complete_rectangle) {
            row_num = comm.rank() % _number_columns; // rank() is member of last incomplete row,
            // therefore append it to one of the first
        }
        {
            auto split_comm = comm.split(static_cast<int>(row_num), comm.rank_signed());
            _row_comm       = LevelCommunicator(split_comm.disown_mpi_communicator(), split_comm.root_signed(), true);
        }
        {
            auto split_comm = comm.split(static_cast<int>(column_num), comm.rank_signed());
            _column_comm    = LevelCommunicator(split_comm.disown_mpi_communicator(), split_comm.root_signed(), true);
        }
    }

    /// @brief Indirect two dimensional grid based personalized alltoall exchange.
    /// The following parameters are required:
    /// - \ref kamping::send_buf() containing the data that is sent to each rank. The size of this buffer has to be at
    /// least the sum of the send_counts argument.
    /// - \ref kamping::send_counts() containing the number of elements to send to each rank.
    ///
    /// Internally, each element in the send buffer is wrapped in an envelope to faciliate the indirect routing. The
    /// envelope consists at least consists of the destination PE of each element but can be extended to also hold the
    /// source PE of the element. The caller can specify whether they want to keep this information also in the output
    /// via the \tparam envelope_level.
    ///
    /// @tparam envelope_level Determines the contents of the envelope of each returned element (no_envelope = use the
    /// actual data type of an exchanged element, source = augment the actual data type with the source PE,
    /// source_and_destination = agument the actual data type with the source and destination PE).
    /// @tparam Args Automatically deducted template parameters.
    /// @param args All required and any number of the optional buffers described above.
    /// @returns
    template <MessageEnvelopeLevel envelope_level = MessageEnvelopeLevel::no_envelope, typename... Args>
    auto alltoallv_with_envelope(Args... args) const {
        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(send_buf, send_counts),
            KAMPING_OPTIONAL_PARAMETERS()
        );
        // Get send_buf
        auto const& send_buf =
            internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();
        // Get send_counts
        auto const& send_counts = internal::select_parameter_type<internal::ParameterType::send_counts>(args...)
                                      .template construct_buffer_or_rebind<DefaultContainerType>();
        auto rowwise_recv_buf = rowwise_exchange<envelope_level>(send_buf, send_counts);
        return columnwise_exchange<envelope_level>(std::move(rowwise_recv_buf));
    }

    /// @brief Indirect two dimensional grid based personalized alltoall exchange.
    /// The following parameters are required:
    /// - \ref kamping::send_buf() containing the data that is sent to each rank. The size of this buffer has to be at
    /// least the sum of the send_counts argument.
    /// - \ref kamping::send_counts() containing the number of elements to send to each rank.
    ///
    /// Internally, each element in the send buffer is wrapped in an envelope to faciliate the indirect routing. The
    /// envelope consists at least consists of the destination PE of each element but can be extended to also hold the
    /// source PE of the element. The caller can specify whether they want to keep this information also in the output
    /// via the \tparam envelope_level.
    ///
    /// @tparam envelope_level Determines the contents of the envelope of each returned element (no_envelope = use the
    /// actual data type of an exchanged element, source = augment the actual data type with the source PE,
    /// source_and_destination = agument the actual data type with the source and destination PE).
    /// @tparam Args Automatically deducted template parameters.
    /// @param args All required and any number of the optional buffers described above.
    /// @returns
    template <typename... Args>
    auto alltoallv(Args... args) const {
        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(send_buf, send_counts),
            KAMPING_OPTIONAL_PARAMETERS(recv_buf, recv_counts)
        );
        constexpr MessageEnvelopeLevel envelope_level = MessageEnvelopeLevel::source;
        // Get send_buf
        auto& send_buf                = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
        using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
        using default_recv_value_type = std::remove_const_t<send_value_type>;
        // Get send_counts
        auto& send_counts        = internal::select_parameter_type<internal::ParameterType::send_counts>(args...);
        auto intermediate_result = alltoallv_with_envelope<envelope_level>(std::move(send_buf), std::move(send_counts));

        // Get recv counts
        using default_recv_counts_type = decltype(kamping::recv_counts_out(alloc_new<DefaultContainerType<int>>));
        auto&& recv_counts =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_counts, default_recv_counts_type>(
                std::tuple(),
                args...
            )
                .template construct_buffer_or_rebind<DefaultContainerType>();
        constexpr bool do_calculate_recv_counts = internal::has_to_be_computed<decltype(recv_counts)>;

        if constexpr (do_calculate_recv_counts) {
            recv_counts.resize_if_requested([&]() { return _size_of_orig_comm; });
            KASSERT(recv_counts.size() >= this->size(), "Recv counts buffer is not large enough.", assert::light);
            Span recv_counts_span(recv_counts.data(), recv_counts.size());
            std::fill(recv_counts_span.begin(), recv_counts_span.end(), 0);

            for (size_t i = 0; i < intermediate_result.size(); ++i) {
                ++recv_counts_span[intermediate_result[i].get_source()];
            }
        } else {
            KASSERT(recv_counts.size() >= this->size(), "Recv counts buffer is not large enough.", assert::light);
        }

        Span                      recv_counts_span(recv_counts.data(), recv_counts.size());
        DefaultContainerType<int> write_pos(_size_of_orig_comm);
        Span                      write_pos_span(write_pos.data(), write_pos.size());
        std::exclusive_scan(recv_counts_span.begin(), recv_counts_span.end(), write_pos_span.begin(), size_t(0));
        const size_t num_recv_elems = asserting_cast<size_t>(write_pos_span.back() + recv_counts_span.back());

        // Get recv_buf
        using default_recv_buf_type =
            decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
        auto&& recv_buf =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
                std::tuple(),
                args...
            )
                .template construct_buffer_or_rebind<DefaultContainerType>();
        auto compute_required_recv_buf_size = [&]() {
            return num_recv_elems;
        };

        recv_buf.resize_if_requested(compute_required_recv_buf_size);
        KASSERT(
            recv_buf.size() >= compute_required_recv_buf_size(),
            "Recv buffer is not large enough to hold all received elements.",
            assert::light
        );
        Span recv_buf_span(recv_buf.data(), recv_buf.size());
        Span intermediate_recv_buf_span(intermediate_result.data(), intermediate_result.size());
        for (auto const& elem: intermediate_recv_buf_span) {
            const size_t pos   = asserting_cast<size_t>(write_pos_span[elem.get_source()]++);
            recv_buf_span[pos] = std::move(elem.get_payload());
        }
        return internal::make_mpi_result<std::tuple<Args...>>(std::move(recv_buf), std::move(recv_counts));
    }

private:
    template <typename SendCounts>
    [[nodiscard]] auto compute_row_send_counts(SendCounts const& send_counts) const {
        SendCounts row_send_counts(_row_comm.size(), 0);
        for (size_t i = 0; i < send_counts.size(); ++i) {
            size_t const destination_pe = get_destination_in_rowwise_exchange(i);
            row_send_counts.data()[destination_pe] += send_counts.data()[i];
        }
        return row_send_counts;
    }

    struct GridPosition {
        size_t row_index;
        size_t col_index;
    };

    [[nodiscard]] GridPosition pos_in_complete_grid(size_t rank) const {
        return GridPosition{rank / _number_columns, rank % _number_columns};
    }

    [[nodiscard]] size_t get_destination_in_rowwise_exchange(size_t destination_rank) const {
        return pos_in_complete_grid(destination_rank).col_index;
    }

    [[nodiscard]] size_t get_destination_in_colwise_exchange(size_t destination_rank) const {
        return pos_in_complete_grid(destination_rank).row_index;
    }

    template <MessageEnvelopeLevel envelope_level, typename SendBuffer, typename SendCounts>
    auto rowwise_exchange(SendBuffer const& send_buf, SendCounts const& send_counts) const {
        using namespace grid_plugin_helpers;
        auto const row_send_counts = compute_row_send_counts(send_counts.underlying());
        auto       row_send_displs = row_send_counts;
        Span       row_send_counts_span(row_send_counts.data(), row_send_counts.size());
        Span       row_send_displs_span(row_send_displs.data(), row_send_displs.size());
        std::exclusive_scan(
            row_send_counts_span.begin(),
            row_send_counts_span.end(),
            row_send_displs_span.begin(),
            size_t(0)
        );
        auto       index_displacements = row_send_displs;
        auto const total_send_count    = static_cast<size_t>(row_send_displs_span.back() + row_send_counts_span.back());

        using value_type = typename SendBuffer::value_type;
        using MsgType    = std::conditional_t<
            envelope_level == MessageEnvelopeLevel::no_envelope,
            MessageEnvelope<value_type, Destination>,
            MessageEnvelope<value_type, Source, Destination>>;
        DefaultContainerType<MsgType> rowwise_send_buf(total_send_count);
        size_t                        cur_chunk_offset = 0;

        for (size_t i = 0; i < send_counts.size(); ++i) {
            int const    destination        = asserting_cast<int>(i);
            auto const   destination_in_row = get_destination_in_rowwise_exchange(asserting_cast<size_t>(i));
            size_t const send_count         = asserting_cast<size_t>(send_counts.underlying().data()[i]);
            for (std::size_t ii = 0; ii < send_count; ++ii) {
                auto       elem  = send_buf.data()[cur_chunk_offset + ii];
                auto const idx   = index_displacements[destination_in_row]++;
                auto&      entry = rowwise_send_buf[static_cast<size_t>(idx)];
                entry            = MsgType(std::move(elem));
                entry.set_destination(destination
                ); // this has to be done independently of the envelope level, otherwise routing is not possible
                   //
                if constexpr (envelope_level != MessageEnvelopeLevel::no_envelope) {
                    entry.set_source(asserting_cast<int>(_rank_in_orig_comm));
                }
            }
            cur_chunk_offset += send_count;
        }
        return _row_comm.alltoallv(
            kamping::send_buf(rowwise_send_buf),
            kamping::send_counts(row_send_counts),
            send_displs(row_send_displs)
        );
    }

    template <
        MessageEnvelopeLevel envelope_level,
        template <typename...>
        typename RowwiseRecvBuf,
        typename... EnvelopeArgs>
    auto columnwise_exchange(RowwiseRecvBuf<grid_plugin_helpers::MessageEnvelope<EnvelopeArgs...>>&& rowwise_recv_buf
    ) const {
        using namespace grid_plugin_helpers;
        using IndirectMessageType = MessageEnvelope<EnvelopeArgs...>;
        using Payload             = typename IndirectMessageType::Payload;

        DefaultContainerType<int> send_counts(_column_comm.size(), 0);
        for (auto const& elem: rowwise_recv_buf) {
            ++send_counts[get_destination_in_colwise_exchange(elem.get_destination())];
        }
        DefaultContainerType<int> send_offsets = send_counts;
        Span                      send_counts_span(send_counts.data(), send_counts.size());
        Span                      send_offsets_span(send_offsets.data(), send_offsets.size());
        std::exclusive_scan(send_counts_span.begin(), send_counts_span.end(), send_offsets_span.begin(), size_t(0));
        DefaultContainerType<int> send_displacements = send_offsets;

        using MsgType = MessageEnvelopeType<envelope_level, Payload>;
        DefaultContainerType<MsgType> colwise_send_buf(rowwise_recv_buf.size());
        for (auto& elem: rowwise_recv_buf) {
            auto const dst_in_column = get_destination_in_colwise_exchange(elem.get_destination());
            auto&      entry         = colwise_send_buf[static_cast<std::size_t>(send_offsets[dst_in_column]++)];
            entry                    = MsgType(std::move(elem.get_payload()));
            if constexpr (envelope_level == MessageEnvelopeLevel::no_envelope) {
                // nothing to be done
            } else if constexpr (envelope_level == MessageEnvelopeLevel::source) {
                entry.set_source(elem.get_source_signed());
            } else if constexpr (envelope_level == MessageEnvelopeLevel::source_and_destination) {
                entry.set_source(elem.get_source_signed());
                entry.set_destination(elem.get_destination_signed());
            } else {
                static_assert(
                    envelope_level == MessageEnvelopeLevel::source_and_destination /*always false*/,
                    "This branch is never used"
                );
            }
        }
        {
            // deallocate rowwise_recv_buf as it is not needed anymore
            rowwise_recv_buf.clear();
            auto tmp = std::move(rowwise_recv_buf);
        }
        return _column_comm.alltoallv(
            kamping::send_buf(colwise_send_buf),
            kamping::send_counts(send_counts),
            kamping::send_displs(send_displacements)
        );
    }

private:
    size_t                                      _size_of_orig_comm;
    size_t                                      _rank_in_orig_comm;
    size_t                                      _size_complete_rectangle;
    size_t                                      _number_columns;
    kamping::Communicator<DefaultContainerType> _row_comm;
    kamping::Communicator<DefaultContainerType> _column_comm;
};
/// @brief Plugin adding a two dimensional communication grid to the communicator.
///
/// PEs are row-major and abs(`#row - #columns`) <= 1
/// 0  1  2  3
/// 4  5  6  7
/// 8  9  10 11
/// 12 13 14 15
///
/// If `#PE != #row * #column` then the PEs of the last incomplete row are transposed and appended to
/// the first rows and do not form an own row based communicator.
///  0  1  2  3 16
///  4  5  6  7 17
///  8  9  10 11
///  12 13 14 15
/// (16 17)
/// This enables personalized alltoall exchanges with a latency in about `sqrt(#PE)`.

template <typename Comm, template <typename...> typename DefaultContainerType>
class GridCommunicatorPlugin : public plugins::PluginBase<Comm, DefaultContainerType, GridCommunicatorPlugin> {
public:
    /// @brief Returns a \ref kamping::plugin::GridCommunicator.
    auto make_grid_communicator() {
        return GridCommunicator<DefaultContainerType>(this->to_communicator());
    }
};

} // namespace kamping::plugin
