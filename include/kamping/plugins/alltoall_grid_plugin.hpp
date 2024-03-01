#include <numeric>

#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/plugins/plugin_helpers.hpp"

namespace kamping::plugin {
namespace grid_plugin_helpers {
struct Source {
    [[nodiscard]] int get_source() const {
        return source;
    }
    void set_source(int value) {
        source = value;
    }
    int source;
};

struct Destination {
    [[nodiscard]] int get_destination() const {
        return destination;
    }
    void set_destination(int value) {
        destination = value;
    }
    int destination;
};

/// @brief Descriptor for different levels for message envelopes used in indirect communication.
enum MsgEnvelopeLevel {
    no_envelope,           ///< do not use an envelope at all (if possible)
    source,                ///< only additionally add the source PE in the envelope (if possible)
    source_and_destination ///< add source and destination PE in the envelope
};

/// @brief Augments a plain message with additional information via \tparam Attributes
/// @tparam PayloadType Plain message type.
/// @tparam Attributes source or destination information
template <typename PayloadType, typename... Attributes>
struct MessageEnvelope : public Attributes... {
    using Payload = PayloadType;
    static constexpr bool has_source_information =
        internal::type_list<Attributes...>::template contains<Source>; ///< Indicates whether the envelope contains the
                                                                       ///< source PE.
    static constexpr bool has_destination_information =
        internal::type_list<Attributes...>::template contains<Destination>; ///< Indicates whether the envelope contains
                                                                            ///< the destination PE.
    MessageEnvelope() = default;
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
template <MsgEnvelopeLevel level, typename T>
using MessageEnvelopeType = std::conditional_t<
    level == MsgEnvelopeLevel::no_envelope,
    T,
    std::conditional_t<
        level == MsgEnvelopeLevel::source,
        MessageEnvelope<T, Source>,
        MessageEnvelope<T, Source, Destination>>>;

} // namespace grid_plugin_helpers

/// @brief Plugin addng a two dimensional grid communicator to the communicator.
///
/// PEs are row-major and abs(#row - #columns) <= 1
/// 0  1  2  3
/// 4  5  6  7
/// 8  9  10 11
/// 12 13 14 15
///
/// If #PE != #row * #column then the PEs of the last incomplete row are transposed and appended to
/// the first rows and do not form an own row based communicator.
///  0  1  2  3 16
///  4  5  6  7 17
///  8  9  10 11
///  12 13 14 15
/// (16 17)
///
template <typename Comm, template <typename...> typename DefaultContainerType>
class GridCommunicatorPlugin : public plugins::PluginBase<Comm, DefaultContainerType, GridCommunicatorPlugin> {
public:
    using LevelCommunicator = kamping::Communicator<DefaultContainerType>; ///< Type of row and column communicator.

    /// @brief initialized the virtual grid and splits the communicator into a row and column communicator.
    void initialize_grid() {
        auto&        self       = this->to_communicator();
        double const sqrt       = std::sqrt(self.size());
        const size_t floor_sqrt = static_cast<size_t>(std::floor(sqrt));
        const size_t ceil_sqrt  = static_cast<size_t>(std::ceil(sqrt));
        // if comm size exceeds the threshold we can afford one more column
        const size_t threshold                   = floor_sqrt * ceil_sqrt;
        _number_columns                          = (self.size() >= threshold) ? ceil_sqrt : floor_sqrt;
        const size_t num_pe_in_incomplete_column = self.size() / _number_columns;
        size_t       row_num                     = row_index_in_complete_grid(self.rank());
        size_t       column_num                  = col_index_in_complete_grid(self.rank());
        _size_complete_rectangle                 = _number_columns * num_pe_in_incomplete_column;
        if (self.rank() >= _size_complete_rectangle) {
            row_num = self.rank() % _number_columns; // rank() is member of last incomplete row,
            // therefore append it to one of the first
        }
        {
            auto split_comm = self.split(static_cast<int>(row_num), self.rank_signed());
            _row_comm       = LevelCommunicator(split_comm.disown_mpi_communicator(), split_comm.root_signed(), true);
        }
        {
            auto split_comm = self.split(static_cast<int>(column_num), self.rank_signed());
            _column_comm    = LevelCommunicator(split_comm.disown_mpi_communicator(), split_comm.root_signed(), true);
        }
    }

    template <
        grid_plugin_helpers::MsgEnvelopeLevel envelop_level =
            grid_plugin_helpers::MsgEnvelopeLevel::source_and_destination,
        typename... Args>
    auto alltoallv_grid(Args... args) const {
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
        auto rowwise_recv_buf = rowwise_exchange<envelop_level>(send_buf, send_counts);
        return columnwise_exchange<envelop_level>(std::move(rowwise_recv_buf));
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

    [[nodiscard]] size_t row_index_in_complete_grid(size_t destination_rank) const {
        return destination_rank / _number_columns;
    }
    [[nodiscard]] size_t col_index_in_complete_grid(size_t destination_rank) const {
        return destination_rank % _number_columns;
    }
    [[nodiscard]] size_t get_destination_in_rowwise_exchange(size_t destination_rank) const {
        return col_index_in_complete_grid(destination_rank);
    }
    [[nodiscard]] size_t get_destination_in_colwise_exchange(size_t destination_rank) const {
        return row_index_in_complete_grid(destination_rank);
    }

    template <grid_plugin_helpers::MsgEnvelopeLevel envelope_level, typename SendBuffer, typename SendCounts>
    auto rowwise_exchange(SendBuffer const& send_buf, SendCounts const& send_counts) const {
        using namespace grid_plugin_helpers;
        auto const row_send_counts        = compute_row_send_counts(send_counts.underlying());
        auto       row_send_displacements = row_send_counts;
        std::exclusive_scan(
            row_send_counts.data(),
            row_send_counts.data() + row_send_counts.size(),
            row_send_displacements.data(),
            0ull
        );
        auto       index_displacements = row_send_displacements;
        int const  idx_last_elem       = _row_comm.size_signed() - 1;
        auto const total_send_count =
            static_cast<size_t>(row_send_displacements.data()[idx_last_elem] + row_send_counts.data()[idx_last_elem]);

        using value_type = typename SendBuffer::value_type;
        static_assert(std::is_same_v<value_type, double>);
        using MsgType = std::conditional_t<
            envelope_level == MsgEnvelopeLevel::no_envelope,
            MessageEnvelope<value_type, Destination>,
            MessageEnvelope<value_type, Source, Destination>>;
        DefaultContainerType<MsgType> rowwise_send_buf(total_send_count);
        size_t                        base_idx = 0;

        for (size_t i = 0; i < send_counts.size(); ++i) {
            int const    destination        = asserting_cast<int>(i);
            auto const   destination_in_row = get_destination_in_rowwise_exchange(asserting_cast<size_t>(i));
            size_t const send_count         = asserting_cast<size_t>(send_counts.underlying().data()[i]);
            for (std::size_t ii = 0; ii < send_count; ++ii) {
                auto       elem  = send_buf.data()[base_idx + ii];
                auto const idx   = index_displacements[destination_in_row]++;
                auto&      entry = rowwise_send_buf[static_cast<size_t>(idx)];
                entry            = MsgType(std::move(elem));
                entry.set_destination(destination
                ); // this has to be done independently of the envelope level, otherwise routing is not possible
                switch (envelope_level) {
                    case MsgEnvelopeLevel::no_envelope:
                        break;
                    case MsgEnvelopeLevel::source:                 // set source
                    case MsgEnvelopeLevel::source_and_destination: // set source
                        entry.set_source(this->to_communicator().rank_signed());
                        break;
                }
            }
            base_idx += send_count;
        }
        return _row_comm.alltoallv(
            kamping::send_buf(rowwise_send_buf),
            kamping::send_counts(row_send_counts),
            send_displs(row_send_displacements)
        );
    }

    template <
        grid_plugin_helpers::MsgEnvelopeLevel envelope_level,
        template <typename...>
        typename RowwiseRecvBuf,
        typename... IndirectMessageArgs>
    auto
    columnwise_exchange(RowwiseRecvBuf<grid_plugin_helpers::MessageEnvelope<IndirectMessageArgs...>>&& rowwise_recv_buf
    ) const {
        using namespace grid_plugin_helpers;
        using IndirectMessageType = MessageEnvelope<IndirectMessageArgs...>;
        using Payload             = typename IndirectMessageType::Payload;

        DefaultContainerType<int> send_counts(_column_comm.size(), 0);
        for (auto const& elem: rowwise_recv_buf) {
            ++send_counts[get_destination_in_colwise_exchange(asserting_cast<size_t>(elem.get_destination()))];
        }
        auto send_offsets = send_counts;
        std::exclusive_scan(send_counts.begin(), send_counts.end(), send_offsets.begin(), 0ull);
        auto send_displacements = send_offsets;

        using MsgType = MessageEnvelopeType<envelope_level, Payload>;
        DefaultContainerType<MsgType> colwise_send_buf(rowwise_recv_buf.size());
        for (auto& elem: rowwise_recv_buf) {
            auto const dst_in_column =
                get_destination_in_colwise_exchange(asserting_cast<size_t>(elem.get_destination()));
            auto& entry = colwise_send_buf[static_cast<std::size_t>(send_offsets[dst_in_column]++)];
            entry       = MsgType(std::move(elem.get_payload()));
            switch (envelope_level) {
                case MsgEnvelopeLevel::no_envelope:
                    break;
                case MsgEnvelopeLevel::source:
                    entry.set_source(elem.get_source());
                    break;
                case MsgEnvelopeLevel::source_and_destination: // set source
                    entry.set_source(this->to_communicator().rank_signed());
                    entry.set_destination(elem.get_destination());
                    break;
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
    size_t                                      _size_complete_rectangle;
    size_t                                      _number_columns;
    kamping::Communicator<DefaultContainerType> _row_comm;
    kamping::Communicator<DefaultContainerType> _column_comm;
};
} // namespace kamping::plugin
