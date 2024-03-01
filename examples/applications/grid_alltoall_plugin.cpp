#include <numeric>
#include <thread>

#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/plugin_helpers.hpp"

using namespace ::kamping;

struct Source {
    int get_source() const {
        return source;
    }
    void set_source(int value) {
        source = value;
    }
    int source;
};

/// @brief Augments a plain message with its destination PE
template <typename PayloadType, typename... Attributes>
struct MessageEnvelope : public Attributes... {
    using Payload                                = PayloadType;
    static constexpr bool has_source_information = internal::type_list<Attributes...>::template contains<Source>;
    MessageEnvelope() {}

    MessageEnvelope(PayloadType payload) : _payload{payload}, _destination{0u} {}

    MessageEnvelope(PayloadType payload, int destination) : _payload{std::move(payload)}, _destination{destination} {}

    void set_destination(int destination) {
        _destination = destination;
    }

    [[nodiscard]] size_t get_destination() const {
        return asserting_cast<size_t>(_destination);
    }

    [[nodiscard]] int get_destination_signed() const {
        return _destination;
    }

    Payload& get_payload() {
        return _payload;
    }

    Payload const& get_payload() const {
        return _payload;
    }

    friend std::ostream& operator<<(std::ostream& out, MessageEnvelope const& msg) {
        out << "(destination: " << msg.get_destination();
        if constexpr (MessageEnvelope::has_source_information) {
            out << ", source: " << msg.get_source();
        }
        out << ", payload: " << msg.get_payload() << ")";
        return out;
    }

    Payload _payload;
    int     _destination;
};

/// @brief Represents a two dimensional grid communicator
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
class GridCommunicator : public plugins::PluginBase<Comm, DefaultContainerType, GridCommunicator> {
public:
    using LevelCommunicator = kamping::Communicator<DefaultContainerType>;

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

    template <bool augment_with_source = false, typename... Args>
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
        auto rowwise_recv_buf = rowwise_exchange<augment_with_source>(send_buf, send_counts);
        return (columnwise_exchange(std::move(rowwise_recv_buf)));
    }

public:
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

    template <bool augment_with_source = false, typename SendBuffer, typename SendCounts>
    auto rowwise_exchange(SendBuffer const& send_buf, SendCounts const& send_counts) const {
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
        using MsgType =
            std::conditional_t<augment_with_source, MessageEnvelope<value_type, Source>, MessageEnvelope<value_type>>;
        DefaultContainerType<MsgType> rowwise_send_buf(total_send_count);
        size_t                        base_idx = 0;

        for (size_t i = 0; i < send_counts.size(); ++i) {
            int const    destination        = asserting_cast<int>(i);
            auto const   destination_in_row = get_destination_in_rowwise_exchange(asserting_cast<size_t>(i));
            size_t const send_count         = asserting_cast<size_t>(send_counts.underlying().data()[i]);
            for (std::size_t ii = 0; ii < send_count; ++ii) {
                auto       elem                            = send_buf.data()[base_idx + ii];
                auto const idx                             = index_displacements[destination_in_row]++;
                rowwise_send_buf[static_cast<size_t>(idx)] = MsgType(elem, destination);
                if constexpr (augment_with_source) {
                    rowwise_send_buf[static_cast<size_t>(idx)].set_source(this->to_communicator().rank_signed());
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

    template <template <typename...> typename RowwiseRecvBuf, typename... IndirectMessageArgs>
    auto columnwise_exchange(RowwiseRecvBuf<MessageEnvelope<IndirectMessageArgs...>>&& rowwise_recv_buf) const {
        using IndirectMessageType = MessageEnvelope<IndirectMessageArgs...>;

        std::vector<IndirectMessageType> colwise_send_buf(rowwise_recv_buf.size());
        std::vector<int>                 send_counts(_column_comm.size(), 0);
        for (auto const& elem: rowwise_recv_buf) {
            ++send_counts[get_destination_in_colwise_exchange(elem.get_destination())];
        }
        auto send_offsets = send_counts;
        std::exclusive_scan(send_counts.begin(), send_counts.end(), send_offsets.begin(), 0ull);
        auto send_displacements = send_offsets;

        for (auto const& elem: rowwise_recv_buf) {
            auto const dst_in_column = get_destination_in_colwise_exchange(elem.get_destination());
            colwise_send_buf[static_cast<std::size_t>(send_offsets[dst_in_column]++)] = elem;
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

int main(int argc, char** argv) {
    // Call MPI_Init() and MPI_Finalize() automatically.
    Environment<> env(argc, argv);

    Communicator<std::vector, GridCommunicator> comm;
    comm.initialize_grid();
    std::vector<double> input(comm.size(), static_cast<double>(comm.rank_signed()) + 0.5);
    std::vector<int>    counts(comm.size(), 1);
    counts[0]     = 1;
    auto recv_buf = comm.alltoallv_grid<true>(send_buf(input), send_counts(counts));
    comm.barrier();

    if (comm.is_root(0)) {
        for (auto const& elem: recv_buf) {
            std::cout << elem << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
