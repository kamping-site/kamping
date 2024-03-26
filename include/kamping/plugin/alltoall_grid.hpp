// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

/// @file
/// @brief Plugin to enable grid communication.

#include <numeric>

#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/plugin/plugin_helpers.hpp"

#pragma once

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

/// @brief Class representing a position within a logical two-dimensional processor grid.
struct GridPosition {
    size_t row_index; ///< Row position.
    size_t col_index; ///< Column position.
};

} // namespace grid_plugin_helpers

namespace grid {
/// @brief Object returned by \ref plugin::GridCommunicator::make_grid_communicator() representing a grid
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
    /// @param comm Communicator to be split into a two-dimensional grid.
    template <
        template <typename...> typename = DefaultContainerType,
        template <typename, template <typename...> typename>
        typename... Plugins>
    GridCommunicator(kamping::Communicator<DefaultContainerType, Plugins...> const& comm)
        : _size_of_orig_comm{comm.size()},
          _rank_in_orig_comm{comm.rank()} {
        double const sqrt       = std::sqrt(comm.size());
        size_t const floor_sqrt = static_cast<size_t>(std::floor(sqrt));
        size_t const ceil_sqrt  = static_cast<size_t>(std::ceil(sqrt));
        // We want to ensure that #columns + 1 >= #rows >= #columns.
        // Therefore, use floor(sqrt(comm.size())) columns unless we have enough PEs to begin another row when using
        // ceil(sqrt(comm.size()) columns.
        size_t const threshold                      = floor_sqrt * ceil_sqrt;
        _num_columns                                = (comm.size() >= threshold) ? ceil_sqrt : floor_sqrt;
        size_t const num_ranks_in_incomplete_column = comm.size() / _num_columns;
        auto [row, col]          = pos_in_complete_grid(comm.rank()); // assume that we have a complete grid,
        _size_complete_rectangle = _num_columns * num_ranks_in_incomplete_column;
        if (comm.rank() >= _size_complete_rectangle) {
            row = comm.rank() % _num_columns; // rank() is member of last incomplete row,
            // therefore append it to one of the first
        }
        {
            auto split_comm = comm.split(static_cast<int>(row), comm.rank_signed());
            _row_comm       = LevelCommunicator(split_comm.disown_mpi_communicator(), split_comm.root_signed(), true);
        }
        {
            auto split_comm = comm.split(static_cast<int>(col), comm.rank_signed());
            _column_comm    = LevelCommunicator(split_comm.disown_mpi_communicator(), split_comm.root_signed(), true);
        }
    }

    /// @brief Indirect two dimensional grid based personalized alltoall exchange.
    /// The following parameters are required:
    /// - \ref kamping::send_buf() containing the data that is sent to each rank. The size of this buffer has to be at
    /// least the sum of the send_counts argument.
    /// - \ref kamping::send_counts() containing the number of elements to send to each rank.
    /// - \ref kamping::send_displs() containing the number of elements to send to each rank.
    ///
    /// The following parameters are optional:
    /// - \ref kamping::send_displs() containing the offsets of the messages in send_buf. The `send_counts[i]` elements
    /// starting at `send_buf[send_displs[i]]` will be sent to rank `i`. If omitted, this is calculated as the exclusive
    /// prefix-sum of `send_counts`.
    ///
    /// Internally, each element in the send buffer is wrapped in an envelope to facilitate the indirect routing. The
    /// envelope consists at least of the destination PE of each element but can be extended to also hold the
    /// source PE of the element. The caller can specify whether they want to keep this information also in the output
    /// via the \tparam envelope_level.
    ///
    /// @tparam envelope_level Determines the contents of the envelope of each returned element (no_envelope = use the
    /// actual data type of an exchanged element, source = augment the actual data type with the source PE,
    /// source_and_destination = argument the actual data type with the source and destination PE).
    /// @tparam Args Automatically deducted template parameters.
    /// @param args All required and any number of the optional buffers described above.
    /// @returns
    template <MessageEnvelopeLevel envelope_level = MessageEnvelopeLevel::no_envelope, typename... Args>
    auto alltoallv_with_envelope(Args... args) const {
        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(send_buf, send_counts),
            KAMPING_OPTIONAL_PARAMETERS(send_displs)
        );
        // Get send_buf
        auto const& send_buf =
            internal::select_parameter_type<internal::ParameterType::send_buf>(args...).construct_buffer_or_rebind();
        // Get send_counts
        auto const& send_counts = internal::select_parameter_type<internal::ParameterType::send_counts>(args...)
                                      .template construct_buffer_or_rebind<DefaultContainerType>();

        using default_send_displs_type = decltype(kamping::send_displs_out(alloc_new<DefaultContainerType<int>>));
        auto&& send_displs =
            internal::select_parameter_type_or_default<internal::ParameterType::send_displs, default_send_displs_type>(
                std::tuple(),
                args...
            )
                .template construct_buffer_or_rebind<DefaultContainerType>();
        // Calculate send_displs if necessary
        constexpr bool do_calculate_send_displs = internal::has_to_be_computed<decltype(send_displs)>;
        if constexpr (do_calculate_send_displs) {
            send_displs.resize_if_requested([&]() { return _size_of_orig_comm; });
            std::exclusive_scan(send_counts.data(), send_counts.data() + _size_of_orig_comm, send_displs.data(), 0);
        }
        auto rowwise_recv_buf = rowwise_exchange<envelope_level>(send_buf, send_counts, send_displs);
        return columnwise_exchange<envelope_level>(std::move(rowwise_recv_buf));
    }

    /// @brief Indirect two dimensional grid based personalized alltoall exchange.
    /// The following parameters are required:
    /// - \ref kamping::send_buf() containing the data that is sent to each rank. The size of this buffer has to be at
    /// least the sum of the send_counts argument.
    ///
    /// - \ref kamping::send_counts() containing the number of elements to send to each rank.
    ///
    /// The following parameters are optional:
    /// - \ref kamping::send_displs() containing the offsets of the messages in send_buf. The `send_counts[i]` elements
    /// starting at `send_buf[send_displs[i]]` will be sent to rank `i`. If omitted, this is calculated as the exclusive
    /// prefix-sum of `send_counts`.
    ///
    /// - \ref kamping::recv_counts() containing the number of elements to receive from each rank.
    ///
    /// - \ref kamping::recv_buf() containing a buffer for the output. Afterwards, this buffer will contain
    ///
    /// the data received as specified for send_buf. The buffer will be resized according to the buffer's
    /// kamping::BufferResizePolicy. If resize policy is kamping::BufferResizePolicy::no_resize, the buffer's underlying
    /// storage must be large enough to store all received elements.
    ///
    /// Internally, \ref alltoallv_with_envelope() is called.
    ///
    /// @tparam envelope_level Determines the contents of the envelope of each returned element (no_envelope = use the
    /// actual data type of an exchanged element, source = augment the actual data type with the source PE,
    /// source_and_destination = argument the actual data type with the source and destination PE).
    /// @tparam Args Automatically deducted template parameters.
    /// @param args All required and any number of the optional buffers described above.
    /// @return Result type wrapping the output buffer and recv_counts (if requested).
    template <typename... Args>
    auto alltoallv(Args... args) const {
        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(send_buf, send_counts),
            KAMPING_OPTIONAL_PARAMETERS(send_displs, recv_buf, recv_counts, recv_displs)
        );
        constexpr MessageEnvelopeLevel envelope_level = MessageEnvelopeLevel::source;
        // get send_buf
        auto& send_buf                = internal::select_parameter_type<internal::ParameterType::send_buf>(args...);
        using send_value_type         = typename std::remove_reference_t<decltype(send_buf)>::value_type;
        using default_recv_value_type = std::remove_const_t<send_value_type>;
        // get send_counts
        auto const& send_counts = internal::select_parameter_type<internal::ParameterType::send_counts>(args...)
                                      .template construct_buffer_or_rebind<DefaultContainerType>();

        using default_send_displs_type = decltype(kamping::send_displs_out(alloc_new<DefaultContainerType<int>>));
        auto&& send_displs =
            internal::select_parameter_type_or_default<internal::ParameterType::send_displs, default_send_displs_type>(
                std::tuple(),
                args...
            )
                .template construct_buffer_or_rebind<DefaultContainerType>();
        // Calculate send_displs if necessary
        constexpr bool do_calculate_send_displs = internal::has_to_be_computed<decltype(send_displs)>;
        if constexpr (do_calculate_send_displs) {
            send_displs.resize_if_requested([&]() { return _size_of_orig_comm; });
            std::exclusive_scan(send_counts.data(), send_counts.data() + _size_of_orig_comm, send_displs.data(), 0);
        }

        // perform the actual message exchange
        auto grid_recv_buf = alltoallv_with_envelope<envelope_level>(
            std::move(send_buf),
            kamping::send_counts(send_counts.underlying()),
            kamping::send_displs(send_displs.underlying())
        );

        // post-processing (fixing ordering problems etc.)
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
            KASSERT(recv_counts.size() >= _size_of_orig_comm, "Recv counts buffer is not large enough.", assert::light);
            Span recv_counts_span(recv_counts.data(), recv_counts.size());
            std::fill(recv_counts_span.begin(), recv_counts_span.end(), 0);

            for (size_t i = 0; i < grid_recv_buf.size(); ++i) {
                ++recv_counts_span[grid_recv_buf[i].get_source()];
            }
        } else {
            KASSERT(recv_counts.size() >= this->size(), "Recv counts buffer is not large enough.", assert::light);
        }

        // Get recv displs
        using default_recv_displs_type = decltype(kamping::recv_displs_out(alloc_new<DefaultContainerType<int>>));
        auto&& recv_displs =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_displs, default_recv_displs_type>(
                std::tuple(),
                args...
            )
                .template construct_buffer_or_rebind<DefaultContainerType>();
        constexpr bool do_calculate_recv_displs = internal::has_to_be_computed<decltype(recv_displs)>;

        if constexpr (do_calculate_recv_displs) {
            recv_displs.resize_if_requested([&]() { return _size_of_orig_comm; });
            KASSERT(recv_displs.size() >= _size_of_orig_comm, "Recv displs buffer is not large enough.", assert::light);
            Span recv_displs_span(recv_displs.data(), recv_displs.size());
            Span recv_counts_span(recv_counts.data(), recv_counts.size());
            std::exclusive_scan(recv_counts_span.begin(), recv_counts_span.end(), recv_displs_span.begin(), 0);
        }

        // get recv_buf
        using default_recv_buf_type =
            decltype(kamping::recv_buf(alloc_new<DefaultContainerType<default_recv_value_type>>));
        auto&& recv_buf =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
                std::tuple(),
                args...
            )
                .template construct_buffer_or_rebind<DefaultContainerType>();

        write_recv_buffer(grid_recv_buf, recv_buf, recv_counts, recv_displs);

        return internal::make_mpi_result<std::tuple<Args...>>(
            std::move(send_displs),
            std::move(recv_buf),
            std::move(recv_counts),
            std::move(recv_displs)
        );
    }

private:
    template <typename GridRecvBuffer, typename RecvBuffer, typename RecvCounts, typename RecvDispls>
    void write_recv_buffer(
        GridRecvBuffer const& grid_recv_buffer,
        RecvBuffer&           recv_buf,
        RecvCounts const&     recv_counts,
        RecvDispls const&     recv_displs
    ) const {
        auto write_pos = recv_displs.underlying();
        Span write_pos_span(write_pos.data(), write_pos.size());
        auto compute_required_recv_buf_size = [&]() {
            Span recv_counts_span(recv_counts.data(), recv_counts.size());
            return asserting_cast<size_t>(write_pos_span.back() + recv_counts_span.back());
        };

        recv_buf.resize_if_requested(compute_required_recv_buf_size);
        KASSERT(
            recv_buf.size() >= compute_required_recv_buf_size(),
            "Recv buffer is not large enough to hold all received elements.",
            assert::light
        );

        Span recv_buf_span(recv_buf.data(), recv_buf.size());
        Span grid_recv_buf_span(grid_recv_buffer.data(), grid_recv_buffer.size());
        for (auto const& elem: grid_recv_buf_span) {
            size_t const pos   = asserting_cast<size_t>(write_pos_span[elem.get_source()]++);
            recv_buf_span[pos] = std::move(elem.get_payload());
        }
    }

    template <typename SendCounts>
    [[nodiscard]] auto compute_row_send_counts(SendCounts const& send_counts) const {
        DefaultContainerType<int> row_send_counts(_row_comm.size(), 0);
        for (size_t i = 0; i < send_counts.size(); ++i) {
            size_t const destination_pe = get_destination_in_rowwise_exchange(i);
            row_send_counts.data()[destination_pe] += send_counts.data()[i];
        }
        return row_send_counts;
    }

    [[nodiscard]] grid_plugin_helpers::GridPosition pos_in_complete_grid(size_t rank) const {
        return grid_plugin_helpers::GridPosition{rank / _num_columns, rank % _num_columns};
    }

    [[nodiscard]] size_t get_destination_in_rowwise_exchange(size_t destination_rank) const {
        return pos_in_complete_grid(destination_rank).col_index;
    }

    [[nodiscard]] size_t get_destination_in_colwise_exchange(size_t destination_rank) const {
        return pos_in_complete_grid(destination_rank).row_index;
    }

    template <MessageEnvelopeLevel envelope_level, typename SendBuffer, typename SendCounts, typename SendDispls>
    auto
    rowwise_exchange(SendBuffer const& send_buf, SendCounts const& send_counts, SendDispls const& send_displs) const {
        using namespace grid_plugin_helpers;
        auto const                row_send_counts = compute_row_send_counts(send_counts.underlying());
        DefaultContainerType<int> row_send_displs(_row_comm.size());
        Span                      row_send_counts_span(row_send_counts.data(), row_send_counts.size());
        Span                      row_send_displs_span(row_send_displs.data(), row_send_displs.size());
        std::exclusive_scan(
            row_send_counts_span.begin(),
            row_send_counts_span.end(),
            row_send_displs_span.begin(),
            int(0)
        );
        auto       index_displacements = row_send_displs;
        auto const total_send_count    = static_cast<size_t>(row_send_displs_span.back() + row_send_counts_span.back());

        using value_type = typename SendBuffer::value_type;
        using MsgType    = std::conditional_t<
            envelope_level == MessageEnvelopeLevel::no_envelope,
            MessageEnvelope<value_type, Destination>,
            MessageEnvelope<value_type, Source, Destination>>;
        DefaultContainerType<MsgType> rowwise_send_buf(total_send_count);

        for (size_t i = 0; i < send_counts.size(); ++i) {
            size_t const send_count = asserting_cast<size_t>(send_counts.underlying().data()[i]);
            if (send_count == 0) {
                continue;
            }
            int const    destination        = asserting_cast<int>(i);
            auto const   destination_in_row = get_destination_in_rowwise_exchange(asserting_cast<size_t>(i));
            size_t const cur_displacement   = asserting_cast<size_t>(send_displs.data()[i]);
            for (std::size_t ii = 0; ii < send_count; ++ii) {
                auto       elem  = send_buf.data()[cur_displacement + ii];
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
        }
        return _row_comm.alltoallv(
            kamping::send_buf(rowwise_send_buf),
            kamping::send_counts(row_send_counts),
            kamping::send_displs(row_send_displs)
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
        std::exclusive_scan(send_counts_span.begin(), send_counts_span.end(), send_offsets_span.begin(), int(0));
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
    size_t                                      _num_columns;
    kamping::Communicator<DefaultContainerType> _row_comm;
    kamping::Communicator<DefaultContainerType> _column_comm;
};
} // namespace grid

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
class GridCommunicator : public plugin::PluginBase<Comm, DefaultContainerType, GridCommunicator> {
public:
    /// @brief Returns a \ref kamping::plugin::GridCommunicator.
    auto make_grid_communicator() const {
        return grid::GridCommunicator<DefaultContainerType>(this->to_communicator());
    }
};

} // namespace kamping::plugin
