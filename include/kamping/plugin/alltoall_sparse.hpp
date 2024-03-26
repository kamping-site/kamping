#include "kamping/checking_casts.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/ibarrier.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/named_parameter_filtering.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameter_types.hpp"
#include "kamping/p2p/iprobe.hpp"
#include "kamping/p2p/isend.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/plugin/plugin_helpers.hpp"
#include "kamping/request_pool.hpp"
#include "kamping/result.hpp"

/// @file
/// @brief File containing the SparseAlltoall plugin.

#pragma once
namespace kamping::plugin {

namespace sparse_alltoall {
/// @brief Class encapsulating a probed message that is ready to be received in a sparse alltoall exchange.
template <typename T, typename Communicator>
class ProbedMessage {
public:
    /// @brief Constructor of a probed message.
    ProbedMessage(Status&& status, Communicator const& comm) : _status(std::move(status)), _comm(comm) {}

    /// @brief Actually receive the probed message into a contiguous memory either provided by the user or allocated by
    /// the library.
    template <typename recv_value_type_tparam = T, typename... Args>
    auto recv(Args... args) const {
        KAMPING_CHECK_PARAMETERS(Args, KAMPING_REQUIRED_PARAMETERS(), KAMPING_OPTIONAL_PARAMETERS(recv_buf, recv_type));

        using default_recv_buf_type =
            decltype(kamping::recv_buf(alloc_new<
                                       typename Communicator::template default_container_type<recv_value_type_tparam>>)
            );
        auto&& recv_buf =
            internal::select_parameter_type_or_default<internal::ParameterType::recv_buf, default_recv_buf_type>(
                std::tuple(),
                args...
            )
                .template construct_buffer_or_rebind<Communicator::template default_container_type>();
        using recv_buf_type = std::remove_reference_t<decltype(recv_buf)>;

        using recv_value_type   = typename std::remove_reference_t<decltype(recv_buf)>::value_type;
        auto&& recv_type        = internal::determine_mpi_recv_datatype<recv_value_type, decltype(recv_buf)>(args...);
        auto   repack_recv_type = [&]() {
            // we cannot simply forward recv_type as kamping::recv_type as there are checks within recv() depending on
            // whether recv_type is caller provided or not
            if constexpr (internal::has_to_be_computed<decltype(recv_type)>) {
                return kamping::recv_type_out(recv_type.underlying());
            } else {
                return kamping::recv_type(recv_type.underlying());
            }
        };
        _comm.recv(
            kamping::recv_buf<recv_buf_type::resize_policy>(recv_buf.underlying()),
            repack_recv_type(),
            kamping::recv_count(recv_count_signed(recv_type.underlying())),
            kamping::source(_status.source_signed()),
            tag(_status.tag())
        );

        return internal::make_mpi_result<std::tuple<Args...>>(std::move(recv_buf), std::move(recv_type));
    }

    /// @brief Computes the size of the probed message depending on the used datatype.
    int recv_count_signed(MPI_Datatype datatype = MPI_DATATYPE_NULL) const {
        if (datatype == MPI_DATATYPE_NULL) {
            datatype = mpi_datatype<T>();
        }
        return _status.count_signed(datatype);
    }

    /// @brief Computes the size of the probed message depending on the used datatype.
    size_t recv_count(MPI_Datatype datatype = MPI_DATATYPE_NULL) const {
        return asserting_cast<size_t>(recv_count_signed(datatype));
    }

    /// @brief Returns the source of the probed message.
    int source_signed() const {
        return _status.source_signed();
    }

    /// @brief Returns the source of the probed message.
    size_t source() const {
        return _status.source();
    }

private:
    Status              _status;
    Communicator const& _comm;
};

/// @brief Parameter types used for the SparseAlltoall plugin.
enum class ParameterType {
    sparse_send_buf, ///< Tag used to represent a sparse send buffer, i.e. a buffer containing destination-message
                     ///< pairs.
    on_message ///< Tag used to represent a call back function operation on a \ref sparse_alltoall::ProbedMessage object
               ///< in alltoallv_sparse.
};

namespace internal {
/// @brief Predicate to check whether an argument provided to sparse_alltoall shall be discarded in the internal calls
/// to \ref Communicator::issend().
struct PredicateForSparseAlltoall {
    /// @brief Function to check whether an argument provided to \ref SparseAlltoall::alltoallv_sparse() shall be
    /// discarded in the send call.
    ///
    /// @tparam Arg Argument to be checked.
    /// @return \c True (i.e. discard) iff Arg's parameter_type is `sparse_send_buf`, `on_message` or `destination`.
    template <typename Arg>
    static constexpr bool discard() {
        using namespace kamping::internal;
        using ptypes_to_ignore = type_list<
            std::integral_constant<sparse_alltoall::ParameterType, sparse_alltoall::ParameterType::sparse_send_buf>,
            std::integral_constant<sparse_alltoall::ParameterType, sparse_alltoall::ParameterType::on_message>,
            std::integral_constant<kamping::internal::ParameterType, kamping::internal::ParameterType::tag>,
            std::integral_constant<kamping::internal::ParameterType, kamping::internal::ParameterType::destination>>;
        using ptype_entry = std::integral_constant<parameter_type_t<Arg>, parameter_type_v<Arg>>;
        return ptypes_to_ignore::contains<ptype_entry>;
    }
};

} // namespace internal

/// @brief Generates buffer wrapper based on the data in the sparse send buffer.
/// \param data is a container consisting of destination-message pairs. Each
/// such pair has to be decomposable via structured bindings with the first parameter being convertible to int and the
/// second parameter being the actual message to be sent for which we require the usual send_buf properties (i.e.,
/// either scalar types or existance  of a `data()` and `size()` member function and the exposure of a `value_type`))
///
/// @tparam Data Data type representing the element(s) to send.
/// @return Object referring to the storage containing the data elements to send.
template <typename Data>
auto sparse_send_buf(Data&& data) {
    using namespace kamping::internal;
    constexpr BufferOwnership ownership =
        std::is_rvalue_reference_v<Data&&> ? BufferOwnership::owning : BufferOwnership::referencing;

    return GenericDataBuffer<
        std::remove_reference_t<Data>,
        sparse_alltoall::ParameterType,
        sparse_alltoall::ParameterType::sparse_send_buf,
        BufferModifiability::constant,
        ownership,
        BufferType::in_buffer>(std::forward<Data>(data));
}

/// @brief Generates wrapper for an callback to be called on the probed messages in \ref
/// SparseAlltoall::alltoallv_sparse(). Its call operator has to accept a \ref ProbedMessage as sole parameter.
template <typename Callback>
auto on_message(Callback&& cb) {
    using namespace kamping::internal;
    constexpr BufferOwnership ownership =
        std::is_rvalue_reference_v<Callback&&> ? BufferOwnership::owning : BufferOwnership::referencing;

    constexpr BufferModifiability modifiability = std::is_const_v<std::remove_reference_t<Callback>>
                                                      ? BufferModifiability::constant
                                                      : BufferModifiability::modifiable;

    return GenericDataBuffer<
        std::remove_reference_t<Callback>,
        sparse_alltoall::ParameterType,
        sparse_alltoall::ParameterType::on_message,
        modifiability,
        ownership,
        BufferType::in_buffer>(std::forward<Callback>(cb));
}
} // namespace sparse_alltoall

/// @brief Plugin providing a sparse alltoall exchange method.
/// @see \ref SparseAlltoall::alltoallv_sparse() for more information.
template <typename Comm, template <typename...> typename DefaultContainerType>
class SparseAlltoall : public plugin::PluginBase<Comm, DefaultContainerType, SparseAlltoall> {
public:
    template <typename... Args>
    void alltoallv_sparse(Args... args) const;
};

/// @brief Sparse alltoall exchange using the NBX algorithm(Hoefler et al., "Scalable communication protocols for
/// dynamic sparse data", ACM Sigplan Noctices 45.5, 2010.)
///
/// This function provides a sparse interface for personalized all-to-all communication using
/// direct message exchange and thus achieving linear complexity in the number of messages to be sent (in
/// contrast to \c MPI_Alltoallv which exhibits complexity (at least) linear in the size of the communicator due to its
/// interface). To achieve this time complexity we can no longer rely on an array of size of the communicator for send
/// counts. Instead we use a sparse representation of the data to be sent.
///
/// The following parameters are required:
/// - \ref sparse_alltoall::sparse_send_buf() containing the messages to be sent to other ranks.
/// Differently from plain alltoallv, in alltoallv_sparse \c send_buf() encapsulates a container consisting of
/// destination-message pairs. Each such pair has to be decomposable via structured bindings with the first parameter
/// being convertible to int and the second parameter being the actual message to be sent for which we require the usual
/// send_buf properties (i.e., either scalar types or existance `data()` and `size()` member function and the exposure
/// of a `value_type`)). Messages of size 0 are not sent.
/// - \ref sparse_alltoall::on_message() containing a callback function `cb` which is responsible to
/// process the received messages via a \ref sparse_alltoall::ProbedMessage object. The callback function `cb` gets
/// called for each probed message ready to be received via `cb(probed_message)`. See \ref
/// sparse_alltoall::ProbedMessage for the member functions to be called on the object.
///
/// The following buffers are optional:
/// - \ref kamping::send_type() specifying the \c MPI datatype to use as send type. If omitted, the \c MPI datatype is
/// derived automatically based on each message's underlying \c value_type.
/// - \ref kamping::tag() the tag added to the directly exchanged messages. Defaults to the communicator's default tag
/// (\ref Communicator::default_tag()) if not present.
///
/// @tparam Args Automatically deducted template parameters.
/// @param args All required and any number of the optional parameters described above.
template <typename Comm, template <typename...> typename DefaultContainerType>
template <typename... Args>
void SparseAlltoall<Comm, DefaultContainerType>::alltoallv_sparse(Args... args) const {
    auto& self = this->to_communicator();
    // Get send_buf
    using send_buf_param_type =
        std::integral_constant<sparse_alltoall::ParameterType, sparse_alltoall::ParameterType::sparse_send_buf>;
    auto const& dst_message_container = kamping::internal::select_parameter_type<send_buf_param_type>(args...);
    using dst_message_container_type =
        typename std::remove_reference_t<decltype(dst_message_container.underlying())>::value_type;
    using message_type = typename std::tuple_element_t<1, dst_message_container_type>;
    // support message_type being a single element.
    using message_value_type = typename kamping::internal::
        ValueTypeWrapper<kamping::internal::has_data_member_v<message_type>, message_type>::value_type;

    // Get tag
    using default_tag_buf_type = decltype(kamping::tag(self.default_tag()));
    auto const&& tag_param     = kamping::internal::select_parameter_type_or_default<
        kamping::internal::ParameterType::tag,
        default_tag_buf_type>(std::tuple(self.default_tag()), args...);

    //// Get callback
    using on_message_param_type =
        std::integral_constant<sparse_alltoall::ParameterType, sparse_alltoall::ParameterType::on_message>;
    auto const& on_message_cb = kamping::internal::select_parameter_type<on_message_param_type>(args...);

    RequestPool<DefaultContainerType> request_pool;
    for (auto const& [dst, msg]: dst_message_container.underlying()) {
        auto send_buf = kamping::send_buf(msg);

        if (send_buf.size() > 0) {
            int       dst_       = dst; // cannot capture structured binding variable
            int const send_count = asserting_cast<int>(send_buf.size());
            auto      callable   = [&](auto... argsargs) {
                self.issend(
                    std::move(send_buf),
                    kamping::send_count(send_count),
                    destination(dst_),
                    request(request_pool.get_request()),
                    tag(tag_param.tag()),
                    std::move(argsargs)...
                );
            };
            std::apply(
                callable,
                filter_args_into_tuple<sparse_alltoall::internal::PredicateForSparseAlltoall>(args...)
            );
        }
    }

    Status  status;
    Request barrier_request(MPI_REQUEST_NULL);
    while (true) {
        bool const got_message = self.iprobe(status_out(status), tag(tag_param.tag()));
        if (got_message) {
            sparse_alltoall::ProbedMessage<message_value_type, Comm> probed_message{std::move(status), self};
            on_message_cb.underlying()(probed_message);
        }
        if (!barrier_request.is_null()) {
            if (barrier_request.test()) {
                break;
            }
        } else {
            if (request_pool.test_all()) {
                self.ibarrier(request(barrier_request));
            }
        }
    }
    self.barrier();
}

} // namespace kamping::plugin
