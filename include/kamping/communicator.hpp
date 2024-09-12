// This file is part of KaMPIng.
//
// Copyright 2021-2022 The KaMPIng Authors
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

#include <algorithm>
#include <cstddef>
#include <cstdlib>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "error_handling.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/environment.hpp"
#include "kamping/group.hpp"
#include "kamping/mpi_constants.hpp"
#include "kamping/mpi_datatype.hpp"
#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"
#include "kamping/rank_ranges.hpp"

namespace kamping {

// Needed by the plugin system to check if a plugin provides a callback function for MPI errors.
KAMPING_MAKE_HAS_MEMBER(mpi_error_handler)

/// @brief Wrapper for MPI communicator providing access to \c rank() and \c size() of the communicator. The \ref
/// Communicator is also access point to all MPI communications provided by KaMPIng.
/// @tparam DefaultContainerType The default container type to use for containers created by KaMPIng. Defaults to
/// std::vector.
/// @tparam Plugins Plugins adding functionality to KaMPIng. Plugins should be classes taking a <tt>Communicator</tt>
/// template parameter and can assume that they are castable to <tt>Communicator</tt> from which they can
/// call any function of <tt>kamping::Communicator</tt>. See <tt>test/plugin_tests.cpp</tt> for examples.
template <
    template <typename...> typename DefaultContainerType = std::vector,
    template <typename, template <typename...> typename>
    typename... Plugins>
class Communicator : public Plugins<Communicator<DefaultContainerType, Plugins...>, DefaultContainerType>... {
public:
    /// @brief Type of the default container type to use for containers created inside operations of this communicator.
    /// @tparam Args Arguments to the container type.
    template <typename... Args>
    using default_container_type = DefaultContainerType<Args...>;

    /// @brief Default constructor not specifying any MPI communicator and using \c MPI_COMM_WORLD by default.
    Communicator() : Communicator(MPI_COMM_WORLD) {}

    /// @brief Constructor where an MPI communicator has to be specified.
    /// @param comm MPI communicator that is wrapped by this \c Communicator.
    /// @param take_ownership Whether the Communicator should take ownership of comm, i.e. free it in the destructor.
    explicit Communicator(MPI_Comm comm, bool take_ownership = false) : Communicator(comm, 0, take_ownership) {}

    /// @brief Constructor where an MPI communicator and the default root have to be specified.
    /// @param comm MPI communicator that is wrapped by this \c Communicator.
    /// @param root Default root that is used by MPI operations requiring a root.
    /// @param take_ownership Whether the Communicator should take ownership of comm, i.e. free it in the destructor.
    explicit Communicator(MPI_Comm comm, int root, bool take_ownership = false)
        : _rank(get_mpi_rank(comm)),
          _size(get_mpi_size(comm)),
          _comm(comm),
          _default_tag(0),
          _owns_mpi_comm(take_ownership) {
        if (take_ownership) {
            KASSERT(comm != MPI_COMM_WORLD, "Taking ownership of MPI_COMM_WORLD is not allowed.");
        }
        this->root(root);
    }

    /// @brief Copy constructor that duplicates the MPI_Comm and takes ownership of the newly created one in the copy.
    /// @param other The Communicator to copy.
    Communicator(Communicator const& other)
        : _rank(other._rank),
          _size(other._size),
          _root(other._root),
          _default_tag(other._default_tag),
          _owns_mpi_comm(true) {
        MPI_Comm_dup(other._comm, &_comm);
    }

    /// @brief Move constructor
    /// @param other The Communicator to move.
    Communicator(Communicator&& other)
        : _rank(other._rank),
          _size(other._size),
          _comm(other._comm),
          _root(other._root),
          _default_tag(other._default_tag),
          _owns_mpi_comm(other._owns_mpi_comm) {
        // This prevents freeing the communicator twice (once in other and once in this)
        other._comm          = MPI_COMM_NULL;
        other._owns_mpi_comm = false;
    }

    /// @brief Destructor that frees the contained \c MPI_Comm if it is owned by the Communicator.
    virtual ~Communicator() {
        if (_owns_mpi_comm) {
            MPI_Comm_free(&_comm);
        }
    }

    /// @brief Move assignment operator.
    /// @param other The Communicator to move.
    Communicator& operator=(Communicator&& other) {
        swap(other);
        return *this;
    }

    /// @brief Copy assignment operator. Behaves according to the copy constructor.
    /// @param other The Communicator to copy.
    Communicator& operator=(Communicator const& other) {
        Communicator tmp(other);
        swap(tmp);
        return *this;
    }

    /// @brief Swaps the Communicator with another Communicator.
    /// @param other The Communicator to swap with.
    void swap(Communicator& other) {
        std::swap(_rank, other._rank);
        std::swap(_size, other._size);
        std::swap(_comm, other._comm);
        std::swap(_default_tag, other._default_tag);
        std::swap(_root, other._root);
        std::swap(_owns_mpi_comm, other._owns_mpi_comm);
    }

    /// @brief Terminates MPI execution environment (on all processes in this Communicator).
    /// Beware of MPI implementations who might terminate all processes, whether they are in this communicator or not.
    ///
    /// @param errorcode Error code to return to invoking environment.
    void abort(int errorcode = 1) const {
        [[maybe_unused]] int err = MPI_Abort(_comm, errorcode);
        this->mpi_error_hook(err, "MPI_Abort");
    }

    /// @brief Rank of the current MPI process in the communicator as <tt>int</tt>.
    /// @return Rank of the current MPI process in the communicator as <tt>int</tt>.
    [[nodiscard]] int rank_signed() const {
        return asserting_cast<int>(_rank);
    }

    /// @brief Rank of the current MPI process in the communicator as <tt>size_t</tt>.
    /// @return Rank of the current MPI process in the communicator as <tt>size_t</tt>.
    [[nodiscard]] size_t rank() const {
        return _rank;
    }

    /// @brief Number of MPI processes in this communicator as <tt>int</tt>.
    /// @return Number of MPI processes in this communicator <tt>int</tt>.
    [[nodiscard]] int size_signed() const {
        return asserting_cast<int>(_size);
    }

    /// @brief Number of MPI processes in this communicator as <tt>size_t</tt>.

    /// @return Number of MPI processes in this communicator as <tt>size_t</tt>.
    [[nodiscard]] size_t size() const {
        return _size;
    }

    /// @brief Number of NUMA nodes (different shared memory regions) in this communicator.
    /// This operation is expensive (communicator splitting and communication). You should cache the result if you need
    /// it multiple times.
    /// @return Number of compute nodes (hostnames) in this communicator.
    [[nodiscard]] size_t num_numa_nodes() const;

    /// @brief Get this 'processor's' name using \c MPI_Get_processor_name.
    /// @return This 'processor's' name. Nowadays, this oftentimes is the hostname.
    std::string processor_name() const {
        // Get the name of this node.
        int  my_len;
        char my_name[MPI_MAX_PROCESSOR_NAME];

        int ret = MPI_Get_processor_name(my_name, &my_len);
        this->mpi_error_hook(ret, "MPI_Get_processor_name");
        return std::string(my_name, asserting_cast<size_t>(my_len));
    }

    /// @brief MPI communicator corresponding to this communicator.
    /// @return MPI communicator corresponding to this communicator.
    [[nodiscard]] MPI_Comm mpi_communicator() const {
        return _comm;
    }

    /// @brief Disowns the wrapped MPI_Comm, i.e. it will not be freed in the destructor.
    /// @return MPI communicator corresponding to this communicator.
    MPI_Comm disown_mpi_communicator() {
        _owns_mpi_comm = false;
        return mpi_communicator();
    }

    /// @brief Set a new default tag used in point to point communication. The initial value is 0.
    void default_tag(int const default_tag) {
        THROWING_KASSERT(
            Environment<>::is_valid_tag(default_tag),
            "invalid tag " << default_tag << ", must be in range [0, " << Environment<>::tag_upper_bound() << "]"
        );
        _default_tag = default_tag;
    }

    /// @brief Default tag used in point to point communication. The initial value is 0.
    [[nodiscard]] int default_tag() const {
        return _default_tag;
    }

    /// @brief Set a new root for MPI operations that require a root.
    /// @param new_root The new default root.
    void root(int const new_root) {
        THROWING_KASSERT(
            is_valid_rank(new_root),
            "invalid root rank " << new_root << " in communicator of size " << size()
        );
        _root = asserting_cast<size_t>(new_root);
    }

    /// @brief Set a new root for MPI operations that require a root.
    /// @param new_root The new default root.
    void root(size_t const new_root) {
        THROWING_KASSERT(
            is_valid_rank(new_root),
            "invalid root rank " << new_root << " in communicator of size " << size()
        );
        _root = new_root;
    }

    /// @brief Default root for MPI operations that require a root as <tt>size_t</tt>.
    /// @return Default root for MPI operations that require a root as <tt>size_t</tt>.
    [[nodiscard]] size_t root() const {
        return _root;
    }

    /// @brief Default root for MPI operations that require a root as <tt>int</tt>.
    /// @return Default root for MPI operations that require a root as <tt>int</tt>.
    [[nodiscard]] int root_signed() const {
        return asserting_cast<int>(_root);
    }

    /// @brief Check if this rank is the root rank.
    /// @return Return \c true if this rank is the root rank.
    /// @param root The custom root's rank.
    [[nodiscard]] bool is_root(int const root) const {
        return rank() == asserting_cast<size_t>(root);
    }

    /// @brief Check if this rank is the root rank.
    /// @return Return \c true if this rank is the root rank.
    /// @param root The custom root's rank.
    [[nodiscard]] bool is_root(size_t const root) const {
        return rank() == root;
    }

    /// @brief Check if this rank is the root rank.
    /// @return Return \c true if this rank is the root rank.
    [[nodiscard]] bool is_root() const {
        return is_root(root());
    }

    /// @brief Split the communicator in different colors.
    /// @param color All ranks that have the same color will be in the same new communicator.
    /// @param key By default, ranks in the new communicator are determined by the underlying MPI library (if \c key is
    /// 0). Otherwise, ranks are ordered the same way the keys are ordered.
    /// @return \ref Communicator wrapping the newly split MPI communicator.
    [[nodiscard]] Communicator split(int const color, int const key = 0) const {
        MPI_Comm new_comm;
        MPI_Comm_split(_comm, color, key, &new_comm);
        return Communicator(new_comm, true);
    }

    /// @brief Split the communicator by the specified type (e.g., shared memory)
    ///
    /// @param type The only standard-conform value is \c MPI_COMM_TYPE_SHARED but your MPI implementation might support
    /// other types. For example: \c OMPI_COMM_TYPE_L3CACHE.
    [[nodiscard]] Communicator split_by_type(int const type) const {
        // MPI_COMM_TYPE_HW_GUIDED is only available starting with MPI-4.0
        // MPI_Info  info;
        // MPI_Info_create(&info);
        // MPI_Info_set(info, "mpi_hw_resource_type", "NUMANode");
        // auto ret = MPI_Comm_split_type(_comm, MPI_COMM_TYPE_HW_GUIDED, rank_signed(), info, &newcomm);

        MPI_Comm   new_comm;
        auto const ret = MPI_Comm_split_type(_comm, type, rank_signed(), MPI_INFO_NULL, &new_comm);
        this->mpi_error_hook(ret, "MPI_Comm_split_type");
        return Communicator(new_comm, true);
    }

    /// @brief Split the communicator into NUMA nodes.
    /// @return \ref Communicator wrapping the newly split MPI communicator. Each rank will be in the communicator
    /// corresponding to its NUMA node.
    [[nodiscard]] Communicator split_to_shared_memory() const {
        return split_by_type(MPI_COMM_TYPE_SHARED);
    }

    /// @brief Return the group associated with this communicator.
    /// @return The group associated with this communicator.
    [[nodiscard]] Group group() const {
        return Group(*this);
    }

    /// @brief Create subcommunicators.
    ///
    /// This method requires globally available information on the ranks in the subcommunicators.
    /// A rank \c r must know all other ranks which will be part of the subcommunicator to which \c r will belong.
    /// This information can be used by the MPI implementation to execute a communicator split more efficiently.
    /// The method must be called by all ranks in the communicator.
    ///
    /// @tparam Ranks Contiguous container storing integers.
    /// @param ranks_in_own_group Contains the ranks that will be part of this rank's new (sub-)communicator.
    /// All ranks specified in \c ranks_in_own_group must have an identical \c ranks_in_own_group argument. Furthermore,
    /// this set must not be empty.
    /// @return \ref Communicator wrapping the newly split MPI communicator.
    template <typename Ranks>
    [[nodiscard]] Communicator create_subcommunicators(Ranks const& ranks_in_own_group) const {
        static_assert(std::is_same_v<typename Ranks::value_type, int>, "Ranks must be of type int");
        KASSERT(
            ranks_in_own_group.size() > 0ull,
            "The set of ranks to include in the new subcommunicator must not be empty."
        );
        auto ranks_contain_own_rank = [&]() {
            return std::find(ranks_in_own_group.begin(), ranks_in_own_group.end(), rank()) != ranks_in_own_group.end();
        };
        KASSERT(ranks_contain_own_rank(), "The ranks to include in the new subcommunicator must contain own rank.");
        MPI_Group comm_group;
        MPI_Comm_group(_comm, &comm_group);
        MPI_Group new_comm_group;
        MPI_Group_incl(
            comm_group,
            asserting_cast<int>(ranks_in_own_group.size()),
            ranks_in_own_group.data(),
            &new_comm_group
        );
        MPI_Comm new_comm;
        MPI_Comm_create(_comm, new_comm_group, &new_comm);
        return Communicator(new_comm, true);
    }

    /// @brief Create (sub-)communicators using a sparse representation for the ranks contained in the
    /// subcommunicators.
    ///
    /// This split method requires globally available information on the ranks in the split communicators.
    /// A rank \c r must know all other ranks which will be part of the subcommunicator to which \c r will belong.
    /// This information can be used by the MPI implementation to execute a communicator split more efficiently.
    /// The method must be called by all ranks in the communicator.
    ///
    /// @param rank_ranges Contains the ranks that will be part of this rank's new (sub-)communicator in a sparse
    /// representation via rank ranges each consisting of (first rank, last rank and stride).
    /// All ranks specified in \c ranks_in_own_group must have
    /// an identical \c ranks_in_own_group argument. Furthermore, this set must not be empty.
    /// @return \ref Communicator wrapping the newly split MPI communicator.
    [[nodiscard]] Communicator create_subcommunicators(RankRanges const& rank_ranges) const {
        KASSERT(rank_ranges.size() > 0ull, "The set of ranks to include in the new subcommunicator must not be empty.");
        KASSERT(
            rank_ranges.contains(rank_signed()),
            "The ranks to include in the new subcommunicator must contain own rank."
        );
        MPI_Group comm_group;
        MPI_Comm_group(_comm, &comm_group);
        MPI_Group new_comm_group;
        MPI_Group_range_incl(comm_group, asserting_cast<int>(rank_ranges.size()), rank_ranges.get(), &new_comm_group);
        MPI_Comm new_comm;
        MPI_Comm_create(_comm, new_comm_group, &new_comm);
        return Communicator(new_comm, true);
    }

    ///@brief Compare this communicator with another given communicator. Uses \c MPI_Comm_compare internally.
    ///
    ///@param other_comm Communicator with which this communicator is compared.
    ///@return Return whether compared communicators are identical, congruent, similar or unequal.
    [[nodiscard]] CommunicatorComparisonResult compare(Communicator const& other_comm) const {
        int result;
        MPI_Comm_compare(_comm, other_comm.mpi_communicator(), &result);
        return static_cast<CommunicatorComparisonResult>(result);
    }

    /// @brief Convert a rank from this communicator to the rank in another communicator.
    /// @param rank The rank in this communicator
    /// @param other_comm The communicator to convert the rank to
    /// @return The rank in other_comm
    [[nodiscard]] int convert_rank_to_communicator(int const rank, Communicator const& other_comm) const {
        MPI_Group my_group;
        MPI_Comm_group(_comm, &my_group);
        MPI_Group other_group;
        MPI_Comm_group(other_comm._comm, &other_group);
        int rank_in_other_comm;
        MPI_Group_translate_ranks(my_group, 1, &rank, other_group, &rank_in_other_comm);
        return rank_in_other_comm;
    }

    /// @brief Convert a rank from another communicator to the rank in this communicator.
    /// @param rank The rank in other_comm
    /// @param other_comm The communicator to convert the rank from
    /// @return The rank in this communicator
    [[nodiscard]] int convert_rank_from_communicator(int const rank, Communicator const& other_comm) const {
        return other_comm.convert_rank_to_communicator(rank, *this);
    }

    /// @brief Computes a rank that is \c distance ranks away from this MPI thread's current rank and checks if this is
    /// valid rank in this communicator.
    ///
    /// The resulting rank is valid, iff it is at least zero and less than this communicator's size. The \c distance can
    /// be negative. Unlike \ref rank_shifted_cyclic(), this does not guarantee a valid rank but can indicate if the
    /// resulting rank is not valid.
    /// @param distance Amount current rank is decreased or increased by.
    /// @return Rank if rank is in [0, size of communicator) and ASSERT/EXCEPTION? otherwise.
    [[nodiscard]] size_t rank_shifted_checked(int const distance) const {
        int const result = rank_signed() + distance;
        THROWING_KASSERT(is_valid_rank(result), "invalid shifted rank " << result);
        return asserting_cast<size_t>(result);
    }

    /// @brief Computes a rank that is some ranks apart from this MPI thread's rank modulo the communicator's size.
    ///
    /// When we need to compute a rank that is greater (or smaller) than this communicator's rank, we can use this
    /// function. It computes the rank that is \c distance ranks appart. However, this function always returns a valid
    /// rank, as it computes the rank in a circular fashion, i.e., \f$ new\_rank=(rank + distance) \% size \f$.
    /// @param distance Distance of the new rank to the rank of this MPI thread.
    /// @return The circular rank that is \c distance ranks apart from this MPI threads rank.
    [[nodiscard]] size_t rank_shifted_cyclic(int const distance) const {
        int const capped_distance = distance % size_signed();
        return asserting_cast<size_t>((rank_signed() + capped_distance + size_signed()) % size_signed());
    }

    /// @brief Checks if a rank is a valid rank for this communicator, i.e., if the rank is in [0, size).
    /// @return \c true if rank in [0,size) and \c false otherwise.
    [[nodiscard]] bool is_valid_rank(int const rank) const {
        return rank >= 0 && rank < size_signed();
    }

    /// @brief Checks if a rank is a valid rank for this communicator, i.e., if the rank is in [0, size).
    /// @return \c true if rank in [0,size) and \c false otherwise.
    [[nodiscard]] bool is_valid_rank(size_t const rank) const {
        return rank < size();
    }

    /// @brief If <tt>error_code != MPI_SUCCESS</tt>, searchs the plugins for a \a public <tt>mpi_error_handler(const
    /// int error_code, std::string& callee)</tt> member. Searches the plugins front to back and calls the \a first
    /// handler found. If no handler is found, calls the default error hook. If error code is \c MPI_SUCCESS, does
    /// nothing.
    void mpi_error_hook(int const error_code, std::string const& callee) const {
        if (error_code != MPI_SUCCESS) {
            mpi_error_hook_impl<Plugins...>(error_code, callee);
        }
    }

    /// @brief Default MPI error callback. Depending on <tt>KASSERT_EXCEPTION_MODE</tt> either throws a \ref
    /// MpiErrorException if \c error_code != \c MPI_SUCCESS or fails an assertion.
    void mpi_error_default_handler(int const error_code, std::string const& function_name) const {
        THROWING_KASSERT_SPECIFIED(
            error_code == MPI_SUCCESS,
            function_name << " failed!",
            kamping::MpiErrorException,
            error_code
        );
    }

    template <typename... Args>
    void send(Args... args) const;

    template <typename... Args>
    void bsend(Args... args) const;

    template <typename... Args>
    void ssend(Args... args) const;

    template <typename... Args>
    void rsend(Args... args) const;

    template <typename... Args>
    auto isend(Args... args) const;

    template <typename... Args>
    auto ibsend(Args... args) const;

    template <typename... Args>
    auto issend(Args... args) const;

    template <typename... Args>
    auto irsend(Args... args) const;

    template <typename... Args>
    auto probe(Args... args) const;

    template <typename... Args>
    auto iprobe(Args... args) const;

    template <typename recv_value_type_tparam = kamping::internal::unused_tparam, typename... Args>
    auto recv(Args... args) const;

    template <typename recv_value_type_tparam, typename... Args>
    auto recv_single(Args... args) const;

    template <typename recv_value_type_tparam = kamping::internal::unused_tparam, typename... Args>
    auto try_recv(Args... args) const;

    template <typename recv_value_type_tparam = kamping::internal::unused_tparam, typename... Args>
    auto irecv(Args... args) const;

    template <typename... Args>
    auto alltoall(Args... args) const;

    template <typename... Args>
    auto alltoall_inplace(Args... args) const;

    template <typename... Args>
    auto alltoallv(Args... args) const;

    template <typename recv_value_type_tparam = kamping::internal::unused_tparam, typename... Args>
    auto scatter(Args... args) const;

    template <typename recv_value_type_tparam = kamping::internal::unused_tparam, typename... Args>
    auto scatter_single(Args... args) const;

    template <typename recv_value_type_tparam = kamping::internal::unused_tparam, typename... Args>
    auto scatterv(Args... args) const;

    template <typename... Args>
    auto reduce(Args... args) const;

    template <typename... Args>
    auto reduce_single(Args... args) const;

    template <typename... Args>
    auto scan(Args... args) const;

    template <typename... Args>
    auto scan_inplace(Args... args) const;

    template <typename... Args>
    auto scan_single(Args... args) const;

    template <typename... Args>
    auto exscan(Args... args) const;

    template <typename... Args>
    auto exscan_inplace(Args... args) const;

    template <typename... Args>
    auto exscan_single(Args... args) const;

    template <typename... Args>
    auto allreduce(Args... args) const;

    template <typename... Args>
    auto allreduce_inplace(Args... args) const;

    template <typename... Args>
    auto allreduce_single(Args... args) const;

    template <typename... Args>
    auto iallreduce(Args... args) const;

    template <typename... Args>
    auto gather(Args... args) const;

    template <typename... Args>
    auto gatherv(Args... args) const;

    template <typename... Args>
    auto allgather(Args... args) const;

    template <typename... Args>
    auto allgather_inplace(Args... args) const;

    template <typename... Args>
    auto allgatherv(Args... args) const;

    template <typename recv_value_type_tparam = kamping::internal::unused_tparam, typename... Args>
    auto bcast(Args... args) const;

    template <typename recv_value_type_tparam = kamping::internal::unused_tparam, typename... Args>
    auto bcast_single(Args... args) const;

    template <typename... Args>
    void barrier(Args... args) const;

    template <typename... Args>
    auto ibarrier(Args... args) const;

    template <typename Value>
    bool is_same_on_all_ranks(Value const& value) const;

private:
    /// @brief Compute the rank of the current MPI process computed using \c MPI_Comm_rank.
    /// @return Rank of the current MPI process in the communicator.
    size_t get_mpi_rank(MPI_Comm comm) const {
        THROWING_KASSERT(comm != MPI_COMM_NULL, "communicator must be initialized with a valid MPI communicator");

        int rank;
        MPI_Comm_rank(comm, &rank);
        return asserting_cast<size_t>(rank);
    }

    /// @brief Compute the number of MPI processes in this communicator using \c MPI_Comm_size.
    /// @return Size of the communicator.
    size_t get_mpi_size(MPI_Comm comm) const {
        THROWING_KASSERT(comm != MPI_COMM_NULL, "communicator must be initialized with a valid MPI communicator");

        int size;
        MPI_Comm_size(comm, &size);
        return asserting_cast<size_t>(size);
    }

    /// See \ref mpi_error_hook
    template <
        template <typename, template <typename...> typename>
        typename Plugin,
        template <typename, template <typename...> typename>
        typename... RemainingPlugins>
    void mpi_error_hook_impl(int const error_code, std::string const& callee) const {
        using PluginType = Plugin<Communicator<DefaultContainerType, Plugins...>, DefaultContainerType>;
        if constexpr (has_member_mpi_error_handler_v<PluginType, int, std::string const&>) {
            static_cast<PluginType const&>(*this).mpi_error_handler(error_code, callee);
        } else {
            if constexpr (sizeof...(RemainingPlugins) == 0) {
                mpi_error_hook_impl<void>(error_code, callee);
            } else {
                mpi_error_hook_impl<RemainingPlugins...>(error_code, callee);
            }
        }
    }

    template <typename = void>
    void mpi_error_hook_impl(int const error_code, std::string const& callee) const {
        mpi_error_default_handler(error_code, callee);
    }

protected:
    size_t   _rank; ///< Rank of the MPI process in this communicator.
    size_t   _size; ///< Number of MPI processes in this communicator.
    MPI_Comm _comm; ///< Corresponding MPI communicator.

    size_t _root;        ///< Default root for MPI operations that require a root.
    int    _default_tag; ///< Default tag value used in point to point communication.

    bool _owns_mpi_comm; ///< Whether the Communicator Objects owns the contained MPI_Comm, i.e. whether it is
                         ///< allowed to free it in the destructor.

}; // class communicator

/// @brief A basic KaMPIng Communicator that uses std::vector when creating new buffers.
using BasicCommunicator = Communicator<>;

/// @brief Gets a \c const reference to a \ref BasicCommunicator for \c MPI_COMM_WORLD.
///
/// Useful if you want access to KaMPIng's base functionality without keeping an instance of \ref Communicator or
/// constructing a new one on the fly.
///
/// @return A \c const reference to a \ref BasicCommunicator for \c MPI_COMM_WORLD.
inline BasicCommunicator const& comm_world() {
    // By using a static variable in a function here, this gets constructed on first use.
    static BasicCommunicator const comm_world;
    return comm_world;
}

/// @brief Gets the rank in \c MPI_COMM_WORLD as size_t.
///
/// @return The rank in \c MPI_COMM_WORLD.
inline size_t world_rank() {
    return comm_world().rank();
}

/// @brief Gets the rank in \c MPI_COMM_WORLD as int.
///
/// @return The rank in \c MPI_COMM_WORLD.
inline int world_rank_signed() {
    return comm_world().rank_signed();
}

/// @brief Gets the size of \c MPI_COMM_WORLD as size_t.
///
/// @return The size of \c MPI_COMM_WORLD.
inline size_t world_size() {
    return comm_world().size();
}

/// @brief Gets the size of \c MPI_COMM_WORLD as int.
///
/// @return The size of \c MPI_COMM_WORLD.
inline int world_size_signed() {
    return comm_world().size_signed();
}

} // namespace kamping
