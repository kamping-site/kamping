#pragma once

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/checking_casts.hpp"

namespace kamping {

/// @brief Describes the equality of two groups.
/// - Identical :: The order and members of the two groups are the same.
/// - Similar :: Only the members are the same, the order is different.
/// - Unequal :: otherwise
enum class GroupEquality { Identical, Similar, Unequal, Invalid };

/// @brief A group of MPI processes.
class Group {
public:
    /// @brief Constructs a new group from an existing MPI group.
    Group(MPI_Group group, bool owning = false) : _group(group), _owns_group(owning) {}

    Group(Group const&)           = delete;
    Group operator=(Group const&) = delete;

    /// @brief Move constructor.
    Group(Group&& other) {
        this->_group      = other._group;
        other._owns_group = false;
        this->_owns_group = true;
    }

    /// @brief Move assignment.
    Group& operator=(Group&& other) {
        this->_group      = other._group;
        other._owns_group = false;
        this->_owns_group = true;
        return *this;
    }

    /// @brief Constructs the group associated with a communicator.
    Group(MPI_Comm comm) : _owns_group(true) {
        MPI_Comm_group(comm, &_group);
    }

    /// @brief Constructs the group associated with a communicator.
    template <typename Comm>
    Group(Comm const& comm) : Group(comm.mpi_communicator()) {}

    /// @brief Constructs an empty group.
    /// @return An empty group.
    [[nodiscard]] static Group empty() {
        return Group(MPI_GROUP_EMPTY);
    }

    /// @brief Constructs the group associated with the world communicator.
    /// @return The group associated with the world communicator.
    [[nodiscard]] static Group world() {
        return Group(MPI_COMM_WORLD);
    }

    /// @brief Default destructor, freeing the encapsulated group if owned.
    ~Group() {
        if (_owns_group) {
            MPI_Group_free(&_group);
        }
    }

    // TODO We need a safe tag for this -> move to another PR
    // template <typename Comm>
    // Comm create_comm(Comm const& comm) const {
    //     constexpr int safe_tag = 1337; // TODO
    //     MPI_Comm      new_comm;
    //     MPI_Comm_group_create(comm, _group, safe_tag, &new_comm);
    //     return Comm(new_comm);
    // }
    // TODO _excl, _incl, _range_incl, _range_excl, and translate_ranks

    /// @brief Compare two groups.
    /// @param other The group to compare with.
    /// @return The equality of the two groups (see \ref GroupEquality).
    GroupEquality compare(Group const& other) const {
        int result;
        MPI_Group_compare(_group, other._group, &result);

        switch (result) {
            case MPI_IDENT:
                return GroupEquality::Identical;
            case MPI_SIMILAR:
                return GroupEquality::Similar;
            case MPI_UNEQUAL:
                return GroupEquality::Unequal;
            default:
                KASSERT(false, "MPI_Group_compare returned an unknown value");
                return GroupEquality::Invalid;
        }
    }

    /// @Compare two groups.
    /// @param other The group to compare with.
    /// @return True if the groups are identical; see \ref GroupEquality. False otherwise.
    [[nodiscard]] bool is_identical(Group const& other) const {
        return compare(other) == GroupEquality::Identical;
    }

    /// @Compare two groups.
    /// @param other The group to compare with.
    /// @return True if the groups are similar; see \ref GroupEquality. False otherwise.
    [[nodiscard]] bool is_similar(Group const& other) const {
        return compare(other) == GroupEquality::Similar;
    }

    /// @Compare two groups.
    /// @param other The group to compare with.
    /// @return True if the groups are identical; see \ref GroupEquality. False otherwise.
    [[nodiscard]] bool has_same_ranks(Group const& other) const {
        auto const result = compare(other);
        return (result == GroupEquality::Identical) || (result == GroupEquality::Similar);
    }

    /// @brief Makes a group from the difference of two groups.
    /// @param other The other group.
    /// @return A group containing all the ranks of the first group that are not in the second group.
    Group difference(Group const& other) const {
        MPI_Group diff;
        MPI_Group_difference(_group, other._group, &diff);
        return Group(diff);
    }

    /// @brief Makes a group from the intersection of two groups.
    /// @param other The other group.
    /// @return A group containing only the ranks present in both groups.
    Group intersection(Group const& other) const {
        MPI_Group inter;
        MPI_Group_intersection(_group, other._group, &inter);
        return Group(inter);
    }

    /// @brief Makes a group from the union of two groups.
    /// @param other The other group.
    /// @return A group containing all ranks present in either of the two groups.
    /// @note The suffixing underscore is to avoid a name clash with the C++ keyword `union`.
    Group union_(Group const& other) const {
        MPI_Group un;
        MPI_Group_union(_group, other._group, &un);
        return Group(un);
    }

    /// @brief Get the number of ranks in the group.
    /// @return The number of ranks in the group.
    size_t size() const {
        int size;
        MPI_Group_size(_group, &size);
        return asserting_cast<size_t>(size);
    }

    /// @brief Get the rank of the calling process in the group.
    /// @return The rank of the calling process in the group.
    size_t rank() const {
        int rank;
        MPI_Group_rank(_group, &rank);
        return asserting_cast<size_t>(rank);
    }

private:
    MPI_Group _group;
    bool      _owns_group;
};

}; // namespace kamping
