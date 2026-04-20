#pragma once

#include <optional>
#include <utility>

#include <mpi.h>

#include "mpi/error.hpp"
#include "mpi/handle.hpp"

namespace mpi::experimental {

// ── GroupEquality ─────────────────────────────────────────────────────────────

/// @brief Result of `MPI_Group_compare`.
///
/// No ordering operator — the values are not comparable by magnitude.
enum class GroupEquality {
    Identical, ///< Same group object (same processes in same order).
    Similar,   ///< Same processes but in different order.
    Unequal,   ///< Different process sets.
};

// ── Forward declarations ──────────────────────────────────────────────────────

class group_view;
class group;

// ── group_accessors ───────────────────────────────────────────────────────────

/// @brief CRTP mixin providing read-only accessors for any group wrapper.
///
/// @tparam Derived Must implement `MPI_Group mpi_handle() const noexcept`.
template <typename Derived>
class group_accessors {
    [[nodiscard]] MPI_Group grp() const noexcept {
        return static_cast<Derived const*>(this)->mpi_handle();
    }

public:
    /// @return Number of processes in the group.
    [[nodiscard]] int size() const {
        int s = 0;
        MPI_Group_size(grp(), &s);
        return s;
    }

    /// @return Rank of the calling process in this group, or `std::nullopt` if
    ///         the calling process is not a member (`MPI_UNDEFINED`).
    [[nodiscard]] std::optional<int> rank() const {
        int r = MPI_UNDEFINED;
        MPI_Group_rank(grp(), &r);
        if (r == MPI_UNDEFINED) {
            return std::nullopt;
        }
        return r;
    }

    /// @return `true` if the calling process is a member of this group.
    [[nodiscard]] bool contains_self() const {
        return rank().has_value();
    }

    /// @brief Compare two groups.
    /// @return `GroupEquality::Identical` if same object, `Similar` if same
    ///         processes in different order, `Unequal` otherwise.
    [[nodiscard]] GroupEquality compare(group_view other) const;

    /// @return The underlying `MPI_Group` (escape hatch).
    [[nodiscard]] MPI_Group native() const noexcept { return grp(); }
};

// ── group_view ────────────────────────────────────────────────────────────────

/// @brief Non-owning wrapper around an `MPI_Group`.
///
/// Satisfies `convertible_to_mpi_handle<MPI_Group>`. Does not free the group
/// on destruction — use the owning `group` for that.
class group_view : public group_accessors<group_view> {
public:
    /// @brief Construct from a raw `MPI_Group`. The group must outlive this view.
    explicit group_view(MPI_Group g) noexcept : _group(g) {}

    /// @return The underlying `MPI_Group` (for `handle()` dispatch).
    [[nodiscard]] MPI_Group mpi_handle() const noexcept { return _group; }

private:
    MPI_Group _group;
};

// ── group ─────────────────────────────────────────────────────────────────────

/// @brief Owning RAII wrapper for `MPI_Group`.
///
/// Move-only. Calls `MPI_Group_free` on destruction unless the stored handle
/// is `MPI_GROUP_EMPTY` (the predefined empty group, which must not be freed).
///
/// Satisfies `convertible_to_mpi_handle<MPI_Group>`.
class group : public group_accessors<group> {
public:
    group(group const&)            = delete;
    group& operator=(group const&) = delete;

    /// @brief Move constructor — transfers ownership; moved-from holds `MPI_GROUP_EMPTY`.
    group(group&& o) noexcept : _group(std::exchange(o._group, MPI_GROUP_EMPTY)) {}

    /// @brief Move assignment — frees the current handle then transfers ownership.
    group& operator=(group&& o) noexcept {
        if (this != &o) {
            free_if_valid();
            _group = std::exchange(o._group, MPI_GROUP_EMPTY);
        }
        return *this;
    }

    /// @brief Free the group (unless it is `MPI_GROUP_EMPTY` or moved from).
    ~group() noexcept {
        free_if_valid();
    }

    /// @brief Return an owning wrapper around `MPI_GROUP_EMPTY`.
    ///
    /// The predefined empty group is never freed — `free_if_valid` skips it.
    [[nodiscard]] static group empty() noexcept {
        return group(MPI_GROUP_EMPTY, adopt_t{});
    }

    /// @brief Adopt an already-created `MPI_Group` handle (takes ownership).
    ///
    /// Use when you have called an MPI routine that returns an `MPI_Group` and
    /// you want RAII management. Do not pass `MPI_GROUP_EMPTY` (it is never
    /// freed anyway, but using `group::empty()` is clearer).
    [[nodiscard]] static group from_native(MPI_Group g) noexcept {
        return group(g, adopt_t{});
    }

    /// @brief Implicit conversion to a non-owning view.
    ///
    /// Allows passing a `group` wherever a `group_view` is expected without
    /// an explicit cast. The view borrows; it does not extend lifetime.
    operator group_view() const noexcept { return group_view{_group}; }

    /// @return The underlying `MPI_Group` (for `handle()` dispatch).
    [[nodiscard]] MPI_Group mpi_handle() const noexcept { return _group; }

private:
    struct adopt_t {};

    group(MPI_Group g, adopt_t) noexcept : _group(g) {}

    void free_if_valid() noexcept {
        if (_group != MPI_GROUP_EMPTY) {
            MPI_Group_free(&_group);
            _group = MPI_GROUP_EMPTY;
        }
    }

    MPI_Group _group = MPI_GROUP_EMPTY;
};

// ── group_accessors::compare (out-of-line — needs group_view definition) ──────

template <typename Derived>
GroupEquality group_accessors<Derived>::compare(group_view other) const {
    int result = MPI_IDENT;
    MPI_Group_compare(grp(), other.mpi_handle(), &result);
    switch (result) {
        case MPI_IDENT:   return GroupEquality::Identical;
        case MPI_SIMILAR: return GroupEquality::Similar;
        default:          return GroupEquality::Unequal;
    }
}

} // namespace mpi::experimental
