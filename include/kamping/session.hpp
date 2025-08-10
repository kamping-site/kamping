// This file is part of KaMPIng.
//
// Copyright 2021-2025 The KaMPIng Authors
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

#include <mpi.h>

#include "kamping/checking_casts.hpp"
#include "kamping/group.hpp"
#include "kamping/info.hpp"
#include "kamping/thread_levels.hpp"

namespace kamping {
namespace psets {
constexpr std::string_view world = "mpi://WORLD";
constexpr std::string_view self  = "mpi://SELF";
} // namespace psets

class PsetNameIterator;

class Session {
public:
    Session(ThreadLevel thread_level) {
        Info info;
        info.set("thread_level", thread_level);
        MPI_Session_init(info.native(), MPI_ERRORS_RETURN, &_session);
    }

    ~Session() {
        MPI_Session_finalize(&_session);
    }

    Info get_info() const {
        MPI_Info info_used = MPI_INFO_NULL;
        MPI_Session_get_info(_session, &info_used);
        Info info{info_used, /* owning = */ true};
        return info;
    }

    std::optional<Group> group_from_pset(std::string_view pset_name) const {
        KASSERT(pset_name_is_valid(pset_name));
        MPI_Group newgroup = MPI_GROUP_NULL;
        MPI_Group_from_session_pset(_session, pset_name.data(), &newgroup);
        if (newgroup == MPI_GROUP_NULL) {
            return std::nullopt;
        }
        Group group{newgroup, /* owning = */ true};
        return group;
    }

    Info pset_info(std::string_view pset_name) const {
        KASSERT(pset_name_is_valid(pset_name));
        MPI_Info pset_info = MPI_INFO_NULL;
        MPI_Session_get_pset_info(_session, pset_name.data(), &pset_info);
        Info info{pset_info, /* owning = */ true};
        return info;
    }

    std::size_t pset_size(std::string_view pset_name) const {
        KASSERT(pset_name_is_valid(pset_name));
        Info info = pset_info(pset_name);
        auto size = info.get<std::size_t>("size");
        KASSERT(size.has_value());
        return *size;
    }

    std::size_t get_num_psets(Info const& info) const {
        int npset_names = 0;
        MPI_Session_get_num_psets(_session, info.native(), &npset_names);
        return asserting_cast<std::size_t>(npset_names);
    }

    PsetNameIterator pset_names_begin(Info const& info) const;
    PsetNameIterator pset_names_begin() const;
    PsetNameIterator pset_names_end(Info const& info) const;
    PsetNameIterator pset_names_end() const;

    std::size_t get_num_psets() const {
        Info info_null{MPI_INFO_NULL, false};
        return get_num_psets(info_null);
    }

    std::string get_nth_pset(std::size_t n, Info const& info) const {
        KASSERT(n < get_num_psets(info));
        int pset_len = 0;
        // query length
        MPI_Session_get_nth_pset(_session, info.native(), asserting_cast<int>(n), &pset_len, nullptr);
        std::string pset_name;
        pset_name.resize(asserting_cast<std::size_t>(pset_len));
        MPI_Session_get_nth_pset(_session, info.native(), asserting_cast<int>(n), &pset_len, pset_name.data());
        return pset_name;
    }

    std::string get_nth_pset(std::size_t n) const {
        Info info_null{MPI_INFO_NULL, false};
        return get_nth_pset(n, info_null);
    }

    MPI_Session const& native() const {
        return _session;
    }

    MPI_Session& native() {
        return _session;
    }

private:
    bool pset_name_is_valid(std::string_view pset_name) const;

    MPI_Session _session;
};

class PsetNameIterator {
public:
    using value_type        = std::string;
    using difference_type   = std::ptrdiff_t;
    using reference         = value_type; // copies
    using pointer           = void;       // no pointer access
    using iterator_category = std::random_access_iterator_tag;

private:
    friend class Session;
    PsetNameIterator(Session const& session, Info const& info, std::size_t index)
        : _session(&session),
          _info(&info),
          _index(asserting_cast<difference_type>(index)) {}

public:
    std::string operator*() const {
        return _session->get_nth_pset(asserting_cast<std::size_t>(_index), *_info);
    }

    // Increment/decrement
    PsetNameIterator& operator++() {
        ++_index;
        return *this;
    }
    PsetNameIterator operator++(int) {
        auto tmp = *this;
        ++*this;
        return tmp;
    }
    PsetNameIterator& operator--() {
        --_index;
        return *this;
    }
    PsetNameIterator operator--(int) {
        auto tmp = *this;
        --*this;
        return tmp;
    }

    // Arithmetic
    PsetNameIterator& operator+=(difference_type n) {
        _index += n;
        return *this;
    }
    PsetNameIterator& operator-=(difference_type n) {
        _index -= n;
        return *this;
    }
    PsetNameIterator operator+(difference_type n) const {
        auto tmp = *this;
        return tmp += n;
    }
    friend PsetNameIterator operator+(difference_type n, PsetNameIterator it) {
        return it += n;
    }
    PsetNameIterator operator-(difference_type n) const {
        auto tmp = *this;
        return tmp -= n;
    }
    difference_type operator-(PsetNameIterator const& other) const {
        return _index - other._index;
    }

    // Element access
    std::string operator[](difference_type n) const {
        return *(*this + n);
    }

    // Comparisons
    bool operator==(PsetNameIterator const& other) const {
        return _index == other._index && _session->native() == other._session->native()
               && _info->native() == other._info->native();
    }
    bool operator!=(PsetNameIterator const& other) const {
        return !(*this == other);
    }
    bool operator<(PsetNameIterator const& other) const {
        return _index < other._index;
    }
    bool operator>(PsetNameIterator const& other) const {
        return other < *this;
    }
    bool operator<=(PsetNameIterator const& other) const {
        return !(other < *this);
    }
    bool operator>=(PsetNameIterator const& other) const {
        return !(*this < other);
    }

private:
    Session const*  _session;
    Info const*     _info;
    difference_type _index;
};

inline PsetNameIterator Session::pset_names_begin(Info const& info) const {
    return PsetNameIterator(*this, info, 0);
}

inline PsetNameIterator Session::pset_names_begin() const {
    Info info_null{MPI_INFO_NULL, false};
    return pset_names_begin(info_null);
}

inline PsetNameIterator Session::pset_names_end(Info const& info) const {
    return PsetNameIterator(*this, info, get_num_psets(info));
}

inline PsetNameIterator Session::pset_names_end() const {
    Info info_null{MPI_INFO_NULL, false};
    return pset_names_end(info_null);
}

inline bool Session::pset_name_is_valid(std::string_view pset_name) const {
    return std::find(pset_names_begin(), pset_names_end(), pset_name) != pset_names_end();
}

} // namespace kamping
