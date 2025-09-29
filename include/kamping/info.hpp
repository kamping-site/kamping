#pragma once

#include <optional>
#include <string>
#include <string_view>

#include <mpi.h>

#include "kamping/checking_casts.hpp"

namespace kamping {

template <typename T>
std::string_view to_info_value_string(T const& value) = delete;

template <typename T>
std::optional<T> from_info_value_string(std::string_view value) = delete;

template <>
inline std::string_view to_info_value_string<bool>(bool const& value) {
    return value ? "true" : "false";
}
template <>
inline std::optional<bool> from_info_value_string<bool>(std::string_view value) {
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    return std::nullopt;
}

class Info {
public:
    Info() {
        MPI_Info_create(&_info);
    }

    Info(MPI_Info info, bool owning = false) : _info(info), _owning(owning) {}

    ~Info() {
        if (!_owning || _info == MPI_INFO_NULL) {
            return;
        }
        MPI_Info_free(&_info);
    }

    Info(Info const& other) {
        MPI_Info_dup(other._info, &_info);
    }

    Info& operator=(Info const& other) {
        if (_owning) {
            MPI_Info_free(&_info);
        }
        MPI_Info_dup(other._info, &_info);
        _owning = true; // no we own the info object, since it's a new one
        return *this;
    }

    Info(Info&& other) : _info(other._info), _owning(other._owning) {
        other._info = MPI_INFO_NULL;
    }

    Info operator=(Info&& other) {
        if (_owning) {
            MPI_Info_free(&_info);
        }
        _info       = other._info;
        _owning     = other._owning;
        other._info = MPI_INFO_NULL;
        return *this;
    }

    void set(std::string_view key, std::string_view value) {
        MPI_Info_set(_info, key.data(), value.data());
    }

    template <typename T>
    void set(std::string_view key, T const& value) {
        auto value_string = to_info_value_string(value);
        set(key, value_string);
    }

    bool contains(std::string_view key) const {
        int flag   = 0;
        int buflen = 0;
        MPI_Info_get_string(_info, key.data(), &buflen, nullptr, &flag);
        return flag;
    }

    std::optional<std::string> get(std::string_view key) const {
        int flag   = 0;
        int buflen = 0;
        MPI_Info_get_string(_info, key.data(), &buflen, nullptr, &flag);
        if (!flag) {
            return std::nullopt;
        }
        std::string value;
        value.resize(asserting_cast<std::size_t>(buflen));

        MPI_Info_get_string(_info, key.data(), &buflen, value.data(), &flag);
        return value;
    }

    template <typename T>
    std::optional<T> get(std::string_view key) const {
        auto value_string = get(key);
        if (!value_string) {
            return std::nullopt;
        }
        std::optional<T> value = from_info_value_string<T>(*value_string);
        return value;
    }

    void erase(std::string_view key) {
        MPI_Info_delete(_info, key.data());
    }

    std::size_t size() const {
        int nkeys = 0;
        MPI_Info_get_nkeys(_info, &nkeys);
        return asserting_cast<std::size_t>(nkeys);
    }

    MPI_Info& native() {
        return _info;
    }

    MPI_Info const& native() const {
        return _info;
    }

    // TODO add key-value iterator

private:
    MPI_Info _info;
    bool     _owning = true;
};
} // namespace kamping
