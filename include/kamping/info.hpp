#pragma once

#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

#include <mpi.h>

#include "kamping/checking_casts.hpp"

namespace kamping {

template <typename T, typename Enable = void>
struct info_value_traits;

template <>
struct info_value_traits<bool> {
    using type = bool;
    static std::string_view to(bool value) {
        return value ? "true" : "false";
    }
    static inline std::optional<bool> from(std::string_view value) {
        if (value == "true") {
            return true;
        }
        if (value == "false") {
            return false;
        }
        return std::nullopt;
    }
};

template <typename T>
struct info_value_traits<T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>> {
    using type = T;
    static std::string_view to(T const& value) {
        return std::to_string(value);
    }

    static std::optional<T> from(std::string_view value) {
        T result;
        auto [ptr, errcode] = std::from_chars(value.data(), value.data() + value.size(), result);
        if (errcode == std::errc{}) {
            return result;
        }
        return std::nullopt;
    }
};

class Info {
public:
    Info() {
        MPI_Info_create(&_info);
    }

    Info(MPI_Info info, bool owning = false) : _info(info), _owning(owning) {}

    ~Info() {
        if (!_owning) {
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
        _owning = true; // now we own the info object, since it's a new one
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
        auto value_string = info_value_traits<T>::to(value);
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
        value.resize(asserting_cast<std::size_t>(buflen) - 1);

        MPI_Info_get_string(_info, key.data(), &buflen, value.data(), &flag);
        return value;
    }

    template <typename T>
    std::optional<T> get(std::string_view key) const {
        auto value_string = get(key);
        if (!value_string) {
            return std::nullopt;
        }
        std::optional<T> value = info_value_traits<T>::from(*value_string);
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
