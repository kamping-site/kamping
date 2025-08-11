#pragma once

#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

#include <mpi.h>

#include "kamping/checking_casts.hpp"
#include "kamping/error_handling.hpp"

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

template <typename T, std::enable_if_t<std::is_integral_v<T> && std::is_same_v<T, bool>, int> = 0>
std::string_view to_info_value_string(T const& value) {
    return std::to_string(value);
}

template <typename T, std::enable_if_t<std::is_integral_v<T> && std::is_same_v<T, bool>, int> = 0>
std::optional<T> from_info_value_string(std::string_view value) {
    T result;
    auto [ptr, errcode] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (errcode == std::errc{}) {
        return result;
    }
    return std::nullopt;
}

class Info {
public:
    Info() {
        int err = MPI_Info_create(&_info);
        THROW_IF_MPI_ERROR(err, "MPI_Info_create");
    }

    Info(MPI_Info info, bool owning = false) : _info(info), _owning(owning) {}

    ~Info() {
        if (!_owning) {
            return;
        }
        MPI_Info_free(&_info);
    }

    Info(Info const& other) {
        int err = MPI_Info_dup(other._info, &_info);
        THROW_IF_MPI_ERROR(err, "MPI_Info_dup");
    }

    Info& operator=(Info const& other) {
        if (_owning) {
            MPI_Info_free(&_info);
        }
        int err = MPI_Info_dup(other._info, &_info);
        THROW_IF_MPI_ERROR(err, "MPI_Info_dup");
        _owning = true; // now we own the info object, since it's a new one
        return *this;
    }

    Info(Info&& other) : _info(other._info), _owning(other._owning) {
        other._info = MPI_INFO_NULL;
    }

    Info operator=(Info&& other) {
        if (_owning) {
            int err = MPI_Info_free(&_info);
            THROW_IF_MPI_ERROR(err, "MPI_Info_free");
        }
        _info       = other._info;
        _owning     = other._owning;
        other._info = MPI_INFO_NULL;
        return *this;
    }

    void set(std::string_view key, std::string_view value) {
        int err = MPI_Info_set(_info, key.data(), value.data());
        THROW_IF_MPI_ERROR(err, "MPI_Info_set");
    }

    template <typename T>
    void set(std::string_view key, T const& value) {
        auto value_string = to_info_value_string(value);
        set(key, value_string);
    }

    bool contains(std::string_view key) const {
        return get_value_length(key).has_value();
    }

    std::optional<std::string> get(std::string_view key) const {
        auto val_size = get_value_length(key);
        if (!val_size) {
            return std::nullopt;
        }
        std::string value;
        value.resize(asserting_cast<std::size_t>(*val_size));

        int flag = 0;
#if MPI_VERSION >= 4
        // From the standard: "In C, buflen includes the required space for the null terminator."
        int buflen = asserting_cast<int>(*val_size) + 1;
        int err = MPI_Info_get_string(_info, key.data(), &buflen, value.data(), &flag);
	THROW_IF_MPI_ERROR(err, "MPI_Info_get_string");
#else
        // From the standard: ""In C, valuelen should be one less than the amount of allocated space to allow for the
        // null terminator."
        int err = MPI_Info_get(_info, key.data(), asserting_cast<int>(*val_size), value.data(), &flag);
        THROW_IF_MPI_ERROR(err, "MPI_Info_get");
#endif
        KASSERT(flag == 1);
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
        int err = MPI_Info_delete(_info, key.data());
        THROW_IF_MPI_ERROR(err, "MPI_Info_delete");
    }

    std::size_t size() const {
        int nkeys = 0;
        int err   = MPI_Info_get_nkeys(_info, &nkeys);
        THROW_IF_MPI_ERROR(err, "MPI_Info_get_nkeys");
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
    /// without null-terminator
    std::optional<std::size_t> get_value_length(std::string_view key) const {
        int flag   = 0;
        int buflen = 0;
#if MPI_VERSION >= 4
        int err = MPI_Info_get_string(_info, key.data(), &buflen, nullptr, &flag);
        THROW_IF_MPI_ERROR(err, "MPI_Info_get_string");
        // From the standard: "In C, buflen includes the required space for the null terminator."
        buflen--;
#else
        // length returned does not include the end-of-string-character
        MPI_Info_get_valuelen(_info, key.data(), &buflen, &flag);
        THROW_IF_MPI_ERROR(err, "MPI_Info_get_valuelen");
#endif
        if (flag) {
            return asserting_cast<std::size_t>(buflen);
        }
        return std::nullopt;
    }

    MPI_Info _info;
    bool     _owning = true;
};
} // namespace kamping
