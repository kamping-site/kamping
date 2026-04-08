#pragma once

#include <exception>
#include <optional>
#include <string>

#include <mpi.h>

namespace kamping::core {
struct mpi_error : public std::exception {
private:
    int                                error_code_;
    mutable std::optional<std::string> error_string_;

public:
    mpi_error() : mpi_error(MPI_ERR_UNKNOWN) {}
    mpi_error(int error_code) : error_code_(error_code) {}

    [[nodiscard]] int error_code() const {
        return error_code_;
    }

    [[nodiscard]] int error_class() const {
        int error_class = 0;
        int err         = MPI_Error_class(error_code_, &error_class);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        return error_class;
    }

    [[nodiscard]] mpi_error as_class_error() const {
        return mpi_error(this->error_class());
    }

    [[nodiscard]] std::string error_string() const {
        std::string error_string;
        error_string.resize(MPI_MAX_ERROR_STRING);
        int error_string_len = 0;
        int err              = MPI_Error_string(error_code_, error_string.data(), &error_string_len);
        if (err != MPI_SUCCESS) {
            throw mpi_error(err);
        }
        error_string.resize(static_cast<std::size_t>(error_string_len));
        return error_string;
    }

    [[nodiscard]] char const* what() const noexcept final {
        if (!error_string_) {
            try {
                error_string_ = error_string();
            } catch (...) {
                error_string_ = "<failed to get error string>";
            }
        }
        return error_string_->data();
    }
};
} // namespace kamping::core
