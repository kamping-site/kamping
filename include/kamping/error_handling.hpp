// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.:

#pragma once

/// @file
/// @brief Code for error handling.

#include <array>
#include <exception>
#include <string>

#include <kassert/kassert.hpp>
#include <mpi.h>

/// @brief Wrapper around THROWING_KASSERT for MPI errors.
///
/// Throws an MpiErrorException if the supplied error code is not \c MPI_SUCCESS.
///
/// The macro accepts 2 parameters:
/// 1. The error code returned by the MPI call.
/// 2. The MPI function that returned the error code.
#define THROW_IF_MPI_ERROR(error_code, function) \
    THROWING_KASSERT_SPECIFIED(                  \
        error_code == MPI_SUCCESS,               \
        #function << " failed!",                 \
        kamping::MpiErrorException,              \
        error_code                               \
    );

namespace kamping {

/// @brief The exception type used when an MPI call did not return \c MPI_SUCCESS.
///
/// When using this with THROWING_KASSERT you should call it like this: `THROWING_KASSERT_SPECIFIED(err == MPI_SUCCESS,
/// "<MPI function that failed> failed", MpiErrorException, err);`
class MpiErrorException : public std::exception {
public:
    /// @brief Constructs the exception
    /// @param message A custom error message.
    /// @param mpi_error_code The error code returned by the MPI call.
    MpiErrorException(std::string message, int mpi_error_code) : _mpi_error_code(mpi_error_code) {
        int                                    errorStringLen;
        std::array<char, MPI_MAX_ERROR_STRING> errorString;
        int err = MPI_Error_string(_mpi_error_code, errorString.data(), &errorStringLen);
        if (err == MPI_SUCCESS) {
            _what = message + "Failed with the following error message:\n" + std::string(errorString.data()) + "\n";
        } else {
            _what = message + "Error message could not be retrieved\n";
        }
    }

    /// @brief Gets a description of this exception.
    /// @return A description of this exception.
    [[nodiscard]] char const* what() const noexcept final {
        return _what.c_str();
    }

    /// @brief Gets the error code returned by the mpi call.
    /// @return The error code returned by the mpi call.
    [[nodiscard]] int mpi_error_code() const {
        return _mpi_error_code;
    }

    /// @brief Gets the error class corresponding to the error code.
    /// @return The error class corresponding to the error code.
    [[nodiscard]] int mpi_error_class() const {
        int error_class;
        MPI_Error_class(_mpi_error_code, &error_class);
        return error_class;
    }

private:
    /// @brief The description of this exception.
    std::string _what;
    /// @brief The error code returned by the MPI call.
    int _mpi_error_code;
};

} // namespace kamping
