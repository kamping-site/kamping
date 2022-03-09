// This file is part of KaMPIng.
//
// Copyright 2021 The KaMPIng Authors
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

#include <string>

#include <mpi.h>

namespace kamping {
/// @brief The exception type used when an MPI call did not return MPI_SUCCESS.
/// When using this with KTHROW you should call it like this: `KTHROW_SPECIFIED(err == MPI_SUCCESS, "<MPI function that
/// failled>", MpiErrorException, err);`
class MpiErrorException : public std::exception {
public:
    /// @brief Constructs the exception
    /// @param message A custom error message.
    /// @param mpi_error_code The error code returned by the MPI call.
    MpiErrorException(std::string message, int mpi_error_code) : _mpi_error_code(mpi_error_code) {
        int  errorStringLen;
        char errorString[MPI_MAX_ERROR_STRING];
        int  err = MPI_Error_string(_mpi_error_code, errorString, &errorStringLen);
        if (err == MPI_SUCCESS) {
            _what = message + "\n Failed with the following error message: " + errorString;
        } else {
            _what = message + "\n Error message could not be retrieved";
        }
    }

    /// @brief Gets a description of this exception.
    /// @return A description of this exception.
    [[nodiscard]] char const* what() const noexcept final {
        return _what.c_str();
    }

    /// @brief Gets the error code returned by the MPI call.
    /// @return The error code returned by the MPI call.
    [[nodiscard]] int mpi_error_code() const {
        return _mpi_error_code;
    }

private:
    /// @brief The description of this exception.
    std::string _what;
    /// @brief The error code returned by the MPI call.
    int _mpi_error_code;
};

} // namespace kamping
