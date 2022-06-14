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
// <https://www.gnu.org/licenses/>.

/// @file
/// @brief Redefines the KASSERT macro such that assertions throw exceptions instead of aborting the process.
/// This is needed because GoogleTest does not support death tests in a multithreaded program (MPI spawns multiple
/// threads).
///
/// *NOTE THAT THIS HEADER MUST BE INCLUDED BEFORE ANY OTHER KAMPING HEADERS* since it redefines the KASSERT macro.
/// This must happen before the preprocessor substitutes the macro invocations.

#pragma once

#include <exception>

#include <kassert/kassert.hpp>

// Redefine KASSERT implementation to throw an exception
#undef KASSERT_KASSERT_HPP_KASSERT_IMPL
#define KASSERT_KASSERT_HPP_KASSERT_IMPL(type, expression, message, level) \
    KASSERT_KASSERT_HPP_THROWING_KASSERT_CUSTOM_IMPL(expression, testing::KassertTestingException, message)

namespace testing {
// Dummy exception class used for remapping assertions to throwing exceptions.
class KassertTestingException : public std::exception {
public:
    // Assertion message (no expression decomposition)
    KassertTestingException(std::string message) : _message(std::move(message)) {}

    const char* what() const noexcept override {
        return _message.c_str();
    }

private:
    std::string _message;
};
} // namespace testing
