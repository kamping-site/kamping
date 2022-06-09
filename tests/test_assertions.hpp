// This file is part of KaMPIng.
//
// Copyright 2021-2022 The KaMPIng Authors
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
/// @brief Turns all assertions into exceptions. This allows one to test if assertions fail.

#pragma once

#include <exception>

#include <kassert/kassert.hpp>

// Redefine KASSERT implementation to throw an exception
#undef KASSERT_KASSERT_HPP_KASSERT_IMPL
#define KASSERT_KASSERT_HPP_KASSERT_IMPL(type, expression, message, level) \
    KASSERT_KASSERT_HPP_THROWING_KASSERT_CUSTOM_IMPL(expression, testing::KassertTestingException, message)

namespace testing {
class KassertTestingException : public std::exception {
public:
    KassertTestingException(std::string message) : _message(std::move(message)) {}

    const char* what() const noexcept override {
        return _message.c_str();
    }

private:
    std::string _message;
};
} // namespace testing
