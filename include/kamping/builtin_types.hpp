// This file is part of KaMPIng.
//
// Copyright 2021-2024, 2026 The KaMPIng Authors
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
/// @brief Mapping of C++ datatypes to builtin MPI types.

#pragma once
#include "kamping/types/builtin_types.hpp"

namespace kamping {
using types::builtin_type;
using types::category_has_to_be_committed;
using types::is_builtin_type_v;
using types::TypeCategory;
} // namespace kamping
