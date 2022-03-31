// This file is part of KaMPI.ng.
//
// Copyright 2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "helpers_for_testing.hpp"

#include "kamping/kassert.hpp"

int main(int /* argc */, char** /* argv */) {
#if defined(FORBIDDEN_AND)
    KASSERT(false && false);
#elif defined(FORBIDDEN_OR)
    KASSERT(false || false);
#else
    // If none of the above sections is active, this file will compile successfully.
#endif
}
