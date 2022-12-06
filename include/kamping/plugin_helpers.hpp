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

namespace kamping {

/// @brief Helper class for using CRTP for mixins.
///
/// Taken from https://www.fluentcpp.com/2017/05/19/crtp-helper/
/// @tparam BaseClass Type of the class we want to add functionality to
/// @tparam MixinClass Type of the class template which inherits from \c CRTPHelper and adds functionality to \c
/// BaseClass.
template <typename BaseClass, template <typename> class MixinClass>
struct CRTPHelper {
    /// @return Reference to the underlying base class.
    BaseClass& underlying() {
        return static_cast<BaseClass&>(*this);
    }
    /// @return const-reference to the underlying base class.
    BaseClass const& underlying() const {
        return static_cast<BaseClass const&>(*this);
    }

private:
    CRTPHelper() {}               ///< private constructor
    friend MixinClass<BaseClass>; // this allows only the class inheriting from \c CRTPHelper to access the constructor.
};

} // namespace kamping
