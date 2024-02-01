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
#pragma once
namespace kamping::plugins {

/// @brief Helper class for using CRTP for mixins. Which are used to implement kamping plugins.
///
/// Taken from https://www.fluentcpp.com/2017/05/19/crtp-helper/
/// @tparam CommunicatorClass Type of the class we want to add functionality to, i.e. `kamping::Communicator`.
/// @tparam PluginClass Type of the plugin class template which inherits from \c PluginBase and adds functionality to \c
/// CommunicatorClass.
template <typename CommunicatorClass, template <typename> class PluginClass>
struct PluginBase {
private:
    /// @return Reference to the underlying Communicator class.
    CommunicatorClass& to_communicator() {
        return static_cast<CommunicatorClass&>(*this);
    }

    /// @return const-reference to the underlying Communicator class.
    CommunicatorClass const& to_communicator() const {
        return static_cast<CommunicatorClass const&>(*this);
    }

    PluginBase() {}                        ///< private constructor
    friend PluginClass<CommunicatorClass>; // this allows only the class inheriting from \c PluginBase to access the
                                           // functions of this class.
};

} // namespace kamping::plugins
