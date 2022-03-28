#pragma once

#include "kamping/checking_casts.hpp"
#include "kamping/mpi_datatype.hpp"

namespace kamping {

/// @brief Helper class for using CRTP for mixins.
///
/// Taken from https://www.fluentcpp.com/2017/05/19/crtp-helper/
/// @tparam BaseClass Type of the class we want to add functionality to
/// @tparam MixinClass Type of the class template which inherits from \c CRTPHelper and adds functionality to \c
/// BaseClass.
template <typename BaseClass, template <typename> class MixinClass>
struct CRTPHelper {
private:
    friend MixinClass<BaseClass>; // this allows only the class inheriting from \c CRTPHelper to access the members.
    /// @return Reference to the underlying base class.
    BaseClass& underlying() {
        return static_cast<BaseClass&>(*this);
    }

    /// @return const-reference to the underlying base class.
    BaseClass const& underlying() const {
        return static_cast<BaseClass const&>(*this);
    }

    CRTPHelper() {} ///< private constructor

    /// @brief Check if all sizes are equal using \b communication (one \c MPI_Gather).
    ///
    /// @param local_size Size at PE that is compared with all other sizes for equality.
    /// @return \c true if all \c local_size are equal and \c false otherwise.
    bool check_equal_sizes(size_t local_size) const {
        std::vector<size_t> result(asserting_cast<size_t>(this->underlying().size()), 0);
        MPI_Gather(
            &local_size, 1, mpi_datatype<size_t>(), result.data(), 1, mpi_datatype<size_t>(), this->underlying().root(),
            this->underlying().mpi_communicator());
        return std::equal(result.begin() + 1, result.end(), result.begin());
    }
};

} // namespace kamping
