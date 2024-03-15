#include <numeric>

#include <mpi.h>
// We have to include mpi.h /before/ mpi-ext.h order for OMPI_DECLSPEC to be defined.
#include <mpi-ext.h>

#include "kamping/checking_casts.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/group.hpp"
#include "kamping/plugin/plugin_helpers.hpp"

#if not(defined(MPIX_ERR_PROC_FAILED)) or not(defined(MPIX_ERR_PROC_FAILED_PENDING)) or not(defined(MPIX_ERR_REVOKED))
    #pragma message \
        "MPIX_ERR_PROC_FAILED, MPIX_ERR_PROC_FAILED_PENDING, or MPIX_ERR_REVOKED not defined. You need a MPI implementation which supports fault-tolerance to enable the FaultTolerance."
#endif

namespace kamping {
/// @brief Base class for all exceptions thrown by the FaultTolerance plugin.
/// Means, that either a process failed or the communicator was revoked.
class MPIFailureDetected : public std::exception {
public:
    /// @brief Returns an explanatory string.
    char const* what() const noexcept override {
        return "A MPI process failed or the communicator was revoked.";
    }
};

/// @brief Thrown when a process failure prevented the completetion of the MPI operation.
class MPIProcFailedError : public MPIFailureDetected {
public:
    /// @brief Returns an explanatory string.
    char const* what() const noexcept override {
        return "A process failure prevented the completetion of the MPI operation.";
    }
};

/// @brief Thrown when a potential sender matching a non-blocking wildcard source receive has failed.
class MPIProcFailedPendingError : public MPIFailureDetected {
public:
    /// @brief Returns an explanatory string.
    char const* what() const noexcept override {
        return "A potential sender matching a non-blocking wildcard source receive has failed.";
    }
};

/// @brief Thrown when the communicator was revoked.
class MPIRevokedError : public MPIFailureDetected {
public:
    /// @brief Returns an explanatory string.
    char const* what() const noexcept override {
        return "The communicator was revoked.";
    }
};
} // namespace kamping

namespace kamping::plugin {

/// @brief A plugin implementing a wrapper around the User-Level Failure-Mitigation (ULFM) feature of the upcoming MPI 4
/// standard. This plugin and the accompanying example is tested with OpenMPI 5.0.2.
template <typename Comm, template <typename...> typename DefaultContainerType>
class UserLevelFailureMitigation : public plugin::PluginBase<Comm, DefaultContainerType, UserLevelFailureMitigation> {
public:
    /// @brief Default constructor; sets the error handler of MPI_COMM_WORLD (!) to MPI_ERRORS_RETURN.
    /// Although the standard allows setting the error handler for only a specific communicator; neither MPICH nor
    /// OpenMPI currently (March 2024) support this.
    UserLevelFailureMitigation() {
        // MPI_Comm_set_errhandler(_comm(), MPI_ERRORS_RETURN);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    }

    /// @brief Revokes the current communicator.
    void revoke() {
        auto const ret = MPIX_Comm_revoke(_comm());
        this->to_communicator().mpi_error_hook(ret, "MPIX_Comm_revoke");
    }

    /// @brief Acknowledges that the application intends to ignore the effect of currently known failures on wildcard
    /// receive completions and agreement return values.
    /// @param num_to_ack The number of failures to acknowledge.
    /// @return The <i>overall</i> number of failures acknowledged.
    uint32_t ack_failed(uint32_t const num_to_ack) {
        int        num_acked;
        auto const ret = MPIX_Comm_ack_failed(_comm(), asserting_cast<int>(num_to_ack), &num_acked);
        this->to_communicator().mpi_error_hook(ret, "MPIX_Comm_ack_failed");
        return asserting_cast<uint32_t>(num_acked);
    }

    /// @brief Gets the number of acknowledged failures.
    /// @return The number of acknowledged failures.
    [[nodiscard]] uint32_t num_ack_failed() {
        return ack_failed(0);
    }

    /// @brief Acknowledge all failures.
    /// @return The <i>overall</i> number of failures acknowledged.
    uint32_t ack_all_failed() {
        return ack_failed(this->mpi_communicator().size());
    }

    /// @brief Creates a new communicator from this communicator, excluding the failed processes.
    /// @return The new communicator.
    [[nodiscard]] Comm shrink() {
        MPI_Comm   newcomm;
        auto const ret = MPIX_Comm_shrink(_comm(), &newcomm);
        this->to_communicator().mpi_error_hook(ret, "MPIX_Comm_shrink");
        return Comm(newcomm);
    }

    /// @brief Agrees on a flag from all live processes and distributes the result back to all live processes, even
    /// after process failures.
    /// @param flag The flag to agree on.
    /// @return The bitwise AND over the contributed input values of \c flag.
    [[nodiscard]] int agree(int flag) {
        auto const ret = MPIX_Comm_agree(_comm(), &flag);
        this->to_communicator().mpi_error_hook(ret, "MPIX_Comm_agree");
        return flag;
    }

    /// @brief Agrees on a boolean flag from all live processes and distributes the result back to all live processes,
    /// even after process failures.
    /// @param flag The flag to agree on.
    /// @return The bitwise AND over the contributed input values of \c flag.
    [[nodiscard]] bool agree(bool flag) {
        return agree(static_cast<bool>(flag));
    }

    /// @brief Obtains the group of currently failed processes.
    /// @return The group of currently failed processes.
    [[nodiscard]] Group get_failed() {
        MPI_Group  failed_group;
        auto const ret = MPIX_Comm_get_failed(_comm(), &failed_group);
        this->to_communicator().mpi_error_hook(ret, "MPIX_Comm_failure_get_acked");
        return Group(failed_group);
    }

    /// @brief Checks if this communicator has been revoked.
    /// @return True if the communicator has been revoked, false otherwise.
    [[nodiscard]] bool is_revoked() {
        int is_revoked;
        MPIX_Comm_is_revoked(_comm(), &is_revoked);
        return static_cast<bool>(is_revoked);
    }

    /// Overwrite the on-MPI-error handler to throw appropriate exceptions for then hardware faults happened.
    void mpi_error_handler(int const ret, [[maybe_unused]] std::string const& callee) const {
        KASSERT(ret != MPI_SUCCESS, "MPI error handler called with MPI_SUCCESS", assert::light);
        switch (ret) {
            case MPIX_ERR_PROC_FAILED:
                throw MPIProcFailedError();
                break;
            case MPIX_ERR_PROC_FAILED_PENDING:
                throw MPIProcFailedPendingError();
                break;
            case MPIX_ERR_REVOKED:
                throw MPIRevokedError();
                break;
            default:
                this->to_communicator().mpi_error_default_handler(ret, callee);
        }
    }

private:
    auto _comm() {
        return this->to_communicator().mpi_communicator();
    }
};

} // namespace kamping::plugin
