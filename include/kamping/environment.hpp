// This file is part of KaMPIng.
//
// Copyright 2022-2023 The KaMPIng Authors
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
/// @brief Wrapper for MPI functions that don't require a communicator.

#pragma once

#include <vector>

#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"
#include "kamping/error_handling.hpp"
#include "kamping/span.hpp"

namespace kamping {

namespace internal {
/// @brief A global list of MPI data types registered to KaMPIng.
inline std::vector<MPI_Datatype> registered_mpi_types;
} // namespace internal

/// @brief Configuration for the behavior of the constructors and destructor of \ref kamping::Environment.
enum class InitMPIMode {
    InitFinalize,           ///< Call \c MPI_Init in the constructor of \ref Environment.
    NoInitFinalize,         ///< Do not call \c MPI_Init in the constructor of \ref Environment.
    InitFinalizeIfNecessary ///< Call \c MPI_Init in the constructor of \ref Environment if \c MPI_Init has not been
                            ///< called before. Call \c MPI_Finalize in the destructor of \ref Environment if \c
                            ///< MPI_Init was called in the constructor.
};

/// @brief Wrapper for MPI functions that don't require a communicator. If the template parameter `init_finalize_mode`
/// is set to \ref InitMPIMode::InitFinalize (default), \c MPI_Init is called in the constructor, and
/// \c MPI_Finalize is called in the destructor.
///
/// Note that \c MPI_Init and \c MPI_Finalize are global, meaning that if they are called on an Environment object they
/// must not be called again in any Environment object (or directly vie the \c MPI_* calls).
template <InitMPIMode init_finalize_mode = InitMPIMode::InitFinalize>
class Environment {
public:
    /// @brief Calls MPI_Init with arguments.
    ///
    /// @param argc Number of arguments.
    /// @param argv The arguments.
    Environment(int& argc, char**& argv) {
        if constexpr (init_finalize_mode == InitMPIMode::InitFinalize) {
            init(argc, argv);
        } else if constexpr (init_finalize_mode == InitMPIMode::InitFinalizeIfNecessary) {
            if (!initialized()) {
                init(argc, argv);
                _finalize = true;
            } else {
                _finalize = false;
            }
        }
    }

    /// @brief Calls MPI_Init without arguments.
    Environment() {
        if constexpr (init_finalize_mode == InitMPIMode::InitFinalize) {
            init();
        } else if constexpr (init_finalize_mode == InitMPIMode::InitFinalizeIfNecessary) {
            if (!initialized()) {
                init();
                _finalize = true;
            } else {
                _finalize = false;
            }
        }
    }

    /// @brief Calls MPI_Init without arguments and doesn't check whether MPI_Init has already been called.
    void init_unchecked() const {
        KASSERT(!initialized(), "Trying to call MPI_Init twice");
        [[maybe_unused]] int err = MPI_Init(NULL, NULL);
        THROW_IF_MPI_ERROR(err, MPI_Init);
    }

    /// @brief Calls MPI_Init with arguments and doesn't check whether MPI_Init has already been called.
    ///
    /// @param argc Number of arguments.
    /// @param argv The arguments.
    void init_unchecked(int& argc, char**& argv) const {
        KASSERT(!initialized(), "Trying to call MPI_Init twice");
        [[maybe_unused]] int err = MPI_Init(&argc, &argv);
        THROW_IF_MPI_ERROR(err, MPI_Init);
    }

    /// @brief Calls MPI_Init without arguments. Checks whether MPI_Init has already been called first.
    void init() const {
        if (initialized()) {
            return;
        }
        [[maybe_unused]] int err = MPI_Init(NULL, NULL);
        THROW_IF_MPI_ERROR(err, MPI_Init);
    }

    /// @brief Calls MPI_Init with arguments. Checks whether MPI_Init has already been called first.
    ///
    /// @param argc Number of arguments.
    /// @param argv The arguments.
    void init(int& argc, char**& argv) const {
        if (initialized()) {
            return;
        }
        [[maybe_unused]] int err = MPI_Init(&argc, &argv);
        THROW_IF_MPI_ERROR(err, MPI_Init);
    }

    /// @brief Calls MPI_Finalize and frees all registered MPI data types.
    ///
    /// Even if you chose InitMPIMode::InitFinalize, you might want to call this function: As MPI_Finalize could
    /// potentially return an error, this function can be used if you want to be able to handle that error. Otherwise
    /// the destructor will call MPI_Finalize and not throw on any errors returned.
    void finalize() const {
        KASSERT(!finalized(), "Trying to call MPI_Finalize twice");
        free_registered_mpi_types();
        [[maybe_unused]] int err = MPI_Finalize();
        THROW_IF_MPI_ERROR(err, MPI_Finalize);
    }

    /// @brief Checks whether MPI_Init has been called.
    ///
    /// @return \c true if MPI_Init has been called, \c false otherwise.
    bool initialized() const {
        int                  result;
        [[maybe_unused]] int err = MPI_Initialized(&result);
        THROW_IF_MPI_ERROR(err, MPI_Initialized);
        return result == true;
    }

    /// @brief Checks whether MPI_Finalize has been called.
    ///
    /// @return \c true if MPI_Finalize has been called, \c false otherwise.
    bool finalized() const {
        int                  result;
        [[maybe_unused]] int err = MPI_Finalized(&result);
        THROW_IF_MPI_ERROR(err, MPI_Finalized);
        return result == true;
    }

    /// @brief Returns the elapsed time since an arbitrary time in the past.
    ///
    /// @return The elapsed time in seconds.
    static double wtime() {
        return MPI_Wtime();
    }

    /// @brief Returns the resolution of Environment::wtime().
    ///
    /// @return The resolution in seconds.
    static double wtick() {
        return MPI_Wtick();
    }

    /// @brief The upper bound on message tags defined by the MPI implementation.
    /// @return The upper bound for tags.
    [[nodiscard]] static int tag_upper_bound() {
        int* tag_ub;
        int  flag;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &flag);
        KASSERT(flag, "Could not retrieve MPI_TAG_UB");
        return *tag_ub;
    }

    /// @brief Checks if the given tag is a valid message tag.
    /// @return Whether the tag is valid.
    [[nodiscard]] static bool is_valid_tag(int tag) {
        return tag >= 0 && tag <= tag_upper_bound();
    }

    /// @brief Register a new MPI data type to KaMPIng that will be freed when using Environment to finalize MPI.
    /// @param type The MPI data type to register.
    static void register_mpi_type(MPI_Datatype type) {
        internal::registered_mpi_types.push_back(type);
    }

    /// @brief Commit an MPI data type (without registering it with KaMPIng).
    static void commit(MPI_Datatype type) {
        int err = MPI_Type_commit(&type);
        THROW_IF_MPI_ERROR(err, MPI_Type_commit);
    }

    /// @brief Free an MPI data type.
    static void free(MPI_Datatype type) {
        int err = MPI_Type_free(&type);
        THROW_IF_MPI_ERROR(err, MPI_Type_free);
    }

    /// @brief Commit an MPI data type and register it with KaMPIng.
    /// @see commit()
    /// @see register_type()
    static void commit_and_register(MPI_Datatype type) {
        commit(type);
        register_mpi_type(type);
    }

    /// @brief Free all registered MPI data types.
    ///
    /// Only call this when you no longer want to use any MPI data types created by KaMPIng as other KaMPIng functions
    /// will assume the created types still exist.
    static void free_registered_mpi_types() {
        for (auto type: internal::registered_mpi_types) {
            if (type != MPI_DATATYPE_NULL) {
                MPI_Type_free(&type);
            }
        }
        internal::registered_mpi_types.clear();
    }

    static inline size_t const bsend_overhead = MPI_BSEND_OVERHEAD; ///< Provides an upper bound on the additional
                                                                    ///< memory required by buffered send operations.

    /// @brief Attach a buffer to use for buffered send operations to the environment.
    ///
    /// @tparam T The type of the buffer.
    /// @param buffer The buffer. The user is responsible for allocating the buffer, attaching it, detaching it and
    /// freeing the memory after detaching. For convenience, the buffer may be a span of any type, but the type is
    /// ignored by MPI.
    template <typename T>
    void buffer_attach(Span<T> buffer) {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(!has_buffer_attached, "You may only attach one buffer at a time.");
#endif
        int err = MPI_Buffer_attach(buffer.data(), asserting_cast<int>(buffer.size() * sizeof(T)));
        THROW_IF_MPI_ERROR(err, MPI_Buffer_attach);
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        has_buffer_attached = true;
#endif
    }

    /// @todo: maybe we want buffer_allocate. This would require us keeping track of the buffer internally.

    /// @brief Detach a buffer attached via \ref buffer_attach().
    ///
    /// @tparam T The type of the span to return. Defaults to \c std::byte.
    /// @return A span pointing to the previously attached buffer. The type of the returned span can be controlled via
    /// the parameter \c T. The pointer to the buffer stored internally by MPI is reinterpreted accordingly.
    template <typename T = std::byte>
    Span<T> buffer_detach() {
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        KASSERT(has_buffer_attached, "There is currently no buffer attached.");
#endif
        void* buffer_ptr;
        int   buffer_size;
        int   err = MPI_Buffer_detach(&buffer_ptr, &buffer_size);
        THROW_IF_MPI_ERROR(err, MPI_Buffer_detach);
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
        has_buffer_attached = false;
#endif
        KASSERT(
            static_cast<size_t>(buffer_size) % sizeof(T) == size_t{0},
            "The buffer size is not a multiple of the size of T."
        );

        // convert the returned pointer and size to a span of type T
        return Span<T>{static_cast<T*>(buffer_ptr), asserting_cast<size_t>(buffer_size) / sizeof(T)};
    }

    /// @brief Calls MPI_Finalize if finalize() has not been called before. Also frees all registered MPI data types.
    ~Environment() {
        if (init_finalize_mode == InitMPIMode::InitFinalize
            || (init_finalize_mode == InitMPIMode::InitFinalizeIfNecessary && _finalize)) {
            bool is_already_finalized = false;
            try {
                is_already_finalized = finalized();
            } catch (MpiErrorException&) {
                // Just kassert. We can't throw exceptions in the destructor.

                // During testing we sometimes force KASSERT to throw exceptions. During the resulting stack unwinding,
                // code in this destructor might be executed and thus throwing another exception will result in calling
                // std::abort(). We're disabling the respective warning here.
#if defined(__GNUC__) and not defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wterminate"
#endif
                KASSERT(false, "MPI_Finalized call failed.");
#if defined(__GNUC__) and not defined(__clang__)
    #pragma GCC diagnostic pop
#endif
            }
            if (!is_already_finalized) {
                free_registered_mpi_types();
                [[maybe_unused]] int err = MPI_Finalize();
                // see above
#if defined(__GNUC__) and not defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wterminate"
#endif
                KASSERT(err == MPI_SUCCESS, "MPI_Finalize call failed.");
#if defined(__GNUC__) and not defined(__clang__)
    #pragma GCC diagnostic pop
#endif
            }
        }
    }

private:
    bool _finalize = false;
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    bool has_buffer_attached = false; ///< Is there currently an attached buffer?
#endif

}; // class Environment

/// @brief A global environment object to use when you don't want to create a new Environment object.
///
/// Note that \c inline \c const results in external linkage since C++17 (see
/// https://en.cppreference.com/w/cpp/language/inline).
inline Environment<InitMPIMode::NoInitFinalize> const mpi_env;

} // namespace kamping
