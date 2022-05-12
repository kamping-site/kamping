#pragma once

/// @brief Assertion levels
namespace kamping::assert {
/// @defgroup assertion-levels Assertion levels
///
/// @{

/// @brief Assertion level for lightweight assertions.
#define KAMPING_ASSERTION_LEVEL_LIGHT 20

/// @brief Assertion level for lightweight assertions.
constexpr int light = KAMPING_ASSERTION_LEVEL_LIGHT;

/// @brief Default assertion level. This level is used if no assertion level is specified.
#define KAMPING_ASSERTION_LEVEL_NORMAL 30

/// @brief Default assertion level. This level is used if no assertion level is specified.
constexpr int normal = KAMPING_ASSERTION_LEVEL_NORMAL;

/// @brief Assertions that perform lightweight communication.
#define KAMPING_ASSERTION_LEVEL_LIGHT_COMMUNICATION 40

/// @brief Assertions that perform lightweight communication.
constexpr int light_communication = KAMPING_ASSERTION_LEVEL_LIGHT_COMMUNICATION;

/// @brief Assertions that perform heavyweight communication.
#define KAMPING_ASSERTION_LEVEL_HEAVY_COMMUNICATION 50

/// @brief Assertions that perform heavyweight communication.
constexpr int heavy_communication = KAMPING_ASSERTION_LEVEL_HEAVY_COMMUNICATION;

/// @brief Assertion level for heavyweight assertions.
#define KAMPING_ASSERTION_LEVEL_HEAVY 60

/// @brief Assertion level for heavyweight assertions.
constexpr int heavy = KAMPING_ASSERTION_LEVEL_HEAVY;

/// @}
} // namespace kamping::assert
