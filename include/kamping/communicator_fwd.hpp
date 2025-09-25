#pragma once

#include <vector>

namespace kamping {
template <
    template <typename...> typename DefaultContainerType = std::vector,
    template <typename, template <typename...> typename> typename... Plugins>
class Communicator;
}
