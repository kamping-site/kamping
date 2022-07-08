#!/bin/bash

if [[ "$PWD" == */scripts ]]; then
    echo "Script must be run from the project's root directory."
    exit 1
fi

clang_format_version=13

# Use clang-format-<version> if available, else use clang-format
if command -v clang-format-$clang_format_version &> /dev/null
then
    formatter=clang-format-$clang_format_version
else
    formatter=clang-format
fi

for directory in "include" "tests" "examples"; do
    find "$directory"                           \
        -type f                                 \
        \( -name "*.cpp" -or -name "*.hpp" \)   \
        -exec $formatter -i {} \;
done
