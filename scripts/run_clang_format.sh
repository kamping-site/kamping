#!/bin/bash

if [[ "$PWD" == */scripts ]]; then
    echo "Script must be run from the project's root directory."
    exit 1
fi

formatter=$("${BASH_SOURCE%/*}/get_clang_format.sh")
for directory in "include" "tests" "examples"; do
    find "$directory"                           \
        -type f                                 \
        \( -name "*.cpp" -or -name "*.hpp" \)   \
        -exec $formatter -i {} \;
done
