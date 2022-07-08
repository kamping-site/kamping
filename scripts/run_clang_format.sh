#!/bin/bash

if [[ "$PWD" == */scripts ]]; then
    echo "Script must be run from the project's root directory."
    exit 1
fi

clang_format_version=13

# Use clang-format-<version> if available, else use clang-format
if command -v clang-format-$clang_format_version &> /dev/null; then
    formatter=clang-format-$clang_format_version
else
    formatter=clang-format
fi

version_string=$($formatter --version)
pattern='.* clang-format version ([0-9]+).'
[[ "$version_string" =~ $pattern ]]
actual_version="${BASH_REMATCH[1]}"

if [[ "$actual_version" != "$clang_format_version" ]]; then
    echo "WARNING: You are using clang-format version $actual_version instead of $clang_format_version."
fi

for directory in "include" "tests" "examples"; do
    find "$directory"                           \
        -type f                                 \
        \( -name "*.cpp" -or -name "*.hpp" \)   \
        -exec $formatter -i {} \;
done
