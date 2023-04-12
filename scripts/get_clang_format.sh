#!/bin/bash

clang_format_version=14

# Use clang-format-<version> if available, else use clang-format
if command -v clang-format-$clang_format_version &> /dev/null; then
    formatter=clang-format-$clang_format_version
else
    formatter=clang-format
fi

# get the actual version used
version_string=$($formatter --version)
pattern='.* clang-format version ([0-9]+).' # regex
[[ "$version_string" =~ $pattern ]] # match the regex
actual_version="${BASH_REMATCH[1]}" # get the first match group

if [[ "$actual_version" != "$clang_format_version" ]]; then
    >&2 echo "WARNING: You are using clang-format version $actual_version instead of $clang_format_version."
fi
echo $formatter
