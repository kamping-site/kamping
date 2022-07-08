#!/bin/bash

if [[ "$PWD" == */scripts ]]; then
    echo "Script must be run from the project's root directory."
    exit 1
fi

# Get the absolute location of this script 
# https://stackoverflow.com/questions/59895/how-can-i-get-the-source-directory-of-a-bash-script-from-within-the-script-itsel
SOURCE=${BASH_SOURCE[0]}

formatter=$("$SOURCE/get_clang_format.sh")
for directory in "include" "tests" "examples"; do
    find "$directory"                           \
        -type f                                 \
        \( -name "*.cpp" -or -name "*.hpp" \)   \
        -exec $formatter -i {} \;
done
