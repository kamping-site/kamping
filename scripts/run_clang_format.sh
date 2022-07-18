#!/bin/bash

if [[ "$PWD" == */scripts ]]; then
    echo "Script must be run from the project's root directory."
    exit 1
fi

# Get the absolute location of this script 
# https://stackoverflow.com/questions/59895/how-can-i-get-the-source-directory-of-a-bash-script-from-within-the-script-itsel
SOURCE=${BASH_SOURCE[0]}
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPT_DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

formatter=$("${SCRIPT_DIR}/get_clang_format.sh")
for directory in "include" "tests" "examples"; do
    find "$directory"                           \
        -type f                                 \
        \( -name "*.cpp" -or -name "*.hpp" \)   \
        -exec $formatter -i {} \;
done
