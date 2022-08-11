#!/bin/bash

CMAKE_FORMAT="cmake-format"

EXIT_SUCCESS=0
EXIT_EXEC_NOT_FOUND=1

# cmake-format does not have multiple versions yet, simply return cmake-format
if command -v "$CMAKE_FORMAT" >/dev/null 2>&1; then
    command -v "$CMAKE_FORMAT"
    exit $EXIT_SUCCESS
else
    exit $EXIT_EXEC_NOT_FOUND
fi
