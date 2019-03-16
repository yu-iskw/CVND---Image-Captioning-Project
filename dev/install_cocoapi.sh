#!/bin/bash

# Get the path of project root.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname $DIR)"

# Install cocoapi
cd "${PROJECT_ROOT}/cocoapi/PythonAPI" || exit
make install
