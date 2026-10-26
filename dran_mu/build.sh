#!/bin/bash

export MU_LIB_PATH=/usr/local/mu_library/mu

source ${MU_LIB_PATH}/script/min_llvm_version_env.sh
if [ -z "${XCENA_LLVM_VERSION}" ] || [ -z "${MU_REVISION}" ]; then
    echo "XCENA_LLVM_VERSION or MU_REVISION is not set in min_llvm_version_env.sh"
    exit 1
fi
export MU_LLVM_PATH=/usr/local/mu_library/mu_llvm/$XCENA_LLVM_VERSION/$MU_REVISION/

rm -rf build
mkdir -p build
cd build
/usr/bin/cmake .. -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja
ninja install
cd -
