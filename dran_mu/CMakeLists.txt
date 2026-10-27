cmake_minimum_required(VERSION 3.11)
project(dran_retrieve)

set(src
    dran_retrieve.cpp
    dran_retrieve_lanec.cpp
    im_probe.cpp
    im_examples.cpp
)

set(include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(shared_libs)
find_package(Threads REQUIRED)
LIST(APPEND shared_libs Threads::Threads)

if (NOT XCENA_SDK_CMAKE_BUILD)
    find_package(pxl REQUIRED)
    message(STATUS "PXL library found at: ${pxl_DIR}")
    LIST(APPEND shared_libs pxl::pxl)
else()
    LIST(APPEND shared_libs pxl)
endif()

foreach(file ${src})
    get_filename_component(target ${file} NAME_WE)
    add_executable(${target} ${file})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_include_directories(${target} PRIVATE ${include})
    target_link_libraries(${target} ${shared_libs})
    install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

# build the MU kernel (.mubin) via the SDK's LLVM RISC-V toolchain
include(ExternalProject)
ExternalProject_Add(build_mu_${PROJECT_NAME}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mu_kernel
    BUILD_ALWAYS TRUE
    CMAKE_ARGS
        -DMU_LIB_PATH=$ENV{MU_LIB_PATH}
        -DMU_LLVM_PATH=$ENV{MU_LLVM_PATH}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
)
