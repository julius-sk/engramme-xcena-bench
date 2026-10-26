cmake_minimum_required(VERSION 3.11)
project(mu_kernel)

set(CMAKE_C_COMPILER ${MU_LLVM_PATH}/bin/clang)
set(CMAKE_CXX_COMPILER ${MU_LLVM_PATH}/bin/clang)

set(output mu_kernel.mubin)
set(output_path ${CMAKE_CURRENT_SOURCE_DIR})

set(include_directories
    ${MU_LLVM_PATH}/picolibc-rv
    ${MU_LLVM_PATH}/picolibc-rv/include
    ${MU_LLVM_PATH}/libcxx-rv/include/c++/v1
    ${MU_LIB_PATH}/include
)

set(link_directories
    ${MU_LLVM_PATH}/builtins-rv/lib/linux
    ${MU_LLVM_PATH}/libcxx-rv/lib
    ${MU_LLVM_PATH}/libcxxabi-rv/lib
    ${MU_LLVM_PATH}/picolibc-rv/newlib
    ${MU_LIB_PATH}/lib
)

set(compile_options
    --target=riscv64
    -mcpu=metisx-d
    -ffreestanding
    -nostdlib
    -nodefaultlibs
    -fno-exceptions
    -Werror=return-type
    -g
    -O3
    -std=c++17
    -fno-rtti
    -ffunction-sections
    -fdata-sections
)

set(link_options
    -lclang_rt.builtins-riscv64
    -z max-page-size=4
    -lc
    -lc++
    -lc++abi
    -Wl,--gc-sections
    --ld-path=${MU_LLVM_PATH}/bin/ld.lld
)

set(start_up ${MU_LIB_PATH}/script/mu_startup.s)
set(linker ${MU_LIB_PATH}/script/mu_linker.ld)

set(src
    mu_dran.cpp
)

add_executable(${output} ${src})

target_include_directories(${output} PRIVATE ${include_directories})
target_link_libraries(${output} PRIVATE mu_std)
target_link_directories(${output} PRIVATE ${link_directories})
target_compile_options(${output} PRIVATE ${compile_options})
target_link_options(${output} PRIVATE ${compile_options} ${link_options} ${start_up} ${linker})
set_target_properties(${output} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${output_path})
install(TARGETS ${output} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/mu_kernel)
