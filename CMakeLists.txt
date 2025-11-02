cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("gptoss.cpp" C CXX)
include(CheckIncludeFileCXX)

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# if building in release, enable aggressive optimization and math fast-paths
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG -march=native)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-fno-math-errno -fno-trapping-math>
                        $<$<COMPILE_LANGUAGE:CXX>:-fno-math-errno -fno-trapping-math>)
endif()

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GPTOSS_STANDALONE ON)

    include(git-vars)

    # configure project version
    # TODO
else()
    set(GPTOSS_STANDALONE OFF)
endif()

option(GPTOSS_USE_SYSTEM_GGML "Use system libggml" OFF)

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(GPTOSS_WASM_SINGLE_FILE "gptoss: embed WASM inside the generated gptoss.js" ON)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(GPTOSS_TOOLS_INSTALL_DEFAULT OFF)
else()
    set(GPTOSS_TOOLS_INSTALL_DEFAULT ${GPTOSS_STANDALONE})
endif()

#
# option list
#

# debug
option(GPTOSS_ALL_WARNINGS           "gptoss: enable all compiler warnings"                   ON)
option(GPTOSS_ALL_WARNINGS_3RD_PARTY "gptoss: enable all compiler warnings in 3rd party libs" OFF)

# build
option(GPTOSS_FATAL_WARNINGS "gptoss: enable -Werror flag" OFF)

# sanitizers
option(GPTOSS_SANITIZE_THREAD    "gptoss: enable thread sanitizer"    OFF)
option(GPTOSS_SANITIZE_ADDRESS   "gptoss: enable address sanitizer"   OFF)
option(GPTOSS_SANITIZE_UNDEFINED "gptoss: enable undefined sanitizer" OFF)

# utils
option(GPTOSS_BUILD_TOOLS   "gptoss: build tools"   ${GPTOSS_STANDALONE})
option(GPTOSS_TOOLS_INSTALL "gptoss: install tools" ${GPTOSS_TOOLS_INSTALL_DEFAULT})

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

if (NOT DEFINED GPTOSS_BUILD_NUMBER)
    set(GPTOSS_BUILD_NUMBER        ${BUILD_NUMBER})
endif()
if (NOT DEFINED GPTOSS_BUILD_COMMIT)
    set(GPTOSS_BUILD_COMMIT        ${BUILD_COMMIT})
endif()
set(GPTOSS_INSTALL_VERSION 0.0.${GPTOSS_BUILD_NUMBER})

# override ggml options
set(GGML_ALL_WARNINGS   ${GPTOSS_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${GPTOSS_FATAL_WARNINGS})

# change the default for these ggml options
if (NOT DEFINED GGML_GPTOSSFILE)
    set(GGML_GPTOSSFILE_DEFAULT ON)
endif()

if (NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()

# transition helpers
function (gptoss_option_depr TYPE OLD NEW)
    if (${OLD})
        message(${TYPE} "${OLD} is deprecated and will be removed in the future.\nUse ${NEW} instead\n")
        set(${NEW} ON PARENT_SCOPE)
    endif()
endfunction()

gptoss_option_depr(FATAL_ERROR GPTOSS_CUBLAS              GGML_CUDA)
gptoss_option_depr(WARNING     GPTOSS_CUDA                GGML_CUDA)
gptoss_option_depr(WARNING     GPTOSS_METAL               GGML_METAL)
gptoss_option_depr(WARNING     GPTOSS_METAL_EMBED_LIBRARY GGML_METAL_EMBED_LIBRARY)
gptoss_option_depr(WARNING     GPTOSS_NATIVE              GGML_NATIVE)
gptoss_option_depr(WARNING     GPTOSS_RPC                 GGML_RPC)
gptoss_option_depr(WARNING     GPTOSS_SYCL                GGML_SYCL)
gptoss_option_depr(WARNING     GPTOSS_SYCL_F16            GGML_SYCL_F16)
gptoss_option_depr(WARNING     GPTOSS_CANN                GGML_CANN)

if (NOT MSVC)
    if (GPTOSS_SANITIZE_THREAD)
        message(STATUS "Using -fsanitize=thread")

        add_compile_options(-fsanitize=thread)
        link_libraries     (-fsanitize=thread)
    endif()

    if (GPTOSS_SANITIZE_ADDRESS)
        message(STATUS "Using -fsanitize=address")

        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries     (-fsanitize=address)
    endif()

    if (GPTOSS_SANITIZE_UNDEFINED)
        message(STATUS "Using -fsanitize=undefined")

        add_compile_options(-fsanitize=undefined)
        link_libraries     (-fsanitize=undefined)
    endif()
endif()

#
# 3rd-party
#

if (GPTOSS_USE_SYSTEM_GGML)
    message(STATUS "Using system-provided libggml, skipping ggml build")
    find_package(ggml REQUIRED)
    add_library(ggml ALIAS ggml::ggml)
endif()

if (NOT TARGET ggml AND NOT GPTOSS_USE_SYSTEM_GGML)
    set(GGML_BUILD_NUMBER ${GPTOSS_BUILD_NUMBER})
    set(GGML_BUILD_COMMIT ${GPTOSS_BUILD_COMMIT})
    add_subdirectory(ggml)
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()

if (MINGW)
    # Target Windows 8 for PrefetchVirtualMemory
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()

#
# build the library
#

add_subdirectory(src)

#
# tools
#

if (GPTOSS_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(GPTOSS_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(GPTOSS_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(GPTOSS_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

set(GPTOSS_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/gptoss.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/gptoss-cpp.h)

set_target_properties(gptoss
    PROPERTIES
        PUBLIC_HEADER "${GPTOSS_PUBLIC_HEADERS}")

install(TARGETS gptoss LIBRARY PUBLIC_HEADER)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gptoss-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/gptoss-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gptoss
    PATH_VARS GPTOSS_INCLUDE_INSTALL_DIR
              GPTOSS_LIB_INSTALL_DIR
              GPTOSS_BIN_INSTALL_DIR )

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/gptoss-version.cmake
    VERSION ${GPTOSS_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gptoss-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/gptoss-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gptoss)

install(
    FILES convert_hf_to_gguf.py
    PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR})

configure_file(cmake/gptoss.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/gptoss.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/gptoss.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
