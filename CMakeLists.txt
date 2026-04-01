cmake_minimum_required(VERSION 3.28)

project(
    godot-llama
    VERSION 0.1.0
    DESCRIPTION "Godot 4 GDExtension embedding llama.cpp for local LLM inference"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Note: exceptions/RTTI are disabled per-target for our own code only.
# Third-party dependencies (llama.cpp, godot-cpp) need exceptions/RTTI enabled.

# ccache support
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# ---- Third-party dependencies ----

# llama.cpp: build as static library, disable standalone extras
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(LLAMA_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)

# CPU-only by default; GPU backends are opt-in
if(NOT DEFINED GGML_CUDA)
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
endif()
if(NOT DEFINED GGML_METAL)
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
endif()
if(NOT DEFINED GGML_VULKAN)
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(thirdparty/llama.cpp EXCLUDE_FROM_ALL)

# godot-cpp
add_subdirectory(thirdparty/godot-cpp EXCLUDE_FROM_ALL)

# ---- Internal libraries ----

# src/llama/ — thin adapter over libllama, no Godot headers
add_library(godot-llama-adapter STATIC
    src/llama/llama_model_handle.cpp
    src/llama/llama_context_handle.cpp
    src/llama/llama_sampler_handle.cpp
    src/llama/llama_params.cpp
)
target_include_directories(godot-llama-adapter
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(godot-llama-adapter PUBLIC llama)
target_compile_options(godot-llama-adapter PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-exceptions>
)
set_target_properties(godot-llama-adapter PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# src/core/ — async runtime, errors, UTF-8 helpers
add_library(godot-llama-core STATIC
    src/core/error.cpp
    src/core/utf8.cpp
    src/core/request.cpp
    src/core/worker.cpp
)
target_include_directories(godot-llama-core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(godot-llama-core PUBLIC godot-llama-adapter)
target_compile_options(godot-llama-core PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-exceptions>
)
set_target_properties(godot-llama-core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ---- GDExtension shared library ----

set(GDEXTENSION_TARGET godot_llama)

add_library(${GDEXTENSION_TARGET} SHARED
    src/godot/register_types.cpp
    src/godot/llama_model_config.cpp
    src/godot/llama_session.cpp
)

target_include_directories(${GDEXTENSION_TARGET}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${GDEXTENSION_TARGET}
    PRIVATE
        godot-llama-core
        godot-cpp
)

# Match godot-cpp naming conventions
get_target_property(GODOTCPP_SUFFIX godot-cpp GODOTCPP_SUFFIX)
set_target_properties(${GDEXTENSION_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    PREFIX "lib"
    OUTPUT_NAME "godot_llama${GODOTCPP_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/demo/bin/"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/demo/bin/"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set_target_properties(${GDEXTENSION_TARGET} PROPERTIES
        SUFFIX ""
    )
endif()

target_compile_options(${GDEXTENSION_TARGET} PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-exceptions>
)

# ---- Tests ----

option(GODOT_LLAMA_BUILD_TESTS "Build unit tests" OFF)

if(GODOT_LLAMA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/unit)
endif()

# ---- Format and lint targets ----

find_program(CLANG_FORMAT clang-format)
find_program(CLANG_TIDY clang-tidy)

file(GLOB_RECURSE ALL_PROJECT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
)

if(CLANG_FORMAT)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -i ${ALL_PROJECT_SOURCES}
        COMMENT "Running clang-format"
        VERBATIM
    )
    add_custom_target(format-check
        COMMAND ${CLANG_FORMAT} --dry-run --Werror ${ALL_PROJECT_SOURCES}
        COMMENT "Checking clang-format"
        VERBATIM
    )
endif()

if(CLANG_TIDY)
    add_custom_target(lint
        COMMAND ${CLANG_TIDY}
            -p ${CMAKE_BINARY_DIR}
            ${ALL_PROJECT_SOURCES}
        COMMENT "Running clang-tidy"
        VERBATIM
    )
endif()
