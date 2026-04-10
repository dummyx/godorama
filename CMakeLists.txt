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

option(GODOT_LLAMA_BUILD_GDEXTENSION "Build the Godot-facing GDExtension shared library" ON)
option(GODOT_LLAMA_ENABLE_MTMD "Build and link llama.cpp libmtmd for multimodal scaffolding" ON)

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

# libSQL: use the pinned local amalgamation with embedded vector SQL support
set(LIBSQL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libsql")
set(LIBSQL_AMALGAMATION_DIR "${LIBSQL_SOURCE_DIR}/libsql-ffi/bundled/src")
set(LIBSQL_AMALGAMATION_C "${LIBSQL_AMALGAMATION_DIR}/sqlite3.c")
set(LIBSQL_AMALGAMATION_H "${LIBSQL_AMALGAMATION_DIR}/sqlite3.h")

if(NOT EXISTS "${LIBSQL_AMALGAMATION_C}" OR NOT EXISTS "${LIBSQL_AMALGAMATION_H}")
    message(FATAL_ERROR "libSQL amalgamation not found under ${LIBSQL_AMALGAMATION_DIR}")
endif()

add_library(godot-llama-libsql STATIC
    ${LIBSQL_AMALGAMATION_C}
)
target_include_directories(godot-llama-libsql
    PUBLIC
        ${LIBSQL_AMALGAMATION_DIR}
)
target_compile_definitions(godot-llama-libsql
    PRIVATE
        SQLITE_THREADSAFE=1
        SQLITE_OMIT_LOAD_EXTENSION=1
)
set_target_properties(godot-llama-libsql PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
add_library(godot_llama::libsql ALIAS godot-llama-libsql)

# llama.cpp: build as static library, disable standalone extras
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
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

if(GODOT_LLAMA_ENABLE_MTMD AND NOT TARGET mtmd)
    set(LLAMA_INSTALL_VERSION "${PROJECT_VERSION}")
    add_subdirectory(thirdparty/llama.cpp/tools/mtmd EXCLUDE_FROM_ALL)
endif()

# godot-cpp (only required for the Godot wrapper target)
if(GODOT_LLAMA_BUILD_GDEXTENSION)
    if(TARGET godot-cpp)
        set(GODOT_LLAMA_GODOT_CPP_TARGET godot-cpp)
        set(GODOT_LLAMA_GODOT_CPP_PROPERTY_TARGET godot-cpp)
    elseif(TARGET godot::cpp)
        set(GODOT_LLAMA_GODOT_CPP_TARGET godot::cpp)
        set(GODOT_LLAMA_GODOT_CPP_PROPERTY_TARGET godot::cpp)
    else()
        add_subdirectory(thirdparty/godot-cpp EXCLUDE_FROM_ALL)
        set(GODOT_LLAMA_GODOT_CPP_TARGET godot-cpp)
        set(GODOT_LLAMA_GODOT_CPP_PROPERTY_TARGET godot-cpp)
    endif()
endif()

# ---- Internal libraries ----

# src/llama/ — thin adapter over libllama, no Godot headers
add_library(godot-llama-adapter STATIC
    src/llama/chat_template_engine.cpp
    src/llama/llama_lora_adapter_handle.cpp
    src/llama/llama_model_handle.cpp
    src/llama/llama_multimodal_handle.cpp
    src/llama/llama_context_handle.cpp
    src/llama/llama_position_layout.cpp
    src/llama/llama_sampler_handle.cpp
    src/llama/llama_params.cpp
)
if(NOT MSVC)
    set_source_files_properties(src/llama/chat_template_engine.cpp PROPERTIES COMPILE_OPTIONS "-fexceptions")
endif()
target_include_directories(godot-llama-adapter
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(godot-llama-adapter PUBLIC llama common)
if(TARGET mtmd)
    target_link_libraries(godot-llama-adapter PRIVATE mtmd)
    target_compile_definitions(godot-llama-adapter PRIVATE GODOT_LLAMA_HAS_MTMD=1)
endif()
target_compile_options(godot-llama-adapter PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-exceptions>
)
set_target_properties(godot-llama-adapter PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
add_library(godot_llama::adapter ALIAS godot-llama-adapter)

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
add_library(godot_llama::core ALIAS godot-llama-core)

# src/core/rag/ — local RAG runtime, persistence, retrieval, prompt packing
add_library(godot-llama-rag STATIC
    src/core/rag/types.cpp
    src/core/rag/corpus.cpp
    src/core/rag/deterministic_chunker.cpp
    src/core/rag/llama_embedder.cpp
    src/core/rag/libsql_corpus_store.cpp
    src/core/rag/dense_retriever.cpp
    src/core/rag/noop_reranker.cpp
    src/core/rag/grounded_context_packer.cpp
    src/core/rag/mock_embedder.cpp
    src/core/rag/factories.cpp
)
target_include_directories(godot-llama-rag
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(godot-llama-rag
    PUBLIC
        godot-llama-core
        godot_llama::libsql
)
target_compile_options(godot-llama-rag PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-exceptions>
)
set_target_properties(godot-llama-rag PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
add_library(godot_llama::rag ALIAS godot-llama-rag)

# ---- GDExtension shared library ----

if(GODOT_LLAMA_BUILD_GDEXTENSION)
    set(GDEXTENSION_TARGET godot_llama)

    add_library(${GDEXTENSION_TARGET} SHARED
        src/godot/register_types.cpp
        src/godot/llama_eval_session.cpp
        src/godot/llama_lora_adapter_config.cpp
        src/godot/llama_model_config.cpp
        src/godot/llama_multimodal_config.cpp
        src/godot/llama_session.cpp
        src/godot/rag_utils.cpp
        src/godot/rag_corpus_config.cpp
        src/godot/rag_corpus.cpp
        src/godot/rag_answer_session.cpp
    )
    add_library(godot_llama::godot ALIAS ${GDEXTENSION_TARGET})

    target_include_directories(${GDEXTENSION_TARGET}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(${GDEXTENSION_TARGET}
        PRIVATE
            godot-llama-core
            godot-llama-rag
            ${GODOT_LLAMA_GODOT_CPP_TARGET}
    )

    # Match godot-cpp naming conventions
    get_target_property(GODOTCPP_SUFFIX ${GODOT_LLAMA_GODOT_CPP_PROPERTY_TARGET} GODOTCPP_SUFFIX)
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
endif()

# ---- Tests ----

option(GODOT_LLAMA_BUILD_TESTS "Build unit tests" OFF)

if(GODOT_LLAMA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/unit)
    add_subdirectory(tests/integration)
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
