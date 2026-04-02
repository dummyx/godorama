# Unit tests — pure C++, no Godot dependency

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(godot_llama_tests
    test_utf8.cpp
    test_error.cpp
    test_params.cpp
    test_rag_types.cpp
    test_rag_chunker.cpp
    test_rag_retriever.cpp
)

target_link_libraries(godot_llama_tests
    PRIVATE
        godot-llama-core
        godot-llama-rag
        Catch2::Catch2WithMain
)

set_target_properties(godot_llama_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Re-enable RTTI for tests (Catch2 needs it)
if(NOT MSVC)
    target_compile_options(godot_llama_tests PRIVATE -frtti)
endif()

include(Catch)
catch_discover_tests(godot_llama_tests)
