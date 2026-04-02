include(FetchContent)
if(NOT TARGET Catch2::Catch2WithMain)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.7.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
endif()

add_executable(godot_llama_rag_integration_tests
    test_rag_pipeline.cpp
)

target_link_libraries(godot_llama_rag_integration_tests
    PRIVATE
        godot-llama-rag
        Catch2::Catch2WithMain
)

set_target_properties(godot_llama_rag_integration_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(NOT MSVC)
    target_compile_options(godot_llama_rag_integration_tests PRIVATE -frtti)
endif()
add_test(NAME rag-pipeline COMMAND godot_llama_rag_integration_tests)

add_executable(godot_llama_rag_eval
    rag_evaluation.cpp
)
target_link_libraries(godot_llama_rag_eval PRIVATE godot-llama-rag)
set_target_properties(godot_llama_rag_eval PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
add_test(NAME rag-evaluation COMMAND godot_llama_rag_eval)
