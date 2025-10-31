gptoss_add_compile_flags()

#
# libraries
#

# gptoss

add_library(gptoss
            ../include/gptoss.h
            gptoss.cpp
            gptoss-adapter.cpp
            gptoss-arch.cpp
            gptoss-batch.cpp
            gptoss-chat.cpp
            gptoss-context.cpp
            gptoss-cparams.cpp
            gptoss-grammar.cpp
            gptoss-graph.cpp
            gptoss-hparams.cpp
            gptoss-impl.cpp
            gptoss-io.cpp
            gptoss-kv-cache.cpp
            gptoss-kv-cache-iswa.cpp
            gptoss-memory.cpp
            gptoss-memory-hybrid.cpp
            gptoss-memory-recurrent.cpp
            gptoss-mmap.cpp
            gptoss-model-loader.cpp
            gptoss-model-saver.cpp
            gptoss-model.cpp
            gptoss-quant.cpp
            gptoss-sampling.cpp
            gptoss-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            )

target_include_directories(gptoss PRIVATE .)
target_include_directories(gptoss PUBLIC ../include)
target_compile_features   (gptoss PRIVATE cxx_std_17) # don't bump

target_link_libraries(gptoss PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(gptoss PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(gptoss PRIVATE GPTOSS_BUILD)
    target_compile_definitions(gptoss PUBLIC  GPTOSS_SHARED)
endif()
