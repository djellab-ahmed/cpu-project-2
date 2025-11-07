set(TARGET gptoss-cli)
add_executable(${TARGET} main.cpp ${CMAKE_SOURCE_DIR}/src/spec_decode.cpp)
target_link_libraries(${TARGET} PRIVATE gptoss ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)

if(GPTOSS_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
