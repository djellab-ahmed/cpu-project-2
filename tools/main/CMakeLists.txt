set(TARGET gptoss-cli)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE common gptoss ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(GPTOSS_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
