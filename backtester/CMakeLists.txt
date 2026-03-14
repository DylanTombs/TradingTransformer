cmake_minimum_required(VERSION 3.15)
project(backtester)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cpp")

# Production binary
add_executable(backtest main.cpp ${SOURCES})

# ---------------------------------------------------------------------------
# Tests — Google Test fetched at configure time, no manual install required.
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# Prevent GTest from overriding compiler/linker settings on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB TEST_SOURCES "tests/*.cpp")

# Test binary links the same src/*.cpp (no main.cpp) + GTest's main().
add_executable(backtester_tests ${TEST_SOURCES} ${SOURCES})
target_include_directories(backtester_tests PRIVATE include)
target_link_libraries(backtester_tests GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(backtester_tests)
