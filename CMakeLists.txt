cmake_minimum_required(VERSION 3.20)
project(diskkv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB DISK_KV_SOURCES CONFIGURE_DEPENDS src/*.cpp)

add_library(diskkv STATIC ${DISK_KV_SOURCES})
target_include_directories(diskkv PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(diskkv PUBLIC Threads::Threads)
target_compile_options(diskkv PRIVATE -Wall -Wextra)

enable_testing()

# Store logic suite: arg1 = backing-file path (created by the test itself).
add_executable(disk_kv_store_test tests/disk_kv_store_test.cpp)
target_link_libraries(disk_kv_store_test PRIVATE diskkv)
add_test(NAME disk_kv_store_test
         COMMAND disk_kv_store_test ${CMAKE_BINARY_DIR}/l.bin 262144 512)

# Bridge round-trip suite: arg1 = output directory for per-family files.
add_executable(disk_kv_bridge_test tests/disk_kv_bridge_test.cpp)
target_link_libraries(disk_kv_bridge_test PRIVATE diskkv)
add_test(NAME disk_kv_bridge_test
         COMMAND disk_kv_bridge_test ${CMAKE_BINARY_DIR}/dkb)
add_custom_command(TARGET disk_kv_bridge_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/dkb)
