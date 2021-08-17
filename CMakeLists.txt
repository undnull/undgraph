cmake_minimum_required(VERSION 3.10)
project(undgraph LANGUAGES C)

set(CMAKE_C_STANDARD 90)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(glad)
add_subdirectory(glfw)

add_executable(undgraph "${CMAKE_CURRENT_LIST_DIR}/undgraph.c")
target_compile_definitions(undgraph PRIVATE _CRT_SECURE_NO_WARNINGS=1)
target_compile_definitions(undgraph PRIVATE GLFW_INCLUDE_NONE=1)
target_link_libraries(undgraph PRIVATE glad glfw)
