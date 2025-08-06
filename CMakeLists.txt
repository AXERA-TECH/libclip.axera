if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

project(libclip)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LEVELDB_BUILD_TESTS OFF)
set(LEVELDB_BUILD_BENCHMARKS OFF)
set(LEVELDB_INSTALL OFF)
add_subdirectory(third-party/leveldb)
include_directories(third-party/leveldb/include)

message(STATUS "OpenCV_DIR Path: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)
include_directories(src)
include_directories(src/utils)
include_directories(src/runner/ax_type_header)

add_library(clip SHARED
    src/clip.cpp
    src/runner/axcl/axcl_manager.cpp
    src/runner/axcl/ax_model_runner_axcl.cpp
    src/runner/ax650/ax_model_runner_ax650.cpp
    src/utils/enum_devices.cpp
    src/ax_devices.cpp
)

target_link_libraries(clip ${OpenCV_LIBS} leveldb dl pthread)

function(build_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} clip)
endfunction()

build_test(test_ax_api)
build_test(test_axcl_api)
build_test(test_enum_devices)
build_test(test_load_model)
build_test(test_match_by_text)

install(TARGETS clip
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SYSTEM_PROCESSOR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(DIRECTORY pyclip/ DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(DIRECTORY tests/ DESTINATION ${CMAKE_INSTALL_PREFIX}/examples)