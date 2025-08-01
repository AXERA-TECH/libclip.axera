if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

project(libclip)
set(CMAKE_CXX_STANDARD 17)

message(STATUS "OpenCV_DIR Path: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)
include_directories(src)
include_directories(src/utils)

include_directories(/usr/include/axcl)

add_library(clip SHARED
    src/clip.cpp
    src/runner/axcl/axcl_manager.cpp
    src/runner/axcl/ax_model_runner_axcl.cpp
    src/runner/ax650/ax_model_runner_ax650.cpp
    src/utils/enum_devices.cpp
)

target_link_libraries(clip ${OpenCV_LIBS} dl pthread)

function(build_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} clip)
endfunction()

build_test(test_ax_api)
build_test(test_axcl_api)
build_test(test_enum_devices)
build_test(test_load_model)
build_test(test_match_by_text)