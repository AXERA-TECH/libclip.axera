if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

project(libclip)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LEVELDB_BUILD_TESTS OFF)
set(LEVELDB_BUILD_BENCHMARKS OFF)
set(LEVELDB_INSTALL OFF)
add_subdirectory(third-party/leveldb)
include_directories(third-party/leveldb/include)

set(SIMPLECV_BUILD_TESTS OFF)
add_subdirectory(third-party/SimpleCV)
include_directories(third-party/SimpleCV/include)

option(AXCL_USE_STATIC_LINK "Use static link for axcl" OFF)

# 如果用户没有显式指定，就在 Windows 上默认开启
if(WIN32 AND NOT DEFINED AXCL_USE_STATIC_LINK)
    set(AXCL_USE_STATIC_LINK ON)
endif()

message(STATUS "AXCL_USE_STATIC_LINK: ${AXCL_USE_STATIC_LINK}")

if(AXCL_USE_STATIC_LINK)
    if (AXCL_DIR)
        message(STATUS "AXCL_DIR Path: ${AXCL_DIR}")
        include_directories(${AXCL_DIR}/include)
        link_directories(${AXCL_DIR}/lib)
        include_directories(${AXCL_DIR}/include/axcl)
        link_directories(${AXCL_DIR}/lib/axcl)
    else()
        include_directories(/usr/include/axcl/)
        link_directories(/usr/lib/axcl/)
    endif()
endif()

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
    src/tokenizer/tokenizer.cpp
)

if(AXCL_USE_STATIC_LINK)
    target_compile_definitions(clip PRIVATE AXCL_USE_STATIC_LINK)
    target_link_libraries(clip axcl_rt)
endif()

if(WIN32)
    target_link_libraries(clip SimpleCV::simplecv leveldb)
else()
    target_link_libraries(clip SimpleCV::simplecv leveldb dl)
endif()

function(build_test test_name source)
    add_executable(${test_name} ${source} tests/utils/cqdm.cpp)
    target_link_libraries(${test_name} clip SimpleCV::simplecv)
endfunction()

build_test(test_ax_api tests/test_ax_api.cpp)
build_test(test_axcl_api tests/test_axcl_api.cpp)
build_test(test_enum_devices tests/test_enum_devices.cpp)
build_test(test_load_model tests/test_load_model.cpp)
build_test(test_match_by_text tests/test_match_by_text.cpp)
build_test(test_tokenizer tests/tokenizer/test_tokenizer.cpp)



install(TARGETS clip
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SYSTEM_PROCESSOR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(DIRECTORY pyclip/ DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(DIRECTORY tests/ DESTINATION ${CMAKE_INSTALL_PREFIX}/examples)