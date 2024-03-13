cmake_minimum_required(VERSION 3.1)

# Set project
set(REDISXLM_VERSION "0.1.0")
message(STATUS "redisxlm version: ${REDISXLM_VERSION}")
project(redisxlm LANGUAGES CXX VERSION ${REDISXLM_VERSION})

# Set debug/release
set(REDISXLM_DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE ${REDISXLM_DEFAULT_BUILD_TYPE} CACHE STRING "Set build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()
message(STATUS "redisxlm build type: ${CMAKE_BUILD_TYPE} in project dir ${PROJECT_SOURCE_DIR}")

# Set compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -Werror -fPIC -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -fPIC -Wno-unused-parameter")
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(REDISXLM_HEADER_DIR src)

# Find pkg for remote llm https api
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)

# Fetch deps lib from git repository, use gemma.cpp to inference gemma model
include(FetchContent)
FetchContent_Declare(sentencepiece GIT_REPOSITORY https://github.com/google/sentencepiece GIT_TAG 53de76561cfc149d3c01037f0595669ad32a5e7c)
FetchContent_MakeAvailable(sentencepiece)
FetchContent_Declare(gemma GIT_REPOSITORY https://github.com/google/gemma.cpp GIT_TAG origin/main)
FetchContent_MakeAvailable(gemma)
FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG da250571a45826b21eebbddc1e50d0c1137dee5f)
FetchContent_MakeAvailable(highway)

# Set shared lib
set(SHARED_LIB shared)
file(GLOB PROJECT_SOURCE_FILES "${PROJECT_SOURCE_DIR}/*.cpp")
add_library(${SHARED_LIB} SHARED ${PROJECT_SOURCE_FILES})

# Executable target
target_compile_definitions(${SHARED_LIB} PRIVATE REDISMODULE_EXPERIMENTAL_API)
target_include_directories(${SHARED_LIB} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${REDISXLM_HEADER_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps>
        $<BUILD_INTERFACE:${OPENSSL_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CURL_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(${SHARED_LIB} PUBLIC ${OPENSSL_LIBRARIES})
target_link_libraries(${SHARED_LIB} PUBLIC ${CURL_LIBRARIES})
# link libgemma 
target_link_libraries(${SHARED_LIB} PUBLIC libgemma hwy hwy_contrib sentencepiece)
FetchContent_GetProperties(gemma)
FetchContent_GetProperties(sentencepiece)
target_include_directories(${SHARED_LIB} PRIVATE ${gemma_SOURCE_DIR})
target_include_directories(${SHARED_LIB} PRIVATE ${sentencepiece_SOURCE_DIR})
if(APPLE)
    target_link_libraries(${SHARED_LIB} PUBLIC "-framework AppKit -framework Security")
endif()
set_target_properties(${SHARED_LIB} PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
set_target_properties(${SHARED_LIB} PROPERTIES CLEAN_DIRECT_OUTPUT 1)

# Install shared lib.
include(GNUInstallDirs)
install(TARGETS ${SHARED_LIB}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
