cmake_minimum_required(VERSION 3.15)

project(cppExaStoLog VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(Boost REQUIRED)

file(GLOB_RECURSE src_files "src/*.cpp")
list(FILTER src_files EXCLUDE REGEX ".*main\\.cpp")
add_library(exaStoLogLib ${src_files})

add_executable(cppExaStoLog src/main.cpp)
target_link_libraries(cppExaStoLog exaStoLogLib)

file(GLOB_RECURSE test_files "test/*")
add_executable(unit_cppExaStoLog ${test_files})

foreach(executable cppExaStoLog unit_cppExaStoLog exaStoLogLib)
	if(MSVC)
		target_compile_options(${executable} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /bigobj>)
	else()
		target_compile_options(${executable} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>)
	endif()
endforeach()

target_include_directories(cppExaStoLog PUBLIC ${Boost_INCLUDE_DIR})

target_include_directories(unit_cppExaStoLog PUBLIC 
	"src")

target_link_libraries(unit_cppExaStoLog PRIVATE 
	gtest_main exaStoLogLib)

add_custom_command(TARGET unit_cppExaStoLog POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/../data $<TARGET_FILE_DIR:${PROJECT_NAME}>/data)
	
enable_testing()
include(GoogleTest)
gtest_discover_tests(unit_cppExaStoLog)
