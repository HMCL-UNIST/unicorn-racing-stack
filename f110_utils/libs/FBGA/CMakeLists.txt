############################################################################
#                                                                          #
#  file: CMakeLists.txt                                                    #
#  Copyright (C) 2025                                                      #
#      Mattia Piazza                                                       #
#      Dipartimento di Ingegneria Industriale                              #
#      Universita` degli Studi di Trento                                   #
#      email: mattia.piazza@unitn.it                                       #
#                                                                          #
############################################################################

cmake_minimum_required(VERSION 3.14)

macro(find_and_log package)
    find_package(${package} REQUIRED)
    message(STATUS "Package ${package} found at ${${package}_DIR}")
endmacro()

project( GIGI )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

project(
  ${PROJECT_NAME}
  VERSION ${UTILS_PROJECT_VERSION}
  HOMEPAGE_URL "https://github.com/XXXXX/GIGI"
)


#   ___
#  / __| ___ _  _ _ _ __ ___ ___
#  \__ \/ _ \ || | '_/ _/ -_|_-<
#  |___/\___/\_,_|_| \__\___/__/
#
set( SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src" )

set( SOURCES )
file( GLOB S ${SOURCE_DIR}/*.cc )
foreach(F ${S})
  # build relative path respect to source directory
  file (RELATIVE_PATH RF ${CMAKE_CURRENT_SOURCE_DIR} "${F}")
  list( APPEND SOURCES ${RF})
endforeach(F ${S})

#   _  _             _
#  | || |___ __ _ __| |___ _ _ ___
#  | __ / -_) _` / _` / -_) '_(_-<
#  |_||_\___\__,_\__,_\___|_| /__/
#
set( PUBLIC_HEADERS )
file(
  GLOB S
  ${SOURCE_DIR}/*.h*
  ${SOURCE_DIR}/*/*.h*
)
foreach (F ${S})
  # build relative path respect to source directory
  file( RELATIVE_PATH RF ${CMAKE_CURRENT_SOURCE_DIR} "${F}" )
  list( APPEND PUBLIC_HEADERS ${RF} )
endforeach(F ${S})

# extra include directory for compilation
include_directories( ./lib3rd/include )

message(STATUS "INCLUDES: ${PUBLIC_HEADERS}")


#       _                           _                 _
#    __| | ___ _ __   ___ _ __   __| | ___ _ __   ___(_) ___  ___
#   / _` |/ _ \ '_ \ / _ \ '_ \ / _` |/ _ \ '_ \ / __| |/ _ \/ __|
#  | (_| |  __/ |_) |  __/ | | | (_| |  __/ | | | (__| |  __/\__ \
#   \__,_|\___| .__/ \___|_| |_|\__,_|\___|_| |_|\___|_|\___||___/
#             |_|
#

# set a third parties directory
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(THIRD_PARTY_DIR_USR ${THIRD_PARTY_DIR}/usr )


add_library(thirdparty INTERFACE)
target_include_directories(
  thirdparty INTERFACE
  ${THIRD_PARTY_DIR}/cxxopts/include
  ${THIRD_PARTY_DIR}/json/single_include/nlohmann
  ${THIRD_PARTY_DIR}/rapidcsv/src
)


#   _____                  _
#  |_   _|_ _ _ _ __ _ ___| |_ ___
#    | |/ _` | '_/ _` / -_)  _(_-<
#    |_|\__,_|_| \__, \___|\__/__/
#                |___/
#

add_library(${PROJECT_NAME} STATIC ${SOURCES} ${PUBLIC_HEADERS})

# -fPIC so this static lib can be linked into the pybind11
# shared module (fbga_native.so). Harmless for the existing .exe builds.
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${PROJECT_NAME} PUBLIC ${thirdpartysubmodules} thirdparty )
target_include_directories(${PROJECT_NAME} PUBLIC ${SOURCE_DIR} )



#   _____       _
#  |_   _|__ __| |_ ___
#    | |/ -_|_-<  _(_-<
#    |_|\___/__/\__/__/
#


set( LIBS ${PROJECT_NAME} ${thirdpartysubmodules} ${ZLIB_LIBRARIES} ${CMAKE_DL_LIBS}  )

if( UNIX )
  if ( NOT APPLE )
    set( THREADS_PREFER_PTHREAD_FLAG ON )
    find_package( Threads REQUIRED )
    set( LIBS ${LIBS} Threads::Threads )
  endif()
else()
  set( LIBS ${LIBS} ws2_32 iphlpapi kernel32 )
endif()

file(GLOB EXELISTCPP_WEXT src_tests/GIGI_test_*.cc)

set(EXELISTCPP)
foreach(FILE ${EXELISTCPP_WEXT})
  get_filename_component(FILENAME ${FILE} NAME_WE)
  list(APPEND EXELISTCPP ${FILENAME})
endforeach()

add_custom_target( "${PROJECT_NAME}_all_tests" ALL )

set( EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin )
foreach( S ${EXELISTCPP} )
  add_executable( ${S} ${CMAKE_CURRENT_SOURCE_DIR}/src_tests/${S}.cc )
  target_link_libraries( ${S} ${LIBS} thirdparty )
  set_target_properties( ${S} PROPERTIES SUFFIX ".exe" )
  add_test( NAME "${S}" COMMAND ./bin/${S}.exe WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
  add_dependencies( "${PROJECT_NAME}_all_tests" ${S} )
endforeach()


# pybind11 Python binding (fbga_native).
# Eliminates the per-solve subprocess fork by exposing FWBW as a Python module.
# Optional — if pybind11 isn't found, just skip and the legacy exe path still works.
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(Python3_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_QUERY_RC
  )
  if(PYBIND11_QUERY_RC EQUAL 0 AND PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
    find_package(pybind11 CONFIG QUIET)
  endif()
endif()

if(pybind11_FOUND)
  message(STATUS "pybind11 found at ${PYBIND11_CMAKE_DIR} - building fbga_native module")
  pybind11_add_module(fbga_native ${CMAKE_CURRENT_SOURCE_DIR}/python/fbga_native.cpp)
  target_link_libraries(fbga_native PRIVATE ${PROJECT_NAME} thirdparty)
  target_include_directories(fbga_native PRIVATE ${SOURCE_DIR})
  # Place the .so next to the test binaries so fbga_runner.py finds it via a
  # fixed path: <FBGA>/bin/fbga_native.cpython-<py>-<arch>.so
  set_target_properties(fbga_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
else()
  message(STATUS "pybind11 NOT found - skipping fbga_native module "
                 "(legacy exe path still works). Run: pip3 install pybind11")
endif()

