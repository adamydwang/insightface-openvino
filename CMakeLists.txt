cmake_minimum_required(VERSION 2.7)
project(insightface)
add_definitions(-std=c++11)

find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${InferenceEngine_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS})

set(LIBS ${InferenceEngine_LIBRARIES} ${OpenCV_LIBRARIES})

set(SRCS  ${PROJECT_SOURCE_DIR}/src/insightface.cpp)
add_executable(demo ${PROJECT_SOURCE_DIR}/src/demo.cpp ${SRCS})
target_link_libraries(demo  ${LIBS})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
