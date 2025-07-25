cmake_minimum_required(VERSION 3.5.1)

project(rknn_simplebev)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_simplebev_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

if(CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../runtime//${CMAKE_SYSTEM_NAME}/librknn_api)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
else()
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
endif()

# ros dependencies
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  std_msgs
  image_transport
  message_filters
)

# opencv
find_package(OpenCV REQUIRED)

include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/../3rdparty/)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${catkin_INCLUDE_DIRS})


# 设置RPATH
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)


add_executable(rknn_simplebev
        src/main.cc
        src/simplebev.cc
        src/multi_camera_subscriber.cpp
)

target_link_libraries(rknn_simplebev
	${RKNN_RT_LIB}
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
	target_link_libraries(rknn_simplebev log)
endif()

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_simplebev_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_simplebev DESTINATION ./)
install(DIRECTORY model/${TARGET_SOC} DESTINATION ./model)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
file(GLOB DATA_FILES "model/*.bin")
install(FILES ${DATA_FILES} DESTINATION ./model/)
