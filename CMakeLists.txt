cmake_minimum_required(VERSION 3.4.1)

project(rknn_cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_multithreading_jz_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

if(CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()

include_directories(${CMAKE_SOURCE_DIR})

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/runtime//${CMAKE_SYSTEM_NAME}/librknn_api)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
else()
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
endif()

include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty)

# opencv
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/OpenCV-android-sdk/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
else()
  if(LIB_ARCH STREQUAL "armhf")
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
  else()
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/usr/share/opasdasdencv)
  endif()
endif()

find_package(OpenCV REQUIRED)

# rga
# comes from https://github.com/airockchip/librga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/3rdparty/rga/)
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RGA_LIB ${RGA_PATH}/libs/AndroidNdk/${CMAKE_ANDROID_ARCH_ABI}/librga.so)
else()
  if(CMAKE_C_COMPILER MATCHES "aarch64")
    set(LIB_ARCH aarch64)
  else()
    set(LIB_ARCH armhf)
  endif()

  set(RGA_LIB ${RGA_PATH}/libs/Linux//gcc-${LIB_ARCH}/librga.so)
endif()
include_directories( ${RGA_PATH}/include)

# mpp
set(MPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mpp)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(MPP_LIBS ${MPP_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librockchip_mpp.so)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(MPP_LIBS ${MPP_PATH}/${CMAKE_SYSTEM_NAME}/${CMAKE_ANDROID_ARCH_ABI}/libmpp.so)
endif()

include_directories(${MPP_PATH}/include)

# zlmediakit
set(ZLMEDIAKIT_PATH ${CMAKE_SOURCE_DIR}/3rdparty/zlmediakit)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include_directories(${ZLMEDIAKIT_PATH}/include)
  set(ZLMEDIAKIT_LIBS ${ZLMEDIAKIT_PATH}/${LIB_ARCH}/libmk_api.so)
endif()

if(ZLMEDIAKIT_LIBS)
  add_definitions(-DBUILD_VIDEO_RTSP)
endif()

# ros
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  std_msgs
  image_transport
)
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


set(CMAKE_INSTALL_RPATH "lib")

# rknn_yolov5_multithreading_jz
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(rknn_yolov5_multithreading_jz
        src/main.cc
        src/preprocess.cc
        src/postprocess.cc
        src/rkYolov5s.cc
)
# catkin_package(
#   INCLUDE_DIRS include
#   LIBRARIES my_ros_pkg
#   CATKIN_DEPENDS roscpp std_msgs
# )

# # 包含头文件路径
# include_directories(
#   ${catkin_INCLUDE_DIRS}
# )

target_link_libraries(rknn_yolov5_multithreading_jz
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_multithreading_jz_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_yolov5_multithreading_jz DESTINATION ./)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY model/${TARGET_SOC} DESTINATION ./model)
file(GLOB IMAGE_FILES "model/*.jpg")
file(GLOB LABEL_FILE "model/*.txt")
install(FILES ${IMAGE_FILES} DESTINATION ./model/)
install(FILES ${LABEL_FILE} DESTINATION ./model/)

if(ZLMEDIAKIT_LIBS)
  install(PROGRAMS ${ZLMEDIAKIT_LIBS} DESTINATION lib)
endif()
