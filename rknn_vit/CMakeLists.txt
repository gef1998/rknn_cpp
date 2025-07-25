cmake_minimum_required(VERSION 3.5.1)

project(rknn_vit)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

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


include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/../3rdparty/)
include_directories(${CMAKE_SOURCE_DIR}/include)

# opencv
find_package(OpenCV REQUIRED)

set(CMAKE_INSTALL_RPATH "lib")

add_executable(rknn_vit
        src/main.cc
        src/label_loader.cc
)

target_link_libraries(rknn_vit
	${RKNN_RT_LIB}
  ${OpenCV_LIBS}
)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
	target_link_libraries(rknn_vit log)
endif()

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_vit_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_vit DESTINATION ./)
install(DIRECTORY model/${TARGET_SOC} DESTINATION ./model)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
file(GLOB DATA_FILES "model/*.bin")
install(FILES ${DATA_FILES} DESTINATION ./model/)
