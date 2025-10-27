export GCC_COMPILER=/usr/bin/aarch64-linux-gnu

export LD_LIBRARY_PATH=/opt/jz/async-service/bin:$LD_LIBRARY_PATH

./build-linux.sh -t rk3588 -a aarch64 -b Release

./launch_multisensor.sh ~/catkin_ws/src/rknn_cpp/rknn_simplebev/model/RK3588/encoder_4cam.rknn ~/catkin_ws/src/rknn_cpp/rknn_simplebev/model/RK3588/grid_sample_4cam.rknn "/home/jz/catkin_ws/src/rknn_cpp/rknn_simplebev/npy/flat_idx_4cam.bin" ~/catkin_ws/src/rknn_cpp/rknn_simplebev/model/RK3588/decoder_4cam.rknn  ~/catkin_ws/src/rknn_cpp/rknn_simplebev/model/RK3588/laser_4cam.rknn