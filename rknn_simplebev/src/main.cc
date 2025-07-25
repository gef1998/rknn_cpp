#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include "simplebev.hpp"
#include "rknnPool.hpp"
#include <dirent.h>
#include <vector>
#include <string>
#include <algorithm>

static std::vector<std::string> split(const std::string& str, const std::string& pattern)
{
  std::vector<std::string> res;
  if (str == "")
    return res;
  std::string strs = str + pattern;
  size_t      pos  = strs.find(pattern);
  while (pos != strs.npos) {
    std::string temp = strs.substr(0, pos);
    res.push_back(temp);
    strs = strs.substr(pos + 1, strs.size());
    pos  = strs.find(pattern);
  }
  return res;
}


int main(int argc, char **argv)
{
      SimpleBEV::ModelPaths modelPaths{
        argv[1], // encoder
        argv[2], // grid_sample
        argv[4], // decoder
        argv[3]  // flat_idx
    };
    // 初始化rknn线程池/Initialize the rknn thread pool
    int threadNum = 1;
    rknnPool<SimpleBEV, unsigned char*, int> testPool(encoder_path, grid_sample_path, flat_id_path, decoder_path, threadNum);
    if (testPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }

    std::vector<std::string> binFiles;
    DIR *dir;
    struct dirent *ent;
    std::string folderPath = "/home/jz/catkin_ws/src/rknn_cpp/rknn_simplebev/npy/test_bins";
    if ((dir = opendir(folderPath.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename.size() > 4 && 
                filename.substr(filename.size() - 4) == ".bin") {
                binFiles.push_back(folderPath + "/" + filename);
            }
        }
        closedir(dir);}else {
          perror("无法打开目录");
      }
    std::sort(binFiles.begin(), binFiles.end());

    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    int frames = 0;
    auto beforeTime = startTime;

    // 按文件名排序
    // std::string inputpath = "/home/jz/catkin_ws/src/rknn_cpp/rknn_simplebev/npy/test/1.bin";
    int input_size = testPool.get_input_size();
    for (const auto& inputpath : binFiles){
      // 处理每个bin文件
      unsigned char* input_data = nullptr;
      // Load input
      input_data = new unsigned char[input_size];
      printf("%s\n", inputpath.c_str());
      FILE* fp = fopen(inputpath.c_str(), "rb");
      if (fp == NULL) {
        perror("open failed!");
        return -1;
      }
      fread(input_data, input_size, 1, fp);
      fclose(fp);
        // 原处理逻辑
      if (testPool.put(input_data) != 0) break;
      int res;
      if (frames >= threadNum && testPool.get(res) != 0){
        printf("empty");
        break;
      }else{
        frames++;
      }

      // 帧率计算保持不变
      if (frames % 120 == 0) {
          gettimeofday(&time, nullptr);
          auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
          printf("120帧内平均帧率:\t %f fps/s\n", 
              120.0 / float(currentTime - beforeTime) * 1000.0);
          beforeTime = currentTime;
      }
  }
    // 清空rknn线程池/Clear the thread pool
    while (true)
    {
        int res;
        if (testPool.get(res) != 0)
            break;
        frames++;
    }

    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);

    return 0;
}