#ifndef RKNNPOOL_H
#define RKNNPOOL_H

#include "ThreadPool.hpp"
#include <vector>
#include <iostream>
#include <mutex>
#include <queue>
#include <memory>

// rknnModel模型类, inputType模型输入类型, outputType模型输出类型
template <typename rknnModel, typename inputType, typename outputType>
class rknnPool
{
private:
    int threadNum;
    std::string encoderPath, grid_samplePath, flat_idxPath, decoderPath;
    rknn_tensor_mem *output_mems_preprosess[2];
    long long id;
    std::mutex idMtx, queueMtx;
    std::unique_ptr<dpool::ThreadPool> pool;
    std::queue<std::future<outputType>> futs;
    std::vector<std::shared_ptr<rknnModel>> models;

    protected:
    int getModelId();

public:
    rknnPool(const std::string encoderPath, 
            const std::string preprosessPath,
            const std::string grid_samplePath,
            const std::string decoderPath,
            int threadNum);
    int init();
    // 模型推理/Model inference
    int put(inputType inputData);
    // 获取推理结果/Get the results of your inference
    int get(outputType &outputData);
    int get_input_size();
    ~rknnPool();
};

template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::rknnPool(const std::string encoderPath, 
                                                    const std::string grid_samplePath,
                                                    const std::string flat_id_path,
                                                    const std::string decoder_path,
                                                    int threadNum)
{
    this->encoderPath = encoderPath;
    this->grid_samplePath = grid_samplePath;
    this->flat_idxPath = flat_id_path;
    this->decoderPath = decoder_path;
    this->threadNum = threadNum;
    this->id = 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::init()
{
    try
    {
        this->pool = std::make_unique<dpool::ThreadPool>(this->threadNum);
        for (int i = 0; i < this->threadNum; i++)
            models.push_back(std::make_shared<rknnModel>(this->encoderPath.c_str(),
                                                         this->grid_samplePath.c_str(),
                                                         this->flat_idxPath.c_str(),
                                                         this->decoderPath.c_str()));
    }
    catch (const std::bad_alloc &e)
    {
        std::cout << "Out of memory: " << e.what() << std::endl;
        return -1;
    }
    // 初始化模型/Initialize the model
    for (int i = 0, ret = 0; i < threadNum; i++)
    {
        ret = models[i]->init(models[0]->get_encoder_pctx(), models[0]->get_grid_sample_pctx(), models[0]->get_decoder_pctx(), models[0]->get_flat_idx_mems(), i != 0);
        if (ret != 0)
            return ret;
    }

    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::getModelId()
{
    std::lock_guard<std::mutex> lock(idMtx);
    int modelId = id % threadNum;
    id++;
    return modelId;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::put(inputType inputData)
{
    std::lock_guard<std::mutex> lock(queueMtx);
    futs.push(pool->submit(&rknnModel::infer, models[this->getModelId()], inputData));
    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::get(outputType &outputData)
{
    std::lock_guard<std::mutex> lock(queueMtx);
      if(futs.empty() == true)
        return 1;
    outputData = futs.front().get();
    futs.pop();
    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::get_input_size()
{
    return models[0]->get_input_size();
}

template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::~rknnPool()
{
    while (!futs.empty())
    {
        outputType temp = futs.front().get();
        futs.pop();
    }
}

#endif
