#pragma once

#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"

using namespace cv;
using namespace std;

enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack
{
public:
	STrack(vector<float> tlwh_, float score);
	~STrack();

	vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
	void static multi_predict(vector<STrack*> &stracks, byte_kalman::KalmanFilter &kalman_filter);
	void static_tlwh();
	void static_tlbr();
	vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
	vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();
	
	void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
	void re_activate(STrack &new_track, int frame_id, bool new_id = false);
	void update(STrack &new_track, int frame_id);

public:
	// 状态标志
	bool is_activated;        // 是否已激活（区分新轨迹和稳定轨迹）
	int track_id;            // 全局唯一的轨迹ID
	int state;               // 轨迹状态（New/Tracked/Lost/Removed）

	// 边界框表示（三种格式）
	vector<float> _tlwh;     // 原始检测框 (top-left x, y, width, height)
	vector<float> tlwh;      // 当前预测框 (从卡尔曼滤波器更新)
	vector<float> tlbr;      // 边界框 (top-left, bottom-right)

	// 时间信息
	int frame_id;            // 最后更新的帧ID
	int tracklet_len;        // 轨迹长度（连续跟踪帧数）
	int start_frame;         // 轨迹开始帧

	// 卡尔曼滤波器状态
	KAL_MEAN mean;           // 8维状态向量 [x, y, a, h, vx, vy, va, vh]
	KAL_COVA covariance;     // 8×8 协方差矩阵

	// 置信度
	float score;             // 检测置信度

private:
	byte_kalman::KalmanFilter kalman_filter; // 卡尔曼滤波器实例
};