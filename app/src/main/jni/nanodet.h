#ifndef NANODET_H
#define NANODET_H

#include <opencv2/opencv.hpp>
#include <android/log.h>

#include <net.h>

#include "cpu.h"
#include "layer.h"

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class NanoDet {
public:
    NanoDet();

    int load(AAssetManager* mgr, bool use_gpu = false);

    int detect(cv::Mat &rgb);

private:
    ncnn::Net yolopv2;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;

    const int target_size = 320;
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
};

#endif // NANODET_H
