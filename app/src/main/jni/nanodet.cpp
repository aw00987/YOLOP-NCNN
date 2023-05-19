#include "nanodet.h"

#define MAX_STRIDE 32
#define nms_threshold 0.45f
#define prob_threshold 0.30f

static void slice(const ncnn::Mat &in, ncnn::Mat &out, int start, int end, int axis) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_arithmetic = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_packed = true;

    ncnn::Layer *op = ncnn::create_layer("Crop");

    // set param
    ncnn::ParamDict pd;

    ncnn::Mat axes = ncnn::Mat(1);
    axes.fill(axis);
    ncnn::Mat ends = ncnn::Mat(1);
    ends.fill(end);
    ncnn::Mat starts = ncnn::Mat(1);
    starts.fill(start);
    pd.set(9, starts);// start
    pd.set(10, ends);// end
    pd.set(11, axes);//axes

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

static void interp(const ncnn::Mat &in, const float &scale, const int &out_w, const int &out_h,
                   ncnn::Mat &out) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_arithmetic = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_packed = true;

    ncnn::Layer *op = ncnn::create_layer("Interp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);// resize_type
    pd.set(1, scale);// height_scale
    pd.set(2, scale);// width_scale
    pd.set(3, out_h);// height
    pd.set(4, out_w);// width

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

static inline float intersection_area(const Object &a, const Object &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat &anchors, int stride, const ncnn::Mat &in_pad,
                               const ncnn::Mat &feat_blob, std::vector<Object> &objects) {
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h) {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    } else {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                const float *featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++) {
                    float score = featptr[5 + k];
                    if (score > class_score) {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold) {

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}

NanoDet::NanoDet() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int NanoDet::load(AAssetManager *mgr, bool use_gpu) {
    yolopv2.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolopv2.opt = ncnn::Option();
    yolopv2.opt.use_fp16_arithmetic = true;
    yolopv2.opt.use_fp16_packed = true;
    yolopv2.opt.use_fp16_storage = true;
#if NCNN_VULKAN
    yolopv2.opt.use_vulkan_compute = use_gpu;
#endif

    yolopv2.opt.num_threads = ncnn::get_big_cpu_count();
    yolopv2.opt.blob_allocator = &blob_pool_allocator;
    yolopv2.opt.workspace_allocator = &workspace_pool_allocator;

    yolopv2.load_param(mgr, "yolopv2-opt.param");
    yolopv2.load_model(mgr, "yolopv2-opt.bin");

    return 0;
}

int NanoDet::detect(cv::Mat &rgb) {

    // 检测结果
    std::vector<Object> objects;
    ncnn::Mat da_seg_mask, ll_seg_mask;

    // 图像信息
    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // 图像缩放
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
        h = target_size;
        w = w * scale;
    }
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

    //输入tmp图像
    ncnn::Mat in, in_pad;

    //padding
    in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h,
                                       w, h);
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 114.f);
    in_pad.substract_mean_normalize(0, norm_vals);//todo:mean_vals?

    //run network
    {
        std::vector<Object> proposals;
        ncnn::Mat da, ll;

        ncnn::Extractor ex = yolopv2.create_extractor();
        ex.input("images", in_pad);
        // stride 8
        {
            ncnn::Mat out;
            ex.extract("det0", out);

            ncnn::Mat anchors(6);
            anchors[0] = 12.f;
            anchors[1] = 16.f;
            anchors[2] = 19.f;
            anchors[3] = 36.f;
            anchors[4] = 40.f;
            anchors[5] = 28.f;

            std::vector<Object> objects8;
            generate_proposals(anchors, 8, in, out, objects8);

            proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        }
        // stride 16
        {
            ncnn::Mat out;
            ex.extract("det1", out);

            ncnn::Mat anchors(6);
            anchors[0] = 36.f;
            anchors[1] = 75.f;
            anchors[2] = 76.f;
            anchors[3] = 55.f;
            anchors[4] = 72.f;
            anchors[5] = 146.f;

            std::vector<Object> objects16;
            generate_proposals(anchors, 16, in, out, objects16);

            proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        }
        // stride 32
        {
            ncnn::Mat out;
            ex.extract("det2", out);

            ncnn::Mat anchors(6);
            anchors[0] = 142.f;
            anchors[1] = 110.f;
            anchors[2] = 192.f;
            anchors[3] = 243.f;
            anchors[4] = 459.f;
            anchors[5] = 401.f;

            std::vector<Object> objects32;
            generate_proposals(anchors, 32, in, out, objects32);

            proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        }
        //make mask for da,ll
        {
            ex.extract("677", da);
            ex.extract("769", ll);
            slice(da, da_seg_mask, hpad / 2, in_pad.h - hpad / 2, 1);
            slice(ll, ll_seg_mask, hpad / 2, in_pad.h - hpad / 2, 1);
            slice(da_seg_mask, da_seg_mask, wpad / 2, in_pad.w - wpad / 2, 2);
            slice(ll_seg_mask, ll_seg_mask, wpad / 2, in_pad.w - wpad / 2, 2);
            interp(da_seg_mask, 1 / scale, 0, 0, da_seg_mask);
            interp(ll_seg_mask, 1 / scale, 0, 0, ll_seg_mask);
        }
        //filter objects from proposals
        {
            // sort all proposals by score from highest to lowest
            qsort_descent_inplace(proposals);

            // apply nms with nms_threshold
            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked);

            int count = picked.size();

            objects.resize(count);
            for (int i = 0; i < count; i++) {
                objects[i] = proposals[picked[i]];

                // adjust offset to original unpadded
                float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
                float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
                float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
                float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

                // clip
                x0 = std::max(std::min(x0, (float) (img_w - 1)), 0.f);
                y0 = std::max(std::min(y0, (float) (img_h - 1)), 0.f);
                x1 = std::max(std::min(x1, (float) (img_w - 1)), 0.f);
                y1 = std::max(std::min(y1, (float) (img_h - 1)), 0.f);

                objects[i].rect.x = x0;
                objects[i].rect.y = y0;
                objects[i].rect.width = x1 - x0;
                objects[i].rect.height = y1 - y0;
            }
        }
    }
    // draw
    {
        for (auto &obj: objects) {
            cv::rectangle(rgb, obj.rect, cv::Scalar(255, 255, 0));
        }
        const float *da_ptr = (float *) da_seg_mask.data;
        const float *ll_ptr = (float *) ll_seg_mask.data;
        int ww = da_seg_mask.w;
        int hh = da_seg_mask.h;
        for (int i = 0; i < hh; i++) {
            auto *image_ptr = rgb.ptr<cv::Vec3b>(i);
            for (int j = 0; j < ww; j++) {
                if (da_ptr[i * ww + j] < da_ptr[ww * hh + i * ww + j]) {
                    image_ptr[j] = cv::Vec3b(0, 255, 0);
                }

                if (std::round(ll_ptr[i * ww + j]) == 1.0) {
                    image_ptr[j] = cv::Vec3b(255, 0, 0);
                }
            }
        }
    }

    return 0;
}
