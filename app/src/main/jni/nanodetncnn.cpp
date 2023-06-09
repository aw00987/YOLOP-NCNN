#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "nanodet.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static int draw_unsupported(cv::Mat &rgb) {
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat &rgb) {
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f) {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--) {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f) {
            return 0;
        }

        for (int i = 0; i < 10; i++) {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static NanoDet *g_nanodet = 0;
static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow {
public:
    virtual void on_image_render(cv::Mat &rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat &rgb) const {
    {
        ncnn::MutexLockGuard g(lock);
        if (g_nanodet) {
            g_nanodet->detect(rgb);    //检测并绘制图案
        } else {
            draw_unsupported(rgb);
        }
    }
    draw_fps(rgb);
}

static MyNdkCamera *g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    g_camera = new MyNdkCamera;
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    {
        ncnn::MutexLockGuard g(lock);
        delete g_nanodet;
        g_nanodet = 0;
    }
    delete g_camera;
    g_camera = 0;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_nanodetncnn_NanoDetNcnn_loadModel(JNIEnv *env, jobject thiz,
                                                   jobject assetManager, jint core) {
    if (core < 0 || core > 1) {
        return JNI_FALSE;
    }
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    bool use_gpu = (int) core == 1;
    {
        ncnn::MutexLockGuard g(lock);
        if (use_gpu && ncnn::get_gpu_count() == 0) {// no gpu
            delete g_nanodet;
            g_nanodet = 0;
        } else {
            if (!g_nanodet) {
                g_nanodet = new NanoDet();
            }
            g_nanodet->load(mgr, use_gpu);
        }
    }
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_nanodetncnn_NanoDetNcnn_openCamera(JNIEnv *env, jobject thiz) {
    g_camera->open(1);//todo: simplify
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_nanodetncnn_NanoDetNcnn_closeCamera(JNIEnv *env, jobject thiz) {
    g_camera->close();
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_nanodetncnn_NanoDetNcnn_setOutputWindow(JNIEnv *env, jobject thiz,
                                                         jobject surface) {
    ANativeWindow *win = ANativeWindow_fromSurface(env, surface);
    g_camera->set_window(win);
    return JNI_TRUE;
}

}
