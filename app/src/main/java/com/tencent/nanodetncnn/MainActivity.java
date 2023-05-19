package com.tencent.nanodetncnn;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

public class MainActivity extends Activity {
    // 初始化的一些变量
    public static final int REQUEST_CAMERA = 100;
    private int core = 0; // 记录当前内核 0为cpu，1为gpu
    private final NanoDetNcnn nanodetncnn = new NanoDetNcnn();

    // 初始化函数
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main); // 加载布局

        Button btnCPU = (Button) findViewById(R.id.btn_cpu);
        Button btnGPU = (Button) findViewById(R.id.btn_gpu);
        SurfaceView cameraView = (SurfaceView) findViewById(R.id.view_camera);// 预览界面

        btnCPU.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (core != 0) {
                    core = 0;
                    nanodetncnn.loadModel(getAssets(), core);
                }
            }
        });

        btnGPU.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (core != 1) {
                    core = 1;
                    nanodetncnn.loadModel(getAssets(), core);
                }
            }
        });

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);// 保持屏幕唤醒不锁屏
        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);// 设置颜色格式为RGBA8888
        cameraView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {
            }

            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {// surface尺寸发生改变的时候调用，如横竖屏切换
                nanodetncnn.setOutputWindow(holder.getSurface());
            }

            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {
            }
        }); //绑定兼容处理回调函数

        System.out.println("hhh1");
        // 所有初始化完成后，加载模型
        nanodetncnn.loadModel(getAssets(), core);
        System.out.println("hhh2");
    }

    // onPause方法，app被覆盖后,关闭摄像头
    @Override
    public void onPause() {
        super.onPause();
        nanodetncnn.closeCamera();
    }

    // onResume方法，app被覆盖恢复后重新开启摄像头
    @Override
    public void onResume() {
        super.onResume();
        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }
        nanodetncnn.openCamera();
    }
}