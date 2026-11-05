package org.opmd.followup.demo;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Independent capture page; returns a file only after successful explicit capture. */
public class GuidedCameraActivity extends AppCompatActivity {
    private final ExecutorService analyzer = Executors.newSingleThreadExecutor();
    private final CaptureHints hints = new CaptureHints();
    private ProcessCameraProvider provider;
    private PreviewView preview;
    private TextView message;
    private Button capture;
    private ImageCapture imageCapture;
    private File pending;
    private long lastFrame;
    private int previousWidth, previousHeight;
    private boolean takingPhoto;
    private final ActivityResultLauncher<String> permission = registerForActivityResult(
        new ActivityResultContracts.RequestPermission(), granted -> {
            if (granted) startCamera(); else fail("未允许相机权限，请使用系统相机或相册");
        });

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setBackgroundColor(Color.BLACK);
        ViewCompat.setOnApplyWindowInsetsListener(layout, (view, insets) -> {
            androidx.core.graphics.Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom); return insets;
        });
        preview = new PreviewView(this);
        preview.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
        layout.addView(preview, new LinearLayout.LayoutParams(-1, 0, 1));
        message = new TextView(this); message.setTextColor(Color.WHITE);
        message.setTextSize(18); message.setPadding(24, 20, 24, 20);
        message.setText("正在打开相机…");
        message.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
        layout.addView(message);
        LinearLayout buttons = new LinearLayout(this);
        Button cancel = new Button(this); cancel.setText("返回系统相机或相册");
        cancel.setOnClickListener(v -> { if (!takingPhoto) finish(); });
        capture = new Button(this); capture.setText("拍照"); capture.setEnabled(false);
        capture.setOnClickListener(v -> takePhoto());
        buttons.addView(cancel, new LinearLayout.LayoutParams(0, -2, 1));
        buttons.addView(capture, new LinearLayout.LayoutParams(0, -2, 1));
        layout.addView(buttons); setContentView(layout);
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            startCamera();
        else permission.launch(Manifest.permission.CAMERA);
    }

    private void startCamera() {
        com.google.common.util.concurrent.ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            if (isFinishing() || isDestroyed()) return;
            try {
                provider = future.get();
                Preview display = new Preview.Builder().build();
                display.setSurfaceProvider(preview.getSurfaceProvider());
                imageCapture = new ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY).build();
                ImageAnalysis analysis = new ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build();
                analysis.setAnalyzer(analyzer, this::analyze);
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, display, imageCapture, analysis);
                capture.setEnabled(true);
                message.setText("请对准要记录的位置");
            } catch (Exception e) { fail("相机暂时无法打开，请使用系统相机或相册"); }
        }, ContextCompat.getMainExecutor(this));
    }

    private void analyze(ImageProxy image) {
        try {
            long now = SystemClock.elapsedRealtime();
            if (now - lastFrame < 250) return;
            lastFrame = now;
            Rect crop = image.getCropRect();
            if (crop.width() < 1 || crop.height() < 1) return;
            ImageProxy.PlaneProxy plane = image.getPlanes()[0];
            ByteBuffer pixels = plane.getBuffer(); int start = pixels.position();
            int[] samples = new int[32 * 24];
            for (int y = 0; y < 24; y++) for (int x = 0; x < 32; x++) {
                int px = crop.left + x * crop.width() / 32;
                int py = crop.top + y * crop.height() / 24;
                samples[y * 32 + x] = pixels.get(start + py * plane.getRowStride() + px * plane.getPixelStride()) & 255;
            }
            // A geometry change is not evidence of phone movement; skip one sample.
            if (previousWidth != crop.width() || previousHeight != crop.height()) {
                previousWidth = crop.width(); previousHeight = crop.height(); hints.reset(); return;
            }
            String text = hints.update(samples);
            runOnUiThread(() -> { if (!isFinishing() && !isDestroyed() && !takingPhoto) message.setText(text); });
        } catch (RuntimeException e) {
            runOnUiThread(() -> { if (!isFinishing() && !takingPhoto) message.setText("暂时无法提供提示，可以手动拍照"); });
        } finally { image.close(); }
    }

    private void takePhoto() {
        if (imageCapture == null || takingPhoto) return;
        takingPhoto = true; capture.setEnabled(false); message.setText("正在保存照片…");
        imageCapture.setTargetRotation(preview.getDisplay().getRotation());
        pending = new File(getCacheDir(), "guided-photo-" + UUID.randomUUID() + ".jpg");
        File target = pending;
        imageCapture.takePicture(new ImageCapture.OutputFileOptions.Builder(target).build(),
            ContextCompat.getMainExecutor(this), new ImageCapture.OnImageSavedCallback() {
                @Override public void onImageSaved(ImageCapture.OutputFileResults output) {
                    if (isFinishing() || isDestroyed()) { target.delete(); return; }
                    Intent result = new Intent(); result.putExtra("path", Uri.fromFile(target).toString());
                    pending = null; setResult(RESULT_OK, result); finish();
                }
                @Override public void onError(ImageCaptureException exception) {
                    target.delete(); pending = null; takingPhoto = false; capture.setEnabled(true);
                    message.setText("照片未保存，请重试或返回系统相机、相册");
                }
            });
    }

    private void fail(String reason) {
        Intent result = new Intent(); result.putExtra("error", reason);
        setResult(RESULT_CANCELED, result); finish();
    }
    @Override protected void onDestroy() {
        if (provider != null) provider.unbindAll();
        analyzer.shutdown();
        if (pending != null) pending.delete();
        super.onDestroy();
    }
}
