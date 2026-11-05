package org.opmd.followup.demo;

import android.app.Activity;
import android.content.Intent;
import androidx.activity.result.ActivityResult;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.ActivityCallback;
import com.getcapacitor.annotation.CapacitorPlugin;

@CapacitorPlugin(name = "GuidedCamera")
public class GuidedCameraPlugin extends Plugin {
    @PluginMethod public void takePhoto(PluginCall call) {
        startActivityForResult(call, new Intent(getContext(), GuidedCameraActivity.class), "captured");
    }
    @ActivityCallback private void captured(PluginCall call, ActivityResult result) {
        if (call == null) return;
        Intent data = result.getData();
        if (result.getResultCode() != Activity.RESULT_OK || data == null) {
            String reason = data == null ? null : data.getStringExtra("error");
            call.reject(reason == null ? "cancelled" : reason); return;
        }
        String path = data.getStringExtra("path");
        if (path == null) { call.reject("拍摄结果无法读取，请使用系统相机或相册"); return; }
        JSObject value = new JSObject(); value.put("path", path); value.put("format", "jpg");
        call.resolve(value);
    }
}
