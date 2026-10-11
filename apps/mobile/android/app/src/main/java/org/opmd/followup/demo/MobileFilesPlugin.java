package org.opmd.followup.demo;

import android.app.Activity;
import android.content.Intent;
import android.content.Context;
import android.view.inputmethod.InputMethodManager;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;
import androidx.activity.result.ActivityResult;
import androidx.core.content.FileProvider;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.ActivityCallback;
import com.getcapacitor.annotation.CapacitorPlugin;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.UUID;

@CapacitorPlugin(name = "MobileFiles")
public class MobileFilesPlugin extends Plugin {
    @PluginMethod public void hideKeyboard(PluginCall call) {
        getActivity().runOnUiThread(() -> {
            InputMethodManager ime = (InputMethodManager) getContext().getSystemService(Context.INPUT_METHOD_SERVICE);
            if (ime != null) ime.hideSoftInputFromWindow(getBridge().getWebView().getWindowToken(), 0);
            call.resolve();
        });
    }
    @PluginMethod public void pickReport(PluginCall call) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.setType("*/*");
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{"application/pdf", "image/jpeg", "image/png"});
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(call, intent, "reportPicked");
    }

    @ActivityCallback private void reportPicked(PluginCall call, ActivityResult result) {
        if (call == null) return;
        if (result.getResultCode() != Activity.RESULT_OK || result.getData() == null) {
            call.reject("cancelled"); return;
        }
        Uri uri = result.getData().getData();
        if (uri == null) { call.reject("文件不可读取"); return; }
        String name = "检测报告.pdf";
        try (Cursor cursor = getContext().getContentResolver().query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) name = cursor.getString(index);
            }
        } catch (Exception ignored) { }
        File target = new File(getContext().getCacheDir(), "report-" + UUID.randomUUID());
        try (InputStream input = getContext().getContentResolver().openInputStream(uri);
             FileOutputStream output = new FileOutputStream(target)) {
            if (input == null) throw new Exception("文件不可读取");
            byte[] buffer = new byte[65536]; int n; long total = 0;
            while ((n = input.read(buffer)) != -1) {
                total += n;
                if (total > 20000000) throw new Exception("文件超过20MB");
                output.write(buffer, 0, n);
            }
            JSObject value = new JSObject(); value.put("name", name); value.put("path", Uri.fromFile(target).toString()); call.resolve(value);
        } catch (Exception e) { target.delete(); call.reject(e.getMessage()); }
    }

    @PluginMethod public void openFile(PluginCall call) {
        try {
            File file = new File(Uri.parse(call.getString("path", "")).getPath()).getCanonicalFile();
            String allowed = getContext().getFilesDir().getCanonicalPath() + File.separator;
            if (!file.getPath().startsWith(allowed)) throw new Exception("附件路径无效");
            Uri uri = FileProvider.getUriForFile(getContext(), getContext().getPackageName() + ".fileprovider", file);
            Intent intent = new Intent(Intent.ACTION_VIEW).setDataAndType(uri, call.getString("mime", "application/pdf"));
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            getActivity().startActivity(intent); call.resolve();
        } catch (Exception e) { call.reject("请安装可打开该报告的应用后重试"); }
    }
}
