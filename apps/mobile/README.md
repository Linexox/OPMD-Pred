# 手机APP

Vue 3＋TypeScript＋Capacitor Android。更新：2026年11月6日，当前交付包为1.3.9。当前线上模型为五个完整二分类模型各占20%汇总；本次更新病损部位示意图和按部位显示的拍照指引。

- 人可安装的包：医工/交付成果/APP/口腔随访-1.3.9.apk。
- 下载页面：http://www.ming56.cn:9020/download。
- 模拟器启动：医工目录的启动安卓APP模拟器.cmd，调用本目录launch-emulator.ps1；默认使用Linux服务，-LocalService才启动Windows旧三分类开发服务。

当前页面支持资料录入、照片拍摄/选择、报告、患者分析、医生反馈、补资料及复诊对照。二分类结果显示正常/轻度或中度及以上，并用grade_reference展示三个真实病理等级的中位数参考位置。

## 开发与构建

npm依赖按package-lock.json锁定。先看根目录[最小部署指南](../../DEPLOYMENT.md)，准备Node.js 24并运行npm ci。npm run dev启动本地前端，npm run build进行类型检查与页面构建。API地址和连接口令由连接设置或.env.local中的VITE_DEMO_SERVICE_URL、VITE_DEMO_PAIRING_TOKEN提供，不写入跟踪源码。Git克隆本身不会取得连接口令。

Windows打包入口build-android.ps1：默认读取医工/运行数据/pairing-token.txt，使用公网服务地址；-ServiceUrl可更换地址，-NoDefaultConnection生成无预填凭据的包。内置连接的包只供团队内部使用。

构建输出仍在android/app/build/outputs/apk/debug/app-debug.apk。脚本自动更新医工/运行数据/distribution/app-debug.apk及latest.json；模拟器优先读取这个固定副本。源码、调试构建输出和供人安装的交付包用途不同，不手工保留一堆日期APK。

构建缓存固定放医工/开发缓存，Android SDK在D:/.Workspace/android-sdk-opmd，Java默认D:/.APP/java21。开发缓存/android/debug.keystore是现有APK签名密钥，必须保留，不能随整个缓存目录删除。

## 数据和接口

开发网页版通过Vite转发到.env.local指定的后端；该转发不改变APK的连接地址。页面按需加载照片，同时最多下载两张，给打开记录和分析请求保留连接。

前端通过src/api.ts调用后端；接口与返回结构见同级server/README.md。提交时不传电脑文件路径，而传附件ID。手机草稿和待上传文件在Android应用私有目录，已提交记录由服务端保存。

当前Linux记录在远程runtime，Windows开发记录在医工/运行数据。模拟器AVD磁盘位置本次未改变；卸载APP会影响客户端私有数据，文件整理没有卸载或清空应用。
