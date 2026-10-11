param([string]$Sdk='D:\.Workspace\android-sdk-opmd',[string]$Java='D:\.APP\java21')
$ErrorActionPreference='Stop'
$env:JAVA_HOME=$Java
$env:ANDROID_HOME=$Sdk
$env:Path='D:\.APP\nodejs24;'+$Java+'\bin;'+$env:Path
Push-Location $PSScriptRoot
try {
  & 'D:\.APP\nodejs24\npm.cmd' ci --no-audit --no-fund
  if ($LASTEXITCODE) { throw 'npm ci 失败' }
  & 'D:\.APP\nodejs24\npm.cmd' run build
  if ($LASTEXITCODE) { throw '页面构建失败' }
  & 'D:\.APP\nodejs24\npx.cmd' cap sync android
  if ($LASTEXITCODE) { throw 'Android同步失败' }
  ('sdk.dir='+$Sdk.Replace('\','/')) | Set-Content -LiteralPath android/local.properties -Encoding ascii
  & ./android/gradlew.bat -p android assembleDebug --console=plain
  if ($LASTEXITCODE) { throw 'APK构建失败' }
  Write-Host 'APK：android/app/build/outputs/apk/debug/app-debug.apk'
} finally { Pop-Location }
