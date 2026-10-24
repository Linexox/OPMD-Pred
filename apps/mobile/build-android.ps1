param([string]$Sdk='D:\.Workspace\android-sdk-opmd',[string]$Java='D:\.APP\java21',[string]$ServiceUrl='http://www.ming56.cn:9020',[switch]$NoDefaultConnection,[switch]$SkipInstall)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'apk-info.ps1')
$projectRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$dataDir=Join-Path (Split-Path $projectRoot -Parent) 'OPMDMobileDemo'
$buildCache=Join-Path $dataDir 'build-cache'
$env:GRADLE_USER_HOME=Join-Path $buildCache 'gradle'
$env:ANDROID_USER_HOME=Join-Path $buildCache 'android'
$env:npm_config_cache=Join-Path $buildCache 'npm'
New-Item -ItemType Directory -Path $env:ANDROID_USER_HOME -Force | Out-Null
# Keep the existing signing identity while writing new Android state on D:.
$debugKey=Join-Path $env:ANDROID_USER_HOME 'debug.keystore'
$existingDebugKey=Join-Path $env:USERPROFILE '.android/debug.keystore'
if (-not (Test-Path -LiteralPath $debugKey) -and (Test-Path -LiteralPath $existingDebugKey)) {
  Copy-Item -LiteralPath $existingDebugKey -Destination $debugKey
}
$env:JAVA_HOME=$Java
$env:ANDROID_HOME=$Sdk
$env:Path='D:\.APP\nodejs24;'+$Java+'\bin;'+$env:Path
Push-Location $PSScriptRoot
try {
  if (-not $NoDefaultConnection) {
    if (-not $ServiceUrl) {
      $address=Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -eq 'WLAN' -and $_.IPAddress -notlike '169.254.*' } | Select-Object -First 1 -ExpandProperty IPAddress
      if (-not $address) { throw '未找到Wi-Fi地址，请用-ServiceUrl指定电脑服务地址' }
      $ServiceUrl='http://'+$address+':8765'
    }
    $tokenPath=Join-Path $dataDir 'pairing-token.txt'
    if (-not (Test-Path -LiteralPath $tokenPath)) { throw '请先启动手机APP服务，生成本机配对口令' }
    $env:VITE_DEMO_SERVICE_URL=$ServiceUrl
    $env:VITE_DEMO_PAIRING_TOKEN=(Get-Content -LiteralPath $tokenPath -Raw -Encoding UTF8).Trim()
    if (-not $env:VITE_DEMO_PAIRING_TOKEN) { throw '本机连接口令为空，请检查服务配置' }
    Write-Host ('已内置服务连接：'+$ServiceUrl)
  } else {
    $env:VITE_DEMO_SERVICE_URL=''
    $env:VITE_DEMO_PAIRING_TOKEN=''
  }
  if (-not $SkipInstall) {
    & 'D:\.APP\nodejs24\npm.cmd' ci --no-audit --no-fund
    if ($LASTEXITCODE) { throw 'npm ci 失败' }
  }
  & 'D:\.APP\nodejs24\npm.cmd' run build
  if ($LASTEXITCODE) { throw '页面构建失败' }
  & 'D:\.APP\nodejs24\npx.cmd' cap sync android
  if ($LASTEXITCODE) { throw 'Android同步失败' }
  ('sdk.dir='+$Sdk.Replace('\','/')) | Set-Content -LiteralPath android/local.properties -Encoding ascii
  & ./android/gradlew.bat -p android assembleDebug --console=plain
  if ($LASTEXITCODE) { throw 'APK构建失败' }
  $apkInfo=Get-ApkInfo -ApkPath (Join-Path $PSScriptRoot 'android/app/build/outputs/apk/debug/app-debug.apk') -SdkRoot $Sdk
  $distribution=Join-Path $dataDir 'distribution'
  New-Item -ItemType Directory -Path $distribution -Force | Out-Null
  $incoming=Join-Path $distribution 'app-debug.incoming.apk'
  $published=Join-Path $distribution 'app-debug.apk'
  Copy-Item -LiteralPath $apkInfo.Path -Destination $incoming -Force
  if ((Get-FileHash -LiteralPath $incoming -Algorithm SHA256).Hash.ToLowerInvariant() -ne $apkInfo.SHA256) { throw 'APK发布副本校验失败。' }
  Move-Item -LiteralPath $incoming -Destination $published -Force
  [ordered]@{versionCode=$apkInfo.VersionCode;versionName=$apkInfo.VersionName;sha256=$apkInfo.SHA256;apk='app-debug.apk';serviceUrl=$(if ($NoDefaultConnection) { '' } else { $ServiceUrl })} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $distribution 'latest.json') -Encoding UTF8
  Write-Host ('APK已发布：'+$published+'，版本'+$apkInfo.VersionName)
} finally {
  Remove-Item Env:VITE_DEMO_SERVICE_URL -ErrorAction SilentlyContinue
  Remove-Item Env:VITE_DEMO_PAIRING_TOKEN -ErrorAction SilentlyContinue
  Pop-Location
}
