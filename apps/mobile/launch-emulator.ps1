param(
    [string]$Sdk='D:\.Workspace\android-sdk-opmd',
    [string]$Avd='OPMD_API36',
    [int]$Port=8765,
    [switch]$LocalService,
    [switch]$ShowPairing
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'apk-info.ps1')
$projectRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$adb=Join-Path $Sdk 'platform-tools/adb.exe'
$emulator=Join-Path $Sdk 'emulator/emulator.exe'
$dataDir=Join-Path (Split-Path $projectRoot -Parent) 'OPMDMobileDemo'
$tokenPath=Join-Path $dataDir 'pairing-token.txt'
$appId='org.opmd.followup.demo'
$apkCandidates=@(
    (Join-Path $dataDir 'distribution/app-debug.apk'),
    (Join-Path $PSScriptRoot 'android/app/build/outputs/apk/debug/app-debug.apk')
)
if (-not (Test-Path -LiteralPath $adb) -or -not (Test-Path -LiteralPath $emulator)) { throw '未找到本项目的Android SDK。' }

function Find-ProjectEmulator {
    $lines=@(& $adb devices)
    foreach ($line in $lines) {
        if ($line -match '^(emulator-\d+)\s+device\s*$') {
            $candidate=$Matches[1]
            $name=(& $adb -s $candidate shell getprop ro.boot.qemu.avd_name 2>$null | Out-String).Trim()
            if ($name -eq $Avd) { return $candidate }
        }
    }
    return $null
}
function Service-Ready {
    if (-not (Test-Path -LiteralPath $tokenPath)) { return $false }
    try {
        $pairing=(Get-Content -LiteralPath $tokenPath -Raw -Encoding UTF8).Trim()
        $health=Invoke-RestMethod -Uri ('http://127.0.0.1:'+$Port+'/api/health') -Headers @{Authorization=('Bearer '+$pairing)} -TimeoutSec 2
        return ($health.status -eq 'ready')
    } catch { return $false }
}

function Set-DisplayIni($path, $values) {
    $content=if (Test-Path -LiteralPath $path) { [IO.File]::ReadAllText($path) } else { '' }
    foreach ($key in $values.Keys) {
        $pattern='(?m)^'+[regex]::Escape($key)+'\s*=.*$'
        $line=$key+' = '+$values[$key]
        if ($content -match $pattern) { $content=[regex]::Replace($content,$pattern,$line) }
        else { $content=$content.TrimEnd()+[Environment]::NewLine+$line+[Environment]::NewLine }
    }
    [IO.File]::WriteAllText($path,$content,[Text.UTF8Encoding]::new($false))
}

$serial=Find-ProjectEmulator
$runningProjectEmulator=@(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('emulator.exe','qemu-system-x86_64.exe') -and
    $_.CommandLine -match ('-avd\s+"?'+[regex]::Escape($Avd)+'(?:"|\s|$)')
})
if (-not $serial -and -not $runningProjectEmulator.Count) {
    $avdDir=Join-Path $Sdk ('avd/'+$Avd+'.avd')
    if (Test-Path -LiteralPath (Join-Path $avdDir 'config.ini')) {
        Set-DisplayIni (Join-Path $avdDir 'config.ini') @{
            'hw.lcd.width'=1080;'hw.lcd.height'=1920;'hw.lcd.density'=420
            'hw.display1.width'=0;'hw.display1.height'=0;'hw.display1.density'=0;'hw.display1.flag'=0
            'hw.display2.width'=0;'hw.display2.height'=0;'hw.display2.density'=0;'hw.display2.flag'=0
            'hw.display3.width'=0;'hw.display3.height'=0;'hw.display3.density'=0;'hw.display3.flag'=0
        }
        Set-DisplayIni (Join-Path $avdDir 'emulator-user.ini') @{'window.x'=80;'window.y'=60;'window.scale'='0.55'}
    }
    Write-Host '正在打开安卓模拟器，请稍候。'
    $emulatorLogDir=Join-Path $dataDir 'emulator-launch'
    New-Item -ItemType Directory -Path $emulatorLogDir -Force | Out-Null
    $emulatorStamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    # This is the interactive emulator window requested by the user.
    # Cold boot preserves user data and avoids a stalled Quick Boot restore.
    Start-Process -FilePath $emulator -ArgumentList @('-avd',$Avd,'-no-snapshot-load','-no-snapshot-save','-no-boot-anim') -WindowStyle Normal -RedirectStandardOutput (Join-Path $emulatorLogDir ($emulatorStamp+'-emulator-stdout.log')) -RedirectStandardError (Join-Path $emulatorLogDir ($emulatorStamp+'-emulator-stderr.log')) | Out-Null
} elseif (-not $serial) {
    Write-Host '模拟器进程已在启动，等待安卓就绪，不重复打开。'
}
$bootDeadline=(Get-Date).AddSeconds(180)
do {
    if (-not $serial) { $serial=Find-ProjectEmulator }
    if ($serial) {
        $boot=(& $adb -s $serial shell getprop sys.boot_completed 2>$null | Out-String).Trim()
        if ($boot -eq '1') { break }
    }
    Start-Sleep -Seconds 1
} while ((Get-Date) -lt $bootDeadline)
if (-not $serial -or $boot -ne '1') { throw ('模拟器尚未启动完成。启动日志：'+(Join-Path $dataDir 'emulator-launch')) }

# Use a phone-sized viewport at a readable desktop scale, including older AVDs.
foreach ($extraDisplay in 1..3) {
    & $adb -s $serial emu multidisplay del $extraDisplay *> $null
}
& $adb -s $serial shell wm size 1080x1920 | Out-Null
& $adb -s $serial shell wm density 420 | Out-Null

if ($LocalService -and -not (Service-Ready)) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw '端口已有其他服务或连接口令不匹配，请检查原手机服务窗口。' }
    $logDir=Join-Path $dataDir 'emulator-launch'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $python=Join-Path $projectRoot '.venv/Scripts/python.exe'
    $server=Join-Path $projectRoot 'apps/server/server.py'
    $serverArgs=@('-X','utf8',('"'+$server+'"'),'--port',$Port,'--data-dir',('"'+$dataDir+'"'))
    Start-Process -FilePath $python -ArgumentList $serverArgs -WindowStyle Hidden -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $logDir ($stamp+'-stdout.log')) -RedirectStandardError (Join-Path $logDir ($stamp+'-stderr.log')) | Out-Null
    $serviceDeadline=(Get-Date).AddSeconds(30)
    while (-not (Service-Ready)) {
        if ((Get-Date) -gt $serviceDeadline) { throw '电脑服务没有准备好，日志在本机OPMDMobileDemo/emulator-launch目录。' }
        Start-Sleep -Seconds 1
    }
}

if ($LocalService) {
    & $adb -s $serial reverse ('tcp:'+$Port) ('tcp:'+$Port) | Out-Null
    if ($LASTEXITCODE) { throw '模拟器到电脑的连接建立失败。' }
}
$installed=(& $adb -s $serial shell pm path $appId | Out-String).Trim()
$packageInfo=if ($installed) { (& $adb -s $serial shell dumpsys package $appId | Out-String) } else { '' }
$installedCode=if ($packageInfo -match 'versionCode=(\d+)') { [int]$Matches[1] } else { 0 }
$apk=$apkCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($apk) {
    $apkInfo=Get-ApkInfo -ApkPath $apk -SdkRoot $Sdk
    $needsInstall=(-not $installed -or $installedCode -lt $apkInfo.VersionCode)
    if ($installed -and $installedCode -eq $apkInfo.VersionCode) {
        $installedApk=(($installed -split '\r?\n' | Select-Object -First 1) -replace '^package:','').Trim()
        $deviceHash=(& $adb -s $serial shell sha256sum $installedApk 2>$null | Out-String).Trim()
        $needsInstall=($deviceHash -notmatch '^([0-9a-fA-F]{64})\s' -or $Matches[1].ToLowerInvariant() -ne $apkInfo.SHA256)
    }
    if ($needsInstall) {
        & $adb -s $serial install -r $apkInfo.Path
        if ($LASTEXITCODE) { throw 'APK安装失败，原应用资料未清除。' }
        Write-Host ('模拟器已更新到APK版本'+$apkInfo.VersionName)
    } elseif ($installedCode -gt $apkInfo.VersionCode) {
        Write-Host '模拟器已安装较新版本，继续使用现有应用。'
    }
} elseif (-not $installed) {
    throw '未找到Android安装包，请先运行build-android.ps1。'
}
& $adb -s $serial shell input keyevent KEYCODE_WAKEUP | Out-Null
& $adb -s $serial shell am start -n ($appId+'/.MainActivity') | Out-Null
if ($LASTEXITCODE) { throw 'APP打开失败。' }
Write-Host '安卓APP已打开，默认使用安装包内的公网服务连接。'
if ($LocalService) { Write-Host ('本机开发服务已准备好，如需使用可在“我的 → 连接设置”填写：http://127.0.0.1:'+$Port) }
if ($ShowPairing) {
    Write-Host '连接口令（仅在本机使用）：'
    Write-Host ((Get-Content -LiteralPath $tokenPath -Raw -Encoding UTF8).Trim())
}
