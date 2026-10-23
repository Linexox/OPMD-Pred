function Get-ApkInfo {
    param([string]$ApkPath, [string]$SdkRoot)
    $resolvedApk = (Resolve-Path -LiteralPath $ApkPath -ErrorAction Stop).Path
    $tools = Get-ChildItem -LiteralPath (Join-Path $SdkRoot 'build-tools') -Directory |
        Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'aapt.exe')) } |
        Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    if (-not $tools) { throw '未找到读取APK版本所需的Android build-tools。' }
    $aapt = Join-Path $tools.FullName 'aapt.exe'
    # Use the filename from its own directory so aapt handles the Chinese workspace path.
    Push-Location -LiteralPath (Split-Path $resolvedApk -Parent)
    try {
        $badging = @(& $aapt dump badging (Split-Path $resolvedApk -Leaf) 2>$null)
        $toolExit = $LASTEXITCODE
    } finally { Pop-Location }
    if ($toolExit) { throw ('APK版本读取失败：' + $resolvedApk) }
    $packageLine = $badging | Where-Object { $_ -like 'package:*' } | Select-Object -First 1
    if ($packageLine -notmatch "versionCode='(\d+)'") { throw 'APK没有有效版本号。' }
    $code = [int]$Matches[1]
    if ($packageLine -notmatch "versionName='([^']+)'") { throw 'APK没有版本名称。' }
    [pscustomobject]@{
        Path = $resolvedApk
        VersionCode = $code
        VersionName = $Matches[1]
        SHA256 = (Get-FileHash -LiteralPath $resolvedApk -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
