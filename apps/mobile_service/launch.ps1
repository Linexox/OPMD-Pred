param([int]$Port=8765,[string]$DataDir="$env:LOCALAPPDATA\OPMDMobileDemo")
$ErrorActionPreference='Stop'
$appPython='D:\.Workspace\医工\OPMD-Pred\.venv\Scripts\python.exe'
$wifiAddresses = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' }
Write-Host '手机与电脑连接同一 Wi-Fi。可用服务地址：'
$wifiAddresses | ForEach-Object { Write-Host ($_.InterfaceAlias+' : http://'+$_.IPAddress+':'+$Port) }
$tokenPath=Join-Path $DataDir 'pairing-token.txt'
if (Test-Path -LiteralPath $tokenPath) {
  try {
    $pairingToken=(Get-Content -LiteralPath $tokenPath -Raw).Trim()
    $health=Invoke-RestMethod -Uri ('http://127.0.0.1:'+$Port+'/api/health') -Headers @{Authorization=('Bearer '+$pairingToken)} -TimeoutSec 2
    if ($health.status -eq 'ready') { Write-Host '服务已在运行。连接口令：'; Write-Host $pairingToken; return }
  } catch { }
}
& $appPython -X utf8 (Join-Path $PSScriptRoot 'server.py') --port $Port --data-dir $DataDir
