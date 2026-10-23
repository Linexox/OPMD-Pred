param([int]$Port=8765,[string]$DataDir='')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $DataDir) { $DataDir=Join-Path (Split-Path $projectRoot -Parent) 'OPMDMobileDemo' }
$appPython=Join-Path $projectRoot '.venv/Scripts/python.exe'
$wifiAddresses = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' }
Write-Host '电脑服务地址（公网连接使用已配置的NPS地址）：'
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
