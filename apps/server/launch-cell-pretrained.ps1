param([int]$Port=8765,[string]$DataDir='')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $DataDir) { $DataDir=Join-Path (Split-Path $projectRoot -Parent) '运行数据' }
$appPython=Join-Path $projectRoot '.venv/Scripts/python.exe'
$tokenPath=Join-Path $DataDir 'pairing-token.txt'
if (-not (Test-Path -LiteralPath $tokenPath)) { throw '请先启动手机APP服务生成配对口令' }
& $appPython -X utf8 (Join-Path $PSScriptRoot 'binary_linux_service.py') --port $Port --data-dir $DataDir --token-file (Join-Path $DataDir 'pairing-token.txt') --cell-pretrained
