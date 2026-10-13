param([string]$PythonPath = '')
if (-not $PythonPath) { $PythonPath=Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) '.venv/Scripts/python.exe' }
$ErrorActionPreference = 'Stop'
$taskScript = Join-Path $PSScriptRoot 'app.py'
if (-not (Test-Path -LiteralPath $PythonPath)) { throw '请用PythonPath指定已安装requirements.txt的Python解释器。' }
& $PythonPath -X utf8 $taskScript
