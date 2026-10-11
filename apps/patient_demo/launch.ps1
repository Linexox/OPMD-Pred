param([string]$PythonPath = 'D:/.Workspace/医工/OPMD-Pred/.venv/Scripts/python.exe')
$ErrorActionPreference = 'Stop'
$taskScript = Join-Path $PSScriptRoot 'app.py'
if (-not (Test-Path -LiteralPath $PythonPath)) { throw '请用PythonPath指定已安装requirements.txt的Python解释器。' }
& $PythonPath -X utf8 $taskScript
