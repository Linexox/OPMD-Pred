param([ValidateRange(1024,65535)][int]$Port=8765,[string]$InterfaceAlias='WLAN')
$ErrorActionPreference='Stop'
if (-not (Get-NetAdapter -Name $InterfaceAlias -ErrorAction SilentlyContinue)) { throw '找不到指定Wi-Fi网卡，请填写InterfaceAlias' }
$ruleName='OPMD-Mobile-Demo-'+$Port
if (-not (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -Name $ruleName -DisplayName 'OPMD手机Demo局域网' -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port -RemoteAddress LocalSubnet -InterfaceAlias $InterfaceAlias -Profile Any | Out-Null
}
Write-Host '已允许当前Wi-Fi同网段访问Demo端口。'
