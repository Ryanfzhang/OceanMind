#Requires -RunAsAdministrator
[CmdletBinding()]
param(
    [string]$PythonPath = (Get-Command python -ErrorAction Stop).Source,
    [string]$NodePath = (Get-Command node -ErrorAction Stop).Source,
    [string]$TaskName = "OceanMind",
    [string]$WebHost = "127.0.0.1",
    [string]$PublicUrl = "",
    [string]$NginxPath = "",
    [string]$TlsCert = "",
    [string]$TlsKey = "",
    [ValidateRange(1,65535)][int]$WebPort = 3000,
    [ValidateRange(1,65535)][int]$ApiPort = 8000
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = (Resolve-Path -LiteralPath $PythonPath).Path
$node = (Resolve-Path -LiteralPath $NodePath).Path
$script = Join-Path $PSScriptRoot "server.py"
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Task '$TaskName' exists. Remove it explicitly or choose another TaskName."
}
foreach ($value in @($python, $node, $script, $WebHost, $PublicUrl, $NginxPath, $TlsCert, $TlsKey)) {
    if ($value -match '["\r\n]') { throw "Quotes and newlines are not supported in paths/host." }
}
$checkArgs = @($script, "check", "--node", $node, "--web-host", $WebHost,
               "--web-port", "$WebPort", "--api-port", "$ApiPort")
if ($PublicUrl) { $checkArgs += @("--public-url", $PublicUrl) }
$proxyArgs = @()
foreach ($pair in @(@("--nginx", $NginxPath), @("--tls-cert", $TlsCert), @("--tls-key", $TlsKey))) {
    if ($pair[1]) {
        $resolved = (Resolve-Path -LiteralPath $pair[1]).Path
        $proxyArgs += @($pair[0], $resolved)
    }
}
$checkArgs += $proxyArgs
& $python @checkArgs
if ($LASTEXITCODE -ne 0) { throw "Preflight failed. Fix the environment/build first." }
$credential = Get-Credential -Message "Account with access to OceanMind, Python and data (password, not PIN)"
if ($null -eq $credential) { throw "Installation cancelled." }
$arguments = '"{0}" start --node "{1}" --web-host "{2}" --web-port {3} --api-port {4}' -f $script,$node,$WebHost,$WebPort,$ApiPort
if ($PublicUrl) { $arguments += ' --public-url "{0}"' -f $PublicUrl }
for ($i = 0; $i -lt $proxyArgs.Count; $i += 2) {
    $arguments += ' {0} "{1}"' -f $proxyArgs[$i], $proxyArgs[$i+1]
}
$action = New-ScheduledTaskAction -Execute $python -Argument $arguments -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT45S"
$options = @{
    StartWhenAvailable = $true
    MultipleInstances = "IgnoreNew"
    ExecutionTimeLimit = [TimeSpan]::Zero
    RestartCount = 999
    RestartInterval = (New-TimeSpan -Minutes 1)
    AllowStartIfOnBatteries = $true
    DontStopIfGoingOnBatteries = $true
}
$settings = New-ScheduledTaskSettingsSet @options
# Password goes to Windows Task Scheduler, never to a script/config file.
$plain = $credential.GetNetworkCredential().Password
try {
    $registration = @{
        TaskName = $TaskName
        Action = $action
        Trigger = $trigger
        Settings = $settings
        User = $credential.UserName
        Password = $plain
        RunLevel = "Limited"
        Description = "OceanMind backend and frontend at boot"
    }
    Register-ScheduledTask @registration | Out-Null
} finally {
    $plain = $null
    $credential = $null
    $registration = $null
}
Write-Host "Installed '$TaskName'. Runs at boot without an interactive login."
Write-Host "Start now: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Logs: $root\logs\server"
Write-Host "Do not also run start-server.bat while the task is running."
