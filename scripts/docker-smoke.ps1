#Requires -Version 5.1
<#
.SYNOPSIS
Runs synthetic Docker acceptance checks using separate, offline containers.
.EXAMPLE
powershell.exe -NoProfile -File .\scripts\docker-smoke.ps1 -Root "C:\Temp\prak acceptance 01"
.NOTES
Build both images before invoking this script. Root must be a NEW absolute
Windows filesystem directory with an existing parent. Spaces are supported;
the workspace and fixture subdirectories always contain spaces as well.
Results are retained on success or failure. No repository datasets are mounted.
#>
[CmdletBinding()]
param(
    [string]$Image = 'prak:local',
    [string]$TestImage = 'prak-tests:local',
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Root
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($env:OS -ne 'Windows_NT') {
    throw 'Run this script from Windows PowerShell with Docker Desktop using Linux containers.'
}
if (($Root -notmatch '^[A-Za-z]:[\\/]') -and ($Root -notmatch '^\\\\[^\\]+\\[^\\]+\\')) {
    throw 'Root must be an absolute Windows path, for example C:\Temp\prak acceptance 01.'
}
$Root = [System.IO.Path]::GetFullPath($Root).TrimEnd([char[]]'\/')
if (Test-Path -LiteralPath $Root) {
    throw "Root already exists; choose a new directory: $Root"
}
$Parent = Split-Path -Path $Root -Parent
if (-not (Test-Path -LiteralPath $Parent -PathType Container)) {
    throw "Root parent must already exist: $Parent"
}
$null = Get-Command docker -CommandType Application -ErrorAction Stop

function Invoke-Docker {
    param([Parameter(Mandatory = $true)][string[]]$DockerArgs)
    # Do not pipe or redirect stderr through PowerShell 5.1's ErrorRecord stream.
    # Native failures are not terminating PowerShell errors: check explicitly.
    & docker @DockerArgs
    $Code = $LASTEXITCODE
    if ($Code -ne 0) {
        throw "Docker failed with exit code $Code. Results (if created): $Root"
    }
}

# Validate the local images before creating the root; never pull implicitly.
Invoke-Docker -DockerArgs @('image', 'inspect', '--format', '{{.Id}}', $Image)
Invoke-Docker -DockerArgs @('image', 'inspect', '--format', '{{.Id}}', $TestImage)
$null = New-Item -ItemType Directory -Path $Root -ErrorAction Stop
$Workspace = Join-Path $Root 'workspace with spaces'
$Raw = Join-Path $Root 'raw fixture'

function Invoke-Helper {
    param([Parameter(Mandatory = $true)][string[]]$HelperArgs)
    $DockerArgs = @('run', '--rm', '--pull', 'never', '--network', 'none',
        '--volume', "${Root}:/acceptance")
    if (Test-Path -LiteralPath $Workspace -PathType Container) {
        $DockerArgs += @('--volume', "${Workspace}:/workspace")
    }
    $DockerArgs += @('--entrypoint', 'python', $TestImage, '/build/tests/container_support.py')
    Invoke-Docker -DockerArgs ($DockerArgs + $HelperArgs)
}

function Invoke-Runtime {
    param([Parameter(Mandatory = $true)][string[]]$CommandArgs)
    $DockerArgs = @('run', '--rm', '--pull', 'never', '--network', 'none',
        '--volume', "${Workspace}:/workspace", '--volume', "${Raw}:/opt/prak/olist:ro", $Image)
    Invoke-Docker -DockerArgs ($DockerArgs + $CommandArgs)
}

$InitArgs = @('init', '--batch-size', '12', '--min-category-count', '1', '--model', 'svd',
    '--initial-users', '6', '--additional-users', '3', '--split-sizes', '4', '2', '2',
    '--svd-n-components', '2', '--svd-n-iter', '3', '--k', '2',
    '--temporal-n-clusters', '3', '--max-evaluation-rows', '8', '--random-state', '17',
    '--temperature', '3600', '--price-threshold', '0.23',
    '--category-threshold', '0.47', '--state-threshold', '0.89')

Write-Host 'Creating the synthetic nine-table bundle and preservation sentinels.'
Invoke-Helper -HelperArgs @('create')

Write-Host 'Initializing and training the first small SVD batch.'
Invoke-Runtime -CommandArgs ($InitArgs + @('--verbose'))
Invoke-Helper -HelperArgs @('check', '--phase', 'init')
Invoke-Helper -HelperArgs @('snapshot', '--name', 'first')

Write-Host 'Resuming in a new container and preserving the first model and histories.'
Invoke-Runtime -CommandArgs @('update', '--verbose')
Invoke-Helper -HelperArgs @('check', '--phase', 'update')
Invoke-Helper -HelperArgs @('snapshot', '--name', 'second')

Write-Host 'Inferring from explicit step_001 with an automatically named output CSV.'
Invoke-Runtime -CommandArgs @('inference', '--model-dir', 'run/steps/step_001/ranking',
    '--user-id', 'user_000001_000000', '--k', '2', '--verbose')
Invoke-Helper -HelperArgs @('check', '--phase', 'inference')

Write-Host 'Removing only summary outputs, then rebuilding them on exhausted update.'
Invoke-Helper -HelperArgs @('snapshot', '--name', 'exhausted')
Invoke-Runtime -CommandArgs @('update', '--verbose')
Invoke-Helper -HelperArgs @('check', '--phase', 'exhausted')

Write-Host 'Repeating init; resetting outputs while preserving every prior log byte.'
Invoke-Helper -HelperArgs @('snapshot', '--name', 'reinit')
Invoke-Runtime -CommandArgs $InitArgs
Invoke-Helper -HelperArgs @('check', '--phase', 'reinit')

Write-Host "PASS: all synthetic Docker acceptance checks. Results: $Root"
Write-Host "Check receipts: $(Join-Path $Root 'checks')"
Write-Host "Runtime logs and final reports: $Workspace"
