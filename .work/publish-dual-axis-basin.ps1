$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskBasin = [IO.Path]::GetFullPath((Join-Path $taskRepo '../basin'))
$taskStage = Join-Path $taskRepo '.work/dual-axis-basin-stage'
$taskManifest = Get-Content -LiteralPath (Join-Path $taskRepo '.work/dual-axis-basin-publish.json') -Raw | ConvertFrom-Json
$taskBranch = git -C $taskBasin branch --show-current
if ($LASTEXITCODE -ne 0 -or $taskBranch -ne 'codex/basin-evidence-demo') { throw 'Unexpected Basin branch; refusing writes' }
foreach ($taskFile in $taskManifest.files) {
    $taskTarget = [IO.Path]::GetFullPath((Join-Path $taskBasin $taskFile.path))
    $taskSource = [IO.Path]::GetFullPath((Join-Path $taskStage $taskFile.path))
    if (-not $taskTarget.StartsWith($taskBasin + [IO.Path]::DirectorySeparatorChar) -or -not $taskSource.StartsWith($taskStage + [IO.Path]::DirectorySeparatorChar)) { throw 'Path outside intended roots' }
    if ((Get-FileHash -LiteralPath $taskSource).Hash.ToLowerInvariant() -ne $taskFile.prepared_sha256) { throw 'Prepared source changed' }
    if ($null -eq $taskFile.expected_sha256) {
        if (Test-Path -LiteralPath $taskTarget) { throw 'New destination already exists' }
    } elseif ((Get-FileHash -LiteralPath $taskTarget).Hash.ToLowerInvariant() -ne $taskFile.expected_sha256) { throw 'Basin destination changed since review' }
}
$taskBackup = Join-Path $taskBasin '.work/dual-axis-before-20260924-001'
if (Test-Path -LiteralPath $taskBackup) { throw 'Backup already exists; preserve it' }
New-Item -ItemType Directory -Path $taskBackup | Out-Null
foreach ($taskFile in $taskManifest.files) {
    $taskTarget = Join-Path $taskBasin $taskFile.path
    if (Test-Path -LiteralPath $taskTarget) {
        $taskSaved = Join-Path $taskBackup $taskFile.path
        New-Item -ItemType Directory -Path (Split-Path -Parent $taskSaved) -Force | Out-Null
        Copy-Item -LiteralPath $taskTarget -Destination $taskSaved
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $taskTarget) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $taskStage $taskFile.path) -Destination $taskTarget
    if ((Get-FileHash -LiteralPath $taskTarget).Hash.ToLowerInvariant() -ne $taskFile.prepared_sha256) { throw 'Published byte mismatch' }
}
Write-Output 'Published 6 reviewed files to existing Basin feature branch; old files backed up; no Git commit or push.'
