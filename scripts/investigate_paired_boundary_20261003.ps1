param([string]$Output = 'reports/training/paired-boundary-independent-check-2026-10-03-002.json')
# File-only investigation: no model, dataset decoding, simulator, network or Python.
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
function Read-Json([string]$relative) {
    Get-Content -LiteralPath (Join-Path $repo $relative) -Encoding UTF8 -Raw | ConvertFrom-Json
}
function Hash-File([string]$relative) {
    (Get-FileHash -LiteralPath (Join-Path $repo $relative) -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Require($condition, [string]$message) {
    if (-not $condition) { throw $message }
}
$destination = [IO.Path]::GetFullPath((Join-Path $repo $Output))
Require ($destination.StartsWith($repo + [IO.Path]::DirectorySeparatorChar) -and $Output.StartsWith('reports/training/')) 'Output must be a repository training report'
Require (-not (Test-Path -LiteralPath $destination)) 'Output exists; retain previous evidence'
$root = '.work/paired-ABC-measurement-20260928-003-received-20260930-001'
$files = "$root/files"
$inventory = @{}
foreach ($line in Get-Content -LiteralPath (Join-Path $repo "$root/remote.sha256") -Encoding UTF8) {
    Require ($line -match '^([0-9a-f]{64})  \./(.+)$') 'Malformed SHA manifest'
    $expected, $relative = $Matches[1], $Matches[2]
    Require (-not [IO.Path]::IsPathRooted($relative) -and $relative -notmatch '(^|/)\.\.(/|$)') 'Manifest path escape'
    Require (-not $inventory.ContainsKey($relative)) 'Duplicate manifest path'
    Require ((Hash-File "$files/$relative") -eq $expected) "Measurement SHA drift: $relative"
    $inventory[$relative] = $expected
}
$actual = @(Get-ChildItem -LiteralPath (Join-Path $repo $files) -File -Recurse)
Require ($actual.Count -eq $inventory.Count) 'Manifest file count differs'
foreach ($file in $actual) {
    $relative = $file.FullName.Substring((Join-Path $repo $files).Length + 1).Replace('\','/')
    Require ($inventory.ContainsKey($relative)) "Unlisted file: $relative"
}
$bank = '.work/paired-bank-v3-20260927-001-independent-received/extracted/collection'
$sourceManifest = Read-Json "$bank/manifest.json"
$selectedSources = @{}
$pair = 'bank-003b-scene-2445'
foreach ($entry in $sourceManifest.files) {
    if (($entry.path -like "$pair/R/*" -and $entry.path -notmatch 'frame-') -or $entry.path -like "$pair/branch-*") {
        Require ((Hash-File "$bank/$($entry.path)") -eq $entry.sha256) "Bank source SHA drift: $($entry.path)"
        $selectedSources[$entry.path] = $entry.sha256
    }
}
$gzipPath=Join-Path $repo "$bank/$pair/branch-restore.json.gz"
$gzipFile=[IO.File]::OpenRead($gzipPath)
$gzip=[IO.Compression.GZipStream]::new($gzipFile,[IO.Compression.CompressionMode]::Decompress)
$reader=[IO.StreamReader]::new($gzip)
try { $parity=$reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose(); $gzip.Dispose(); $gzipFile.Dispose() }
Require ($parity.passed -eq $true -and $parity.engineering_steps -eq 2 -and $parity.exact_replays.Count -eq 2 -and $parity.exact_replays[0] -eq $parity.exact_replays[1]) 'Saved source branch hold-command parity failed'
$sourcePath = "$bank/$pair/R/trace.jsonl"
$source = @(Get-Content -LiteralPath (Join-Path $repo $sourcePath) -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json })
$deviation = @($source | Where-Object action_source -eq 'registered_deliberate_deviation')
Require ($deviation.Count -eq 1) 'Expected one source deviation'
$next = @($source | Where-Object step -eq ($deviation[0].step + 1))
Require ($next.Count -eq 1 -and $next[0].action_source -eq 'state_conditioned_teacher') 'Source correction missing'
$teacher = $next[0]
Require (($source | Where-Object { -not $_.safe -or $_.after.joint_limits -or $_.after.unexpected_collisions }).Count -eq 0) 'Source teacher contains safety failures'
Require ($source[-1].after.info.is_success -eq $true) 'Teacher terminal success missing'
$arms = @()
$sourceHashes = @{}
$processorHashes = @{}
foreach ($arm in 'A','B','C') {
    $plan = Read-Json "$files/evaluation-plan-$arm.json"
    Require ((Hash-File "$bank/manifest.json") -eq $plan.bank_manifest_sha256) 'Bank identity mismatch'
    Require ((Hash-File '.work/paired-bank-v3-20260927-001-independent-received/extracted/collection-audit.json') -eq $plan.bank_audit_sha256) 'Bank audit mismatch'
    foreach ($property in $plan.source_sha256.PSObject.Properties) {
        Require ((Hash-File $property.Name) -eq $property.Value) "Runner source drift: $($property.Name)"
        $sourceHashes[$property.Name] = $property.Value
    }
    $artifact=Read-Json "$files/artifacts/$arm/manifest.json"
    $checkpoint='.work/paired-native-ABC-formal-20260927-001-received/paired-native-ABC-formal-20260927-001/checkpoints/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/train/paired-mixture-'+$arm.ToLowerInvariant()+'-train-20260927-008/checkpoints/005000'
    foreach ($property in $artifact.files.PSObject.Properties) {
        if ($property.Name -match 'processor' -or $property.Name -eq 'pretrained_model/config.json') {
            Require ((Hash-File "$checkpoint/$($property.Name)") -eq $property.Value) "Saved processor/config drift: $arm/$($property.Name)"
            $processorHashes["$arm/$($property.Name)"]=$property.Value
        }
    }
    $relative = "$files/measurement-$arm/recovery-bank/$pair.jsonl"
    $rows = @(Get-Content -LiteralPath (Join-Path $repo $relative) -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json })
    Require ($rows.Count -eq 1 -and $rows[0].step -eq 1) '2445 endpoint did not stop after exactly one recorded physical step'
    $row = $rows[0]
    Require ($row.joint_limit_violations -gt 0 -or $row.unexpected_contacts.Count -gt 0) 'No recorded stop trigger'
    Require ($row.raw_clips -eq 0) 'Bank first command required clipping'
    $dimensions = @()
    for ($i=0; $i -lt 14; $i++) {
        $dimensions += [ordered]@{
            index=$i; source_state=[double]$teacher.state[$i]; teacher_command=[double]$teacher.action[$i]
            student_command=[double]$row.executed_action[$i]; student_after=[double]$row.state_after[$i]
            student_minus_source=[double]$row.executed_action[$i]-[double]$teacher.state[$i]
            student_minus_teacher=[double]$row.executed_action[$i]-[double]$teacher.action[$i]
            teacher_minus_source=[double]$teacher.action[$i]-[double]$teacher.state[$i]
        }
    }
    $arms += [ordered]@{ arm=$arm; trace=$relative; line=1; recorded_steps=$rows.Count
        violations=$row.joint_limit_violations; contacts=$row.unexpected_contacts; raw_clips=$row.raw_clips
        dimensions=$dimensions; actual_initial_state_recorded=$false; same_runtime_teacher_student_parity_proven=$false }
}
$result = [ordered]@{
    schema='rosetta.file_boundary_investigation.v1'; date='2026-10-03'
    branch=(git -C $repo branch --show-current); head=(git -C $repo rev-parse HEAD)
    measurement_manifest_sha256=(Hash-File "$root/remote.sha256")
    measurement_files_verified=$inventory.Count; runner_sources_verified=$sourceHashes.Count; runner_source_sha256=$sourceHashes
    selected_bank_files_verified=$selectedSources.Count; selected_bank_source_sha256=$selectedSources
    saved_processor_config_files_verified=$processorHashes.Count; saved_processor_config_sha256=$processorHashes
    source_branch_hold_command_parity=[ordered]@{passed=$parity.passed; engineering_steps=$parity.engineering_steps; exact_replays=$parity.exact_replays; policy_inference_included=$false}
    teacher_trace=$sourcePath; teacher_rows=$source.Count; deviation_step=$deviation[0].step
    source_correction_step=$teacher.step; source_correction_line=$teacher.step+1
    source_correction_grasps=[ordered]@{ socket=$teacher.before.socket_grasp_contact; peg=$teacher.before.peg_grasp_contact }
    teacher_terminal_step=$source[-1].step; teacher_safe=$true; arms=$arms
    scope='hashes and recorded JSON arithmetic only; no ML imports or physics execution'
    causal_attribution_proven=$false; old_python_unit_tests_rerun=$false
}
$json = ($result | ConvertTo-Json -Depth 25) + [Environment]::NewLine
$stream = [IO.File]::Open($destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try { $bytes=[Text.UTF8Encoding]::new($false).GetBytes($json); $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
Write-Output "PASS: $($inventory.Count) measurement hashes; $($sourceHashes.Count) runner source hashes; $($selectedSources.Count) selected bank files; 3 post-step boundary records; output=$Output"
