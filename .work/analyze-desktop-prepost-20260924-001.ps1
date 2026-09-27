param([Parameter(Mandatory=$true)][string]$EvidenceRoot)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = (Resolve-Path -LiteralPath $EvidenceRoot).Path
$output = 'reports/training/m2-smolvla-prepost-analysis-verification-2026-09-24.json'
if (Test-Path -LiteralPath $output) { throw 'Output already exists; use a new identity.' }
function ReadJson([string]$Path) { Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json }
function Hash([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function EqualJson($Left, $Right) {
    ($Left | ConvertTo-Json -Depth 100 -Compress) -ceq ($Right | ConvertTo-Json -Depth 100 -Compress)
}
function Require([bool]$Condition, [string]$Message) { if (!$Condition) { throw $Message } }
$manifest = ReadJson (Join-Path $root 'manifest.json')
$fileChecks = @()
foreach ($entry in $manifest.files) {
    $path = Join-Path $root $entry.destination
    Require (Test-Path -LiteralPath $path -PathType Leaf) ('Missing copy: ' + $entry.destination)
    $actual = Hash $path
    Require ((Get-Item -LiteralPath $path).Length -eq $entry.bytes) ('Size mismatch: ' + $entry.destination)
    Require ($actual -ceq $entry.sha256) ('Copy SHA mismatch: ' + $entry.destination)
    Require ((Hash $entry.source) -ceq $entry.sha256) ('Source SHA mismatch: ' + $entry.source)
    $fileChecks += [ordered]@{path=$entry.destination; source=$entry.source; bytes=$entry.bytes; sha256=$actual}
}
$native = Join-Path $root 'native-gate'
$exp = 'm2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003'
$gates = Join-Path $native "results/$exp/gates"
$base = ReadJson (Join-Path $gates 'gate3-smolvla-sim-481.json')
$trained = ReadJson (Join-Path $gates 'gate3-smolvla-sim-482.json')
$gate4 = ReadJson (Join-Path $gates 'gate4-smolvla-sim-482.json')
$gateReports = @(
    @{arm='base'; file='gate3-smolvla-sim-481.json'; report=$base},
    @{arm='trained'; file='gate3-smolvla-sim-482.json'; report=$trained},
    @{arm='trained'; file='gate4-smolvla-sim-482.json'; report=$gate4}
)
$bindings = @()
foreach ($item in $gateReports) {
    $r = $item.report
    $artifactRelative = 'artifacts/' + $exp + '/' + $r.artifact_id + '/manifest.json'
    Require ((Hash (Join-Path $native $artifactRelative)) -ceq $r.artifact_manifest_sha256) 'Artifact binding mismatch'
    Require ((Hash (Join-Path $native ($item.arm + '-gate.yaml'))) -ceq $r.simulation_plan_sha256) 'Gate plan binding mismatch'
    Require (!$r.hidden_test_loaded) 'Unexpected hidden-test claim'
    $bindings += [ordered]@{report=$item.file;sha256=(Hash (Join-Path $gates $item.file));plan_sha256=$r.simulation_plan_sha256;artifact_manifest_sha256=$r.artifact_manifest_sha256}
}
Require ((Hash (Join-Path $gates 'gate3-smolvla-sim-482.json')) -ceq $gate4.gate3_report_sha256) 'Gate 3 reference mismatch'
Require (EqualJson $trained.code_identity $gate4.code_identity) 'Paired Gate code mismatch'
Require (!(Test-Path -LiteralPath (Join-Path $gates 'gate4-smolvla-sim-481.json'))) 'Unexpected base Gate 4'
$episodes = @()
$contactPairs = @{}
$episodeFiles = @(Get-ChildItem -LiteralPath (Join-Path $gates 'gate4-smolvla-sim-482-episodes') -Filter '*.json' | Sort-Object Name)
Require ($episodeFiles.Count -eq 5) 'Expected five episode reports'
foreach ($file in $episodeFiles) {
    $r = ReadJson $file.FullName
    $index = $episodes.Count
    Require ($r.seed -eq 1000 + $index -and $r.episode_index -eq $index) 'Episode order mismatch'
    Require (EqualJson $r.metrics $gate4.episodes[$index]) 'Episode metrics differ from aggregate report'
    foreach ($key in @('artifact_id','artifact_manifest_sha256','simulation_plan_sha256','gate3_report_sha256')) {
        Require ($r.$key -ceq $gate4.$key) ('Episode binding mismatch: ' + $key)
    }
    Require (EqualJson $r.code_identity $gate4.code_identity) 'Episode code identity mismatch'
    $m = $r.metrics
    $pairSum = 0
    foreach ($pair in $m.unexpected_collision_pairs.PSObject.Properties) {
        $pairSum += [int]$pair.Value
        if (!$contactPairs.ContainsKey($pair.Name)) { $contactPairs[$pair.Name] = 0 }
        $contactPairs[$pair.Name] += [int]$pair.Value
    }
    Require ($pairSum -eq $m.unexpected_collisions) 'Contact histogram mismatch'
    $episodes += [ordered]@{seed=$r.seed;steps=$m.rollout_length;success=$m.success;maximum_reward=$m.maximum_reward;joint_limit_count=$m.joint_limit_violations;unexpected_contact_count=$m.unexpected_collisions;sha256=(Hash $file.FullName)}
}
$steps = ($episodes | ForEach-Object { $_.steps } | Measure-Object -Sum).Sum
$recomputed = [ordered]@{
    task_success_rate=(@($episodes | Where-Object { $_.success }).Count / 5.0)
    mean_rollout_length=($steps / 5.0)
    joint_limit_violations=($episodes | ForEach-Object { $_.joint_limit_count } | Measure-Object -Sum).Sum
    unexpected_collisions=($episodes | ForEach-Object { $_.unexpected_contact_count } | Measure-Object -Sum).Sum
}
foreach ($key in @('invalid_action_rate','raw_limit_violation_rate','unprojected_limit_violation_rate','policy_output_limit_violation_rate','executed_limit_violation_rate')) {
    $sum = 0.0
    foreach ($m in $gate4.episodes) { $sum += $m.$key * $m.rollout_length }
    $recomputed[$key] = $sum / $steps
}
foreach ($key in @('action_smoothness_l2','policy_inference_seconds','simulation_step_seconds')) {
    $sourceKey = if ($key -eq 'action_smoothness_l2') { $key } else { 'mean_' + $key }
    $sum = 0.0
    foreach ($m in $gate4.episodes) { $sum += $m.$sourceKey }
    $recomputed['mean_' + $key] = $sum / 5.0
}
foreach ($key in $recomputed.Keys) {
    Require ([Math]::Abs([double]$recomputed[$key] - [double]$gate4.aggregate.$key) -le 1e-12) ('Aggregate mismatch: ' + $key)
}
$criteria = [ordered]@{
    completed_nonempty_rollouts=($steps -gt 0)
    executed_actions_within_contract=($recomputed.executed_limit_violation_rate -eq 0)
    finite_actions=($recomputed.invalid_action_rate -eq 0)
    joint_limits_respected=($recomputed.joint_limit_violations -eq 0)
    maximum_unexpected_collisions=($recomputed.unexpected_collisions -le 0)
    minimum_task_success_rate=($recomputed.task_success_rate -ge 0.2)
    policy_output_projection_registered=($gate4.policy_output_projection -ceq 'action_contract_clip')
    projected_policy_actions_within_contract=($recomputed.policy_output_limit_violation_rate -eq 0)
}
foreach ($key in $criteria.Keys) { Require ($criteria[$key] -eq $gate4.acceptance_criteria.$key) ('Criterion mismatch: ' + $key) }
$reload = @()
foreach ($arm in @('base','trained')) {
    $one = Join-Path $native "$arm-probe-1.json"
    $two = Join-Path $native "$arm-probe-2.json"
    Require ((Hash $one) -ceq (Hash $two)) 'Reload JSON byte mismatch'
    $p = ReadJson $one
    Require ($p.raw.Count -eq 50 -and $p.projected.Count -eq 50) 'Reload chunk shape mismatch'
    foreach ($row in @($p.raw) + @($p.projected)) { Require ($row.Count -eq 14) 'Reload action dimension mismatch' }
    $reload += [ordered]@{arm=$arm;probe_sha256=(Hash $one);byte_equal=$true;shape=@(50,14);scope='one reset observation, zero noise, raw and projected action arrays; runtime execution not repeated'}
}
$historyChecks = @()
foreach ($dir in (Get-ChildItem -LiteralPath (Join-Path $root 'basin-history/history') -Directory | Sort-Object Name)) {
    $hm = ReadJson (Join-Path $dir.FullName 'manifest.json')
    foreach ($entry in $hm.files.PSObject.Properties) { Require ((Hash (Join-Path $dir.FullName $entry.Name)) -ceq $entry.Value) 'Basin manifest mismatch' }
    $n = ReadJson (Join-Path $dir.FullName 'artifacts/native.json')
    $source = Join-Path $native $n.parameters.source_report_path
    Require ((Hash $source) -ceq $n.parameters.source_report_sha256) 'Basin source SHA mismatch'
    Require (EqualJson (ReadJson $source) $n.parameters.source_report) 'Basin embedded source mismatch'
    $eventCount = @(Get-Content -LiteralPath (Join-Path $dir.FullName 'events.jsonl') | Where-Object { $_.Trim() }).Count
    Require ($eventCount -eq $n.events.Count) 'Basin event count mismatch'
    $historyChecks += [ordered]@{run=$dir.Name;status=$n.status;event_count=$eventCount;verified_files=@($hm.files.PSObject.Properties).Count;source_sha256=$n.parameters.source_report_sha256}
}
$pairChecks = @()
$summary = ReadJson (Join-Path $root 'parameter-evidence/result.json')
$nameSet = $null
foreach ($pair in @('base-to-2500','base-to-5000','2500-to-5000')) {
    $path = Join-Path $root "parameter-evidence/$pair.jsonl"
    $rows = @(Get-Content -LiteralPath $path | ForEach-Object { $_ | ConvertFrom-Json })
    $s = $summary.pairs.$pair
    Require ((Hash $path) -ceq $s.rows_sha256) 'Parameter row seal mismatch'
    Require ($rows.Count -eq 500 -and @($rows.name | Sort-Object -Unique).Count -eq 500) 'Parameter name coverage mismatch'
    $names = ($rows.name | Sort-Object) -join "`n"
    if ($null -ne $nameSet) { Require ($names -ceq $nameSet) 'Cross-pair names differ' }
    $nameSet = $names
    Require (@($rows | Where-Object { !$_.finite }).Count -eq 0) 'Nonfinite parameter record'
    $numel = ($rows | Measure-Object numel -Sum).Sum
    $same = @($rows | Where-Object numeric_equal).Count
    $byte = @($rows | Where-Object byte_equal).Count
    Require ($numel -eq $s.numel -and $same -eq $s.numeric_equal -and $byte -eq $s.byte_equal) 'Parameter aggregate mismatch'
    $norms = @($rows | Where-Object { $_.name -match 'lm_expert\..*norm.*weight$' })
    Require ($norms.Count -eq 33) 'Norm coverage mismatch'
    foreach ($norm in $norms) {
        Require ($norm.numeric_equal -and $norm.left.mean -eq 1 -and $norm.right.mean -eq 1 -and $norm.left.std_population -eq 0 -and $norm.right.std_population -eq 0) 'Norm summary mismatch'
    }
    $pairChecks += [ordered]@{pair=$pair;rows=$rows.Count;numel=$numel;numeric_changed=500-$same;numeric_equal=$same;byte_equal=$byte;dtype_changed=@($rows|Where-Object {$_.dtype[0] -cne $_.dtype[1]}).Count;unchanged_all_one_norms=$norms.Count;rows_sha256=(Hash $path)}
}
$sourceHashes = [ordered]@{}
foreach ($p in @('scripts/run_prepost_gate_comparison.py','scripts/smolvla_sim_gate.py','src/rosetta_reality/sim/gym_aloha.py')) { $sourceHashes[$p] = Hash $p }
$result = [ordered]@{
    id='prepost-desktop-analysis-20260924-001'
    checked_at_utc=[DateTime]::UtcNow.ToString('o')
    status='passed'
    scope='read-only saved-evidence identity and JSON arithmetic; no model/data/optimizer/simulation execution'
    package_name=(Split-Path -Leaf $root)
    package_manifest_sha256=(Hash (Join-Path $root 'manifest.json'))
    manifest_file_count=$fileChecks.Count
    manifest_bytes=($manifest.files|Measure-Object bytes -Sum).Sum
    file_checks=$fileChecks
    gate_bindings=$bindings
    gate3=@{base=$base.metrics;trained=$trained.metrics}
    gate4_episode_recalculation=$episodes
    gate4_aggregate=$recomputed
    gate4_criteria=$criteria
    contact_pair_histogram=$contactPairs
    independent_reload_saved_evidence=$reload
    basin_import_checks=$historyChecks
    parameter_record_checks=$pairChecks
    reviewed_current_source_sha256=$sourceHashes
    verifier_sha256=(Hash $PSCommandPath)
    limits=@('No independent reconstruction of physical contacts or joint states: current package lacks per-step arrays.','Parameter statistics reconciled from saved JSONL only; historical raw-weight numerical verification was not rerun.','Artifact manifests bound; linked model and processor payloads are excluded from this desktop package.','No fresh platform power, remote runtime, or actual gradient observation.','Basin records are derived imports, not independent experimental replications.')
}
$json = $result | ConvertTo-Json -Depth 100
$stream = [IO.File]::Open((Join-Path (Get-Location).Path $output), [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try { $bytes = [Text.UTF8Encoding]::new($false).GetBytes($json + "`n"); $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
$result | Select-Object id,status,manifest_file_count,manifest_bytes,gate4_episode_recalculation,gate4_aggregate,contact_pair_histogram,basin_import_checks,parameter_record_checks | ConvertTo-Json -Depth 10
