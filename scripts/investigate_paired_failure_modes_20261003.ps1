param([string]$Output='reports/training/paired-failure-modes-2026-10-03-002.json',[switch]$HelpersOnly)
# Saved JSON arithmetic only. No model, dataset decoder, physics or network calls.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
function Require($condition,[string]$message) { if (-not $condition) { throw $message } }
function Hash([string]$relative) { (Get-FileHash -LiteralPath (Join-Path $repo $relative) -Algorithm SHA256).Hash.ToLowerInvariant() }
function Read([string]$relative) { Get-Content -LiteralPath (Join-Path $repo $relative) -Encoding UTF8 -Raw | ConvertFrom-Json }
function Rows([string]$relative) { @(Get-Content -LiteralPath (Join-Path $repo $relative) -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json }) }
function Stats($values) {
    $v=@($values); Require ($v.Count -gt 0) 'Empty statistics input'
    foreach($x in $v) { Require (-not [double]::IsNaN($x) -and -not [double]::IsInfinity($x)) 'Nonfinite statistic' }
    $m=$v | Measure-Object -Minimum -Maximum -Average -Sum
    $sorted=@($v | Sort-Object)
    [ordered]@{count=$v.Count;minimum=$m.Minimum;maximum=$m.Maximum;mean=$m.Average;sum=$m.Sum;p95_nearest_rank=$sorted[[int][Math]::Ceiling(.95*$v.Count)-1]}
}
function Gripper($rows,[int]$dim) {
    $commands=@($rows | ForEach-Object { [double]$_.executed_action[$dim] })
    $observed=@($rows | ForEach-Object { [double]$_.state_after[$dim] })
    $errors=@($rows | ForEach-Object { [Math]::Abs([double]$_.executed_action[$dim]-[double]$_.state_after[$dim]) })
    $changes=@(for($i=1;$i -lt $commands.Count;$i++){[Math]::Abs($commands[$i]-$commands[$i-1])})
    [ordered]@{dimension=$dim;commands=(Stats $commands);observed=(Stats $observed);absolute_command_observation_gap=(Stats $errors)
        adjacent_command_change=$(if($changes.Count){Stats $changes}else{$null})
        command_le_0_2=@($commands | Where-Object {$_ -le .2}).Count;command_ge_0_8=@($commands | Where-Object {$_ -ge .8}).Count
        thresholds_are_descriptive_only=$true;physical_grasp_inferred=$false}
}
if($HelpersOnly){return}
$destination=[IO.Path]::GetFullPath((Join-Path $repo $Output))
Require ($Output.StartsWith('reports/training/') -and $destination.StartsWith($repo+[IO.Path]::DirectorySeparatorChar)) 'Output must stay in training reports'
Require (-not (Test-Path -LiteralPath $destination)) 'Output exists; preserve evidence'
$receiver='.work/paired-ABC-measurement-20260928-003-received-20260930-001'
$root="$receiver/files"
$inventory=@{}
foreach($line in Get-Content -LiteralPath (Join-Path $repo "$receiver/remote.sha256")) {
    Require ($line -match '^([0-9a-f]{64})  \./(.+)$') 'Invalid manifest row'
    $expected,$name=$Matches[1],$Matches[2]
    Require ($name -notmatch '(^|/)\.\.(/|$)|[:\\]' -and -not [IO.Path]::IsPathRooted($name)) 'Unsafe manifest path'
    Require (-not $inventory.ContainsKey($name)) 'Duplicate manifest entry'
    Require ((Hash "$root/$name") -eq $expected) "Hash drift: $name"
    $inventory[$name]=$expected
}
$actual=@(Get-ChildItem -LiteralPath (Join-Path $repo $root) -File -Recurse)
Require ($actual.Count -eq $inventory.Count) 'Inventory count mismatch'
foreach($file in $actual) { Require ($inventory.ContainsKey($file.FullName.Substring((Join-Path $repo $root).Length+1).Replace('\','/'))) 'Unlisted received file' }
$episodes=@();$bank=@();$sources=@{}
foreach($arm in 'A','B','C') {
    $plan=Read "$root/evaluation-plan-$arm.json"
    foreach($p in $plan.source_sha256.PSObject.Properties) { Require ((Hash $p.Name) -eq $p.Value) "Source drift: $($p.Name)"; $sources[$p.Name]=$p.Value }
    $summary=Read "$root/measurement-$arm/summary.json"
    $contract=Read "$root/artifacts/$arm/action_contract.json"
    Require ($summary.status -eq 'completed' -and $summary.episodes.Count -eq 5 -and $summary.optimizer_updates -eq 0) 'Incomplete measurement'
    foreach($item in $summary.episodes) {
        $path="$root/measurement-$arm/seed-$($item.seed).jsonl"
        Require ((Hash $path) -eq $item.trace_sha256) 'Trace/summary binding drift'
        $all=@(Rows $path); $rows=@($all | Select-Object -Skip 1)
        Require ($all[0].event -eq 'reset' -and $rows.Count -eq $item.rollout_length) 'Reset or length mismatch'
        $limits=[int]$all[0].joint_limit_violations;$contacts=$all[0].unexpected_pairs.Count;$clips=0;$success=$false;$maxReward=0.0
        for($i=0;$i -lt $rows.Count;$i++) {
            $r=$rows[$i]; Require ($r.step -eq $i+1 -and $r.event -eq 'step' -and $r.seed -eq $item.seed) 'Step identity gap'
            foreach($key in 'raw_action','processed_action','executed_action','state_before','state_after') {
                Require ($r.$key.Count -eq 14) 'Wrong vector width'
                foreach($v in $r.$key) { Require (-not [double]::IsNaN($v) -and -not [double]::IsInfinity($v)) 'Nonfinite vector' }
            }
            $changed=0
            for($d=0;$d -lt 14;$d++) {
                $lo=[double][single]$contract.dimensions[$d].minimum;$hi=[double][single]$contract.dimensions[$d].maximum
                $projected=[Math]::Max($lo,[Math]::Min($hi,[double]$r.raw_action[$d]))
                Require ($r.processed_action[$d] -eq $projected -and $r.executed_action[$d] -eq $projected) 'Action chain mismatch'
                if($projected -ne $r.raw_action[$d]){$changed++}
                if($i){Require ($r.state_before[$d] -eq $rows[$i-1].state_after[$d]) 'State continuity mismatch'}
            }
            Require ($changed -eq $r.raw_clipped_fields) 'Clip count mismatch'
            Require (-not [double]::IsNaN($r.reward) -and -not [double]::IsInfinity($r.reward)) 'Nonfinite reward'
            $limits+=$r.joint_limit_violations;$contacts+=$r.unexpected_pairs.Count;$clips+=$changed;$success=$success -or $r.success;$maxReward=[Math]::Max($maxReward,$r.reward)
            if($r.done -or $r.joint_limit_violations -or $r.unexpected_pairs.Count){Require ($i -eq $rows.Count-1) 'Continued after stop'}
        }
        Require ($limits -eq $item.joint_limit_violations -and $contacts -eq $item.unexpected_collisions -and $clips -eq $item.raw_clipped_fields -and $success -eq $item.success -and $maxReward -eq $item.maximum_reward) 'Summary counts mismatch'
        $full=($rows.Count -eq 500 -and $limits -eq 0 -and $contacts -eq 0 -and -not $success)
        Require (($item.outcome -eq 'task_failed') -eq $full) 'Failure classification mismatch'
        $tail=@($rows | Select-Object -Last 100)
        $episodes += [ordered]@{arm=$arm;seed=$item.seed;trace=$path;trace_sha256=$item.trace_sha256;steps=$rows.Count
            outcome=$item.outcome;maximum_reward=$maxReward;joint_limit_counts=$limits;unexpected_contact_counts=$contacts;raw_clipped_fields=$clips
            full_horizon_safe_failure=$full;recorded_geometry_rows=@($rows | Where-Object {$_.PSObject.Properties.Name -contains 'geometry'}).Count
            left_full=(Gripper $rows 6);right_full=(Gripper $rows 13);tail_steps=$tail.Count;tail_start_step=$tail[0].step
            left_tail=(Gripper $tail 6);right_tail=(Gripper $tail 13)}
    }
    $b=Read "$root/measurement-$arm/recovery-bank/bank-summary.json"
    foreach($item in $b.rows) {
        if($item.measured_steps -eq 0) { Require ($item.status -eq 'unmeasured_branch_not_created') 'Unmeasured state mislabeled';continue }
        $path="$root/measurement-$arm/recovery-bank/$($item.pair_id).jsonl";$rows=@(Rows $path)
        Require ($rows.Count -eq $item.measured_steps -and $item.status -eq 'physical_safety_stop') 'Bank outcome mismatch'
        $left=0;$right=0;$both=0;$safeBoth=0;$longest=0;$streak=0;$unsafeBoth=0;$rewards=@();$bothSteps=@();$lossWindows=@()
        for($i=0;$i -lt $rows.Count;$i++) {
            $r=$rows[$i];Require ($r.step -eq $i+1) 'Bank step gap'
            foreach($k in 'socket_grasp_contact','peg_grasp_contact','peg_socket_contact'){Require ($r.geometry.$k -is [bool]) 'Missing explicit contact boolean'}
            $safe=($r.joint_limit_violations -eq 0 -and $r.unexpected_contacts.Count -eq 0)
            if(-not $safe){Require ($i -eq $rows.Count-1) 'Bank continued after safety stop'}
            if($r.geometry.socket_grasp_contact){$left++};if($r.geometry.peg_grasp_contact){$right++}
            if($r.geometry.socket_grasp_contact -and $r.geometry.peg_grasp_contact){$both++;$bothSteps+=$r.step;if($safe){$safeBoth++}else{$unsafeBoth++}}
            if($i -gt 0){
                $previous=$rows[$i-1]
                if($previous.geometry.socket_grasp_contact -and $previous.geometry.peg_grasp_contact -and -not ($r.geometry.socket_grasp_contact -and $r.geometry.peg_grasp_contact)){
                    $lossWindows += [ordered]@{previous_step=$previous.step;step=$r.step;previous_line=$i;line=$i+1
                        previous_geometry=$previous.geometry;geometry=$r.geometry;safe=$safe;previous_reward=$previous.reward;reward=$r.reward
                        previous_command=$previous.executed_action;command=$r.executed_action;previous_state_after=$previous.state_after;state_after=$r.state_after
                        command_delta=@(for($d=0;$d -lt 14;$d++){[double]$r.executed_action[$d]-[double]$previous.executed_action[$d]})
                        observed_delta=@(for($d=0;$d -lt 14;$d++){[double]$r.state_after[$d]-[double]$previous.state_after[$d]})
                        state_before_was_not_recorded=$true;causal_attribution_proven=$false}
                }
            }
            if($safe -and $r.geometry.socket_grasp_contact -and $r.geometry.peg_grasp_contact){$streak++}else{$streak=0}
            $longest=[Math]::Max($longest,$streak);$rewards+=[double]$r.reward
        }
        Require (-not $safe -and -not $item.recovery_event.geometric_recovery -and $longest -lt 5) 'Recovery event/count mismatch'
        $bank += [ordered]@{arm=$arm;pair_id=$item.pair_id;trace=$path;trace_sha256=(Hash $path);steps=$rows.Count
            socket_contact_steps=$left;peg_contact_steps=$right;both_contact_steps=$both;safe_both_contact_steps=$safeBoth
            unsafe_both_contact_steps=$unsafeBoth;both_contact_step_indices=$bothSteps;contact_loss_windows=$lossWindows
            longest_safe_both_contact_streak=$longest;reward=(Stats $rewards)
            contact_is_not_force_closure=$true}
    }
}
$full=@($episodes | Where-Object full_horizon_safe_failure)
# Bind the source 2447 trace for context, without claiming a verified recovery chain.
$bankRoot='.work/paired-bank-v3-20260927-001-independent-received/extracted/collection'
$sourceManifest=Read "$bankRoot/manifest.json";$contextHashes=@{}
Require ((Hash "$bankRoot/manifest.json") -eq $plan.bank_manifest_sha256) 'Source bank manifest drift'
foreach($entry in $sourceManifest.files){
    if($entry.path -in @('bank-003b-scene-2447/R/trace.jsonl','bank-003b-scene-2447/R/summary.json')){
        Require ((Hash "$bankRoot/$($entry.path)") -eq $entry.sha256) '2447 source context drift';$contextHashes[$entry.path]=$entry.sha256
    }
}
Require ($contextHashes.Count -eq 2) 'Missing source context inventory'
$sourceSummary=Read "$bankRoot/bank-003b-scene-2447/R/summary.json"
$result=[ordered]@{schema='rosetta.saved_failure_modes.v1';date='2026-10-03';branch=(git -C $repo branch --show-current);head=(git -C $repo rev-parse HEAD)
    manifest_sha256=(Hash "$receiver/remote.sha256");verified_measurement_files=$inventory.Count;verified_runner_sources=$sources.Count;source_sha256=$sources
    episodes=$episodes;bank=$bank;scene_count=$episodes.Count;scene_steps=($episodes.steps|Measure-Object -Sum).Sum
    full_horizon_safe_failure_count=$full.Count;full_horizon_safe_failure_steps=($full.steps|Measure-Object -Sum).Sum
    scene_physical_stop_count=@($episodes | Where-Object {-not $_.full_horizon_safe_failure}).Count
    bank_measured_states=$bank.Count;bank_steps=($bank.steps|Measure-Object -Sum).Sum;bank_one_step_stops=@($bank|Where-Object steps -eq 1).Count
    source_2447_context=[ordered]@{source_files_verified=$contextHashes;deviation_verified=$sourceSummary.deviation.verified
        teacher_terminal_success=$sourceSummary.task_success;teacher_verified_recovery_chain=$sourceSummary.verified_recovery_chain
        correction_start=$sourceSummary.correction_start;terminal_success_step=$sourceSummary.terminal_success_step
        same_runtime_teacher_student_parity_proven=$false}
    scope='saved file integrity, endpoint contacts and descriptive arithmetic only';model_or_simulator_executed=$false
    independent_agent_verification=$false;causal_attribution_proven=$false;formal_gate_result=$null;m2_complete=$false
    limitations=@('No ordinary-scene grasp/object/EEF geometry','No full physical joint limits or actuator signals','No current restore/processor/noise parity','Commands and observed apertures do not certify grasp','Control-step snapshots do not exclude transient within-step contacts')}
$json=($result | ConvertTo-Json -Depth 15)+[Environment]::NewLine
$stream=[IO.File]::Open($destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes($json);$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
Write-Output "PASS: $($inventory.Count) hashes, $($episodes.Count) scenes, $($bank.Count) bank endpoints, $($full.Count) complete safe failures; $Output"
