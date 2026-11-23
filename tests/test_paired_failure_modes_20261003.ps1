# Arithmetic counterexamples and separately reconstructed saved-evidence checks.
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$checks=0
function Check($condition,[string]$message){if(-not $condition){throw "FAIL: $message"};$script:checks++}
$script=Join-Path $repo 'scripts/investigate_paired_failure_modes_20261003.ps1'
$tokens=$null;$errors=$null
[void][Management.Automation.Language.Parser]::ParseFile($script,[ref]$tokens,[ref]$errors)
Check ($errors.Count -eq 0) 'Parser'
. $script -HelpersOnly
$v=Stats @(10,1,9,2,8,3,7,4,6,5)
Check ($v.mean -eq 5.5 -and $v.p95_nearest_rank -eq 10 -and $v.sum -eq 55) 'Known arithmetic and percentile'
foreach($bad in @(@(),@([double]::NaN),@([double]::PositiveInfinity))){
    $rejected=$false;try{Stats $bad | Out-Null}catch{$rejected=$true}
    Check $rejected 'Empty/nonfinite rejection'
}
# Two full-width vectors catch PowerShell pipeline flattening of array-valued properties.
$r1=[pscustomobject]@{executed_action=@(0,0,0,0,0,0,.1,0,0,0,0,0,0,.8);state_after=@(0,0,0,0,0,0,.2,0,0,0,0,0,0,.7)}
$r2=[pscustomobject]@{executed_action=@(0,0,0,0,0,0,.3,0,0,0,0,0,0,1);state_after=@(0,0,0,0,0,0,.4,0,0,0,0,0,0,.9)}
$g=Gripper @($r1,$r2) 6
Check ($g.commands.count -eq 2 -and [Math]::Abs($g.commands.mean-.2) -lt 1e-14 -and $g.command_le_0_2 -eq 1) 'Vector-column extraction'
Check ([Math]::Abs($g.absolute_command_observation_gap.mean-.1) -lt 1e-14 -and [Math]::Abs($g.adjacent_command_change.mean-.2) -lt 1e-14) 'Observation gap and adjacent command arithmetic'
$relative='reports/training/paired-failure-modes-2026-10-03-002.json'
$result=Get-Content -LiteralPath (Join-Path $repo $relative) -Raw -Encoding UTF8 | ConvertFrom-Json
Check ($result.verified_measurement_files -eq 151 -and $result.verified_runner_sources -eq 14) 'Integrity inventory'
Check ($result.scene_count -eq 15 -and $result.scene_steps -eq 4187 -and $result.full_horizon_safe_failure_steps -eq 2000) 'Scene denominator'
Check ($result.bank_measured_states -eq 24 -and $result.bank_steps -eq 200 -and $result.bank_one_step_stops -eq 19) 'Bank denominator'
Check (-not $result.model_or_simulator_executed -and -not $result.causal_attribution_proven -and -not $result.independent_agent_verification) 'Evidence scope'
$full=@($result.episodes|Where-Object full_horizon_safe_failure)
Check ((@($full|ForEach-Object {"$($_.arm):$($_.seed)"}) -join ',') -eq 'A:1000,A:1004,B:1001,C:1000') 'Full safe failure identities'
foreach($e in $full){
    $raw=@(Get-Content -LiteralPath (Join-Path $repo $e.trace) -Encoding UTF8|ForEach-Object {$_|ConvertFrom-Json})
    Check ($raw.Count -eq 501 -and @($raw|Where-Object {$_.PSObject.Properties.Name -contains 'geometry'}).Count -eq 0) 'Geometry absent rather than imputed'
    foreach($d in 6,13){
        $sum=0.0;$low=0;$high=0
        for($i=401;$i -le 500;$i++){$value=[double]$raw[$i].executed_action[$d];$sum+=$value;if($value -le .2){$low++};if($value -ge .8){$high++}}
        $stats=if($d -eq 6){$e.left_tail}else{$e.right_tail}
        Check ($stats.commands.count -eq 100 -and [Math]::Abs($stats.commands.mean-$sum/100) -lt 1e-14 -and $low -eq $stats.command_le_0_2 -and $high -eq $stats.command_ge_0_8) 'Tail re-sum from original rows'
    }
}
$c=@($result.bank|Where-Object {$_.arm -eq 'C' -and $_.pair_id -eq 'bank-003b-scene-2447'})[0]
$b=@($result.bank|Where-Object {$_.arm -eq 'B' -and $_.pair_id -eq 'bank-003b-scene-2447'})[0]
Check (($b.both_contact_step_indices -join ',') -eq '1,2,3' -and $b.longest_safe_both_contact_streak -eq 3 -and $b.reward.maximum -eq 0) 'Contact and reward are distinct'
Check (($c.both_contact_step_indices -join ',') -eq '1,3' -and $c.longest_safe_both_contact_streak -eq 1 -and $c.reward.maximum -eq 2) 'Intermittent versus sustained contact'
$loss=$c.contact_loss_windows[0]
Check ($loss.step -eq 2 -and $loss.command_delta[6] -lt 0 -and $loss.command_delta[13] -lt 0 -and -not $loss.geometry.peg_grasp_contact -and $loss.safe) 'Right contact loss while closing before safety stop'
Check ($result.source_2447_context.deviation_verified -and $result.source_2447_context.teacher_terminal_success -and -not $result.source_2447_context.teacher_verified_recovery_chain -and -not $result.source_2447_context.same_runtime_teacher_student_parity_proven) 'Source terminal success is not recovery proof'
$before=(Get-FileHash -LiteralPath (Join-Path $repo $relative)).Hash
$rejected=$false;try{& $script -Output $relative}catch{$rejected=$_.Exception.Message -like '*Output exists*'}
Check ($rejected -and (Get-FileHash -LiteralPath (Join-Path $repo $relative)).Hash -eq $before) 'Create-only protection'
Write-Output "PASS: $checks arithmetic/evidence checks; no ML/container tests or new physical measurements"
