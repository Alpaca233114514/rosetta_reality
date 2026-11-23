param([string]$Output='reports/training/paired-physical-command-localization-2026-10-04-002.json')
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $Output){throw 'Output already exists; preserve evidence'}
$receivers=@(
 '.work/paired-boundary-cpu-20261004-002-received',
 '.work/paired-saved-command-physics-20261004-001-received',
 '.work/paired-saved-command-physics-20261004-002-received',
 '.work/paired-gripper-targets-20261004-001-received',
 '.work/paired-arm-targets-20261004-001-received')
$inventories=foreach($root in $receivers){
 $lines=Get-Content -LiteralPath "$root/remote.sha256"
 foreach($line in $lines){
  if($line -notmatch '^([a-f0-9]{64})  \./(.+)$'){throw 'Invalid receipt manifest'}
  $expected=$Matches[1];$name=$Matches[2]
  if($name -match '(^/|(^|/)\.\.(/|$)|\\|:)'){throw 'Receipt path escapes root'}
  if((Get-FileHash -LiteralPath "$root/files/$name" -Algorithm SHA256).Hash.ToLower() -ne $expected){throw 'Receipt SHA mismatch'}
 }
 if(@(Get-ChildItem -LiteralPath "$root/files" -Recurse -File).Count -ne $lines.Count){throw 'Receipt file set mismatch'}
 $worker=Get-Content -LiteralPath "$root/files/worker-exited.json" -Raw|ConvertFrom-Json
 $guard=Get-Content -LiteralPath "$root/files/guard-exited.json" -Raw|ConvertFrom-Json
 if($guard.status -ne 'worker_exited' -or $guard.error_type -or $worker.cleanup.status -ne 'owned_group_empty'){throw 'Unclosed process boundary'}
 $nested="$root/files/physics-readout/manifest.json"
 if(Test-Path -LiteralPath $nested){
  $manifest=Get-Content -LiteralPath $nested -Raw|ConvertFrom-Json
  foreach($entry in $manifest.files.PSObject.Properties){
   if((Get-FileHash -LiteralPath "$root/files/physics-readout/$($entry.Name)" -Algorithm SHA256).Hash.ToLower() -ne $entry.Value){throw 'Nested physical readout SHA mismatch'}
  }
 }
 [ordered]@{receiver=$root;regular_files=$lines.Count;manifest_sha256=(Get-FileHash "$root/remote.sha256").Hash.ToLower();worker_error_type=$worker.error_type;outcomes=$worker.outcomes;guard_status=$guard.status;shutdown_requested=$worker.shutdown_requested}
}
$artifact=Get-Content '.work/paired-boundary-cpu-20261004-002-received/artifact-verification.json' -Raw|ConvertFrom-Json
if(-not $artifact.all_match -or @($artifact.arms).Count -ne 3 -or @($artifact.arms|ForEach-Object files).Count -ne 39){throw 'Endpoint artifact identity incomplete'}
$physicalRoots=@($receivers[2],$receivers[3],$receivers[4])
$experiments=foreach($root in $physicalRoots){
 $result=Get-Content "$root/files/physics-readout/result.json" -Raw|ConvertFrom-Json
 if($result.status -ne 'completed' -or $result.model_forward_calls -ne 0 -or $result.optimizer_updates -ne 0){throw 'Diagnostic result incomplete or scope changed'}
 $result
}
$scanRoot="$($receivers[4])/files/physics-readout"
$controls=foreach($case in @('teacher','A','B','C')){
 $record=Get-Content "$scanRoot/$case.json" -Raw|ConvertFrom-Json
 if(-not $record.full_saved_physical_readout_exact -or -not $record.historical_state_exact){throw 'Saved physical control not exact'}
 if($case -eq 'teacher' -and -not $record.historical_integration_exact){throw 'Teacher integration changed'}
 [ordered]@{case=$case;command=$record.command;before_logical_state=$record.before.logical_state;after_logical_state=$record.after.logical_state;safe_after=$record.safe_after;violation_count=$record.violation_count;violation_details=@($record.after.joint_limits|Where-Object violation);unexpected_contacts=$record.after.unexpected_contacts;compiled_model_sha256=$record.compiled_model_sha256;full_saved_physical_readout_exact=$record.full_saved_physical_readout_exact;historical_integration_exact=$record.historical_integration_exact}
}
$jointRows=foreach($file in Get-ChildItem $scanRoot -File -Filter '*joint*.json'){
 $r=Get-Content $file.FullName -Raw|ConvertFrom-Json
 $baseline=Get-Content "$scanRoot/$($r.base).json" -Raw|ConvertFrom-Json
 $donor=Get-Content "$scanRoot/$($r.donor).json" -Raw|ConvertFrom-Json
 for($i=0;$i -lt 14;$i++){
  $expected=if($i -eq $r.index){$donor.command[$i]}else{$baseline.command[$i]}
  if($r.command[$i] -ne $expected){throw 'More than registered joint target changed'}
 }
 [ordered]@{case=$r.case;base=$r.base;donor=$r.donor;index=$r.index;joint=$r.joint;target=$r.command[$r.index];initial_joint_state=$r.before.logical_state[$r.index];after_joint_state=$r.after.logical_state[$r.index];safe_after=$r.safe_after;violations=$r.violation_count;violation_details=@($r.after.joint_limits|Where-Object violation);unexpected_contacts=$r.after.unexpected_contacts}
}
if(@($jointRows).Count -ne 72){throw 'Single-joint scan incomplete'}
$safeRestorations=@($jointRows|Where-Object {$_.base -ne 'teacher' -and $_.safe_after})
if($safeRestorations.Count -ne 2 -or @($safeRestorations|Where-Object {$_.index -ne 8}).Count){throw 'Safety restoration conclusion drift'}
foreach($arm in @('B','C')){
 $reciprocal=@($jointRows|Where-Object {$_.base -eq 'teacher' -and $_.donor -eq $arm -and $_.index -eq 8})
 if($reciprocal.Count -ne 1 -or $reciprocal[0].safe_after -or $reciprocal[0].unexpected_contacts.Count -ne 2){throw 'Reciprocal shoulder collision not reproduced'}
}
$grippers=foreach($case in @('A_teacher_grippers','B_teacher_grippers','C_teacher_grippers','teacher_A_grippers','teacher_B_grippers','teacher_C_grippers')){
 $r=Get-Content "$($receivers[3])/files/physics-readout/$case.json" -Raw|ConvertFrom-Json
 [ordered]@{case=$case;safe_after=$r.safe_after;violations=$r.violation_count;unexpected_contacts=$r.after.unexpected_contacts;violation_details=@($r.after.joint_limits|Where-Object violation)}
}
$data=[ordered]@{
 schema='rosetta.paired_physical_localization_report.v1';date='2026-10-04';pair='bank-003b-scene-2445';
 status='physical_command_boundary_localized_model_generation_unmeasured';
 scope='Physical-only CPU AutoDL replay of saved commands; full Python owners and pixels not restored; no new policy inference';
 registered_runtime_profile='configs/runtime/autodl_rtx4090.yaml';nested_docker_used=$false;gpu_admitted=$false;
 new_control_steps=($experiments|Measure-Object -Property control_steps -Sum).Sum;new_model_forward_calls=0;optimizer_updates=0;
 exact_control_runs=12;teacher_integration_values_per_control=293;receipts=$inventories;
 verified_receipt_regular_files=[int]($inventories|ForEach-Object {$_.regular_files}|Measure-Object -Sum).Sum;
 artifact_payload_checks=39;artifact_payload_all_sha_match=$true;artifact_inventory_sha256=(Get-FileHash '.work/paired-boundary-cpu-20261004-002-received/artifact-verification.json').Hash.ToLower();
 failed_attempts=@([ordered]@{name='boundary-cpu-001';stage='before_launch';reason='Source drift in old supervisor';expected_sha256='acff18892854a745946f86f216c4aa550bb25e67074d6afba7d28094638797c5';actual_sha256='14202a5c8ffa2b1d371e358e3eef8f1dbea86e661180676d21cd30cd337f455c';executed_tests=0},[ordered]@{name='saved-command-physics-001';stage='dm_control_import';reason='Cannot initialize a headless EGL display';control_steps=0});
 controls=$controls;gripper_interventions=$grippers;single_joint_interventions=$jointRows;safe_single_joint_restorations=$safeRestorations;
 teacher_damaging_single_joint_swaps=@($jointRows|Where-Object {$_.base -eq 'teacher' -and -not $_.safe_after});
 interpretation=[ordered]@{bc_first_collision='right_shoulder saved target has reciprocal one-step causal evidence at this initial state';a_first_violation='two counter-finger joint limits; no single teacher arm target removes both';gripper_only_repair_sufficient=$false;model_input_processor_noise_cause_proven=$false;unique_learning_root_cause_proven=$false;rollout_repair_proven=$false};
 formal_gate_result=$null;m2_complete=$false;no_safety_threshold_changed=$true;gpu_power_changed=$false;weights_changed=$false;
 limitations=@('No images or full Python-owner restore in physical replay','Only this saved scene and first command interval tested','Teacher targets use privileged information; interventions are diagnostic, not deployable policy fixes','Original A/B/C generated inputs, processor transitions and Gaussian tensors still need registered CUDA capture','A multi-joint interactions and later task failures remain unresolved')
}
[IO.File]::WriteAllText((Join-Path $PWD $Output),($data|ConvertTo-Json -Depth 20)+"`n",(New-Object System.Text.UTF8Encoding($false)))
Write-Output ('Verified '+$data.verified_receipt_regular_files+' new receipt files; '+$data.new_control_steps+' physical control steps; 72 single-joint interventions; zero new policy forwards.')
