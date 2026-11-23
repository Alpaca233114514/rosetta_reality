# Historical evidence regressions; PowerShell only, no ML or simulator imports.
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$checks=0
function Check($condition,[string]$name) {
    if (-not $condition) { throw "FAIL: $name" }
    $script:checks++
}
$tokens=$null; $errors=$null
[void][Management.Automation.Language.Parser]::ParseFile((Join-Path $repo 'scripts/investigate_paired_boundary_20261003.ps1'),[ref]$tokens,[ref]$errors)
Check ($errors.Count -eq 0) 'Investigation script syntax'
$relative='reports/training/paired-boundary-independent-check-2026-10-03-002.json'
$path=Join-Path $repo $relative
$result=Get-Content -LiteralPath $path -Encoding UTF8 -Raw | ConvertFrom-Json
Check ($result.measurement_files_verified -eq 151 -and $result.runner_sources_verified -eq 14) 'Fresh integrity coverage'
Check ($result.saved_processor_config_files_verified -eq 15 -and $result.source_branch_hold_command_parity.passed -and -not $result.source_branch_hold_command_parity.policy_inference_included) 'Saved processors and limited physics A/A evidence'
Check ($result.teacher_rows -eq 293 -and $result.teacher_safe -and $result.teacher_terminal_step -eq 292) 'Teacher safety and terminal boundary'
Check ($result.deviation_step -eq 177 -and $result.source_correction_step -eq 178 -and $result.source_correction_line -eq 179) 'Zero-based source versus one-based line alignment'
Check (-not $result.source_correction_grasps.socket -and -not $result.source_correction_grasps.peg) 'Correction starts before either grasp'
Check (-not $result.causal_attribution_proven -and -not $result.old_python_unit_tests_rerun) 'No causal or runtime overclaim'
foreach ($arm in $result.arms) {
    Check ($arm.recorded_steps -eq 1 -and $arm.raw_clips -eq 0 -and ($arm.violations -gt 0 -or $arm.contacts.Count -gt 0)) "Arm $($arm.arm) legal command then physical stop"
    Check (-not $arm.actual_initial_state_recorded -and -not $arm.same_runtime_teacher_student_parity_proven) "Arm $($arm.arm) missing evidence retained"
    $grippers=@($arm.dimensions | Where-Object index -in 6,13)
    Check ($grippers.Count -eq 2 -and @($grippers | Where-Object { [Math]::Abs($_.teacher_command-0.09000000357627869) -gt 1e-12 -or $_.student_minus_teacher -le 0.3 }).Count -eq 0) "Arm $($arm.arm) first closing-command discrepancy"
}
$before=(Get-FileHash -LiteralPath $path).Hash
$rejected=$false
try { & (Join-Path $repo 'scripts/investigate_paired_boundary_20261003.ps1') -Output $relative } catch { $rejected=$_.Exception.Message -like '*Output exists*' }
Check ($rejected -and (Get-FileHash -LiteralPath $path).Hash -eq $before) 'Create-only output preserves evidence'
Write-Output "PASS: $checks PowerShell static/evidence regression checks; Python/ML/container tests not run"
