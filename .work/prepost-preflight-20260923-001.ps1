$ErrorActionPreference = 'Stop'
# File/schema inspection only: no model, tensor payload decoding or ML runtime.
$base = 'models/lerobot--smolvla_base/c83c3163b8ca9b7e67c509fffd9121e66cb96205'
$receiptPath = 'runs/canonical-furnace-received-20260914-002/runs/canonical-fullframes-20260914-001/checkpoint.json'
$receipt = Get-Content -Raw $receiptPath | ConvertFrom-Json
$manifest = Get-Content -Raw "$base/model_manifest.json" | ConvertFrom-Json
$rows = [System.Collections.Generic.List[object]]::new()
function Check-File($path, $expected) {
    $present = Test-Path -LiteralPath $path -PathType Leaf
    $actual = if ($present) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    $rows.Add([ordered]@{path=$path; expected_sha256=$expected; actual_sha256=$actual; matched=($present -and $actual -eq $expected)})
}
foreach ($file in $manifest.files.PSObject.Properties) { Check-File "$base/$($file.Name)" $file.Value.sha256 }
foreach ($step in @('2500','5000')) {
    $folder = 'runs/canonical-checkpoints-received-20260914-002/step-' + $step.PadLeft(6,'0') + '/verified'
    foreach ($file in $receipt.recovery_checkpoints.$step.PSObject.Properties) { Check-File "$folder/$($file.Name)" $file.Value }
}
$result = [ordered]@{
    id='prepost-preflight-20260923-001'; checked_at_utc=[DateTime]::UtcNow.ToString('o')
    scope='static_file_identity_only'; comparison_complete=$false
    expected_hash_sources=@("$base/model_manifest.json", $receiptPath)
    checks=@($rows.ToArray()); files_checked=$rows.Count
    all_hashes_matched=(@($rows | Where-Object { -not $_.matched }).Count -eq 0)
    model_loaded=$false; tensor_values_compared=$false; optimizer_steps=0; ssh_used=$false
    container_probe='Docker Desktop is manually paused'
    step0='No standalone step-0 artifact found in inspected canonical recovery inventory; reconstruct and verify before use'
}
$out = 'reports/training/m2-smolvla-prepost-preflight-2026-09-23.json'
$stream = [IO.File]::Open($out, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($result | ConvertTo-Json -Depth 12) + "`n")
    $stream.Write($bytes, 0, $bytes.Length)
} finally { $stream.Dispose() }
[pscustomobject]@{files_checked=$result.files_checked; all_hashes_matched=$result.all_hashes_matched; output=$out} | ConvertTo-Json
if (-not $result.all_hashes_matched) { exit 1 }
