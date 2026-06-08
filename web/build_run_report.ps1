<#
  build_run_report.ps1 — READ-ONLY MaskBench run viewer generator.

  Reads a live/finished MaskBench run directory and writes web/run-report.html
  in the MaskingOPS design language. It only READS the output/dataset dirs
  (log file, inference_times.json, poses/, npz/, config.yml) — it never writes
  to them, imports maskbench, or touches Docker, so it is safe to run against a
  job that is still in progress.

  Usage (from repo root):
    pwsh ./web/build_run_report.ps1                 # latest run, paths from .env
    pwsh ./web/build_run_report.ps1 -Run <name>     # a specific run folder
    pwsh ./web/build_run_report.ps1 -SkipDatasetCount   # skip the slow 29k-file scan
#>
[CmdletBinding()]
param(
  [string]$OutputDir,
  [string]$DatasetDir,
  [string]$Run,
  [switch]$SkipDatasetCount
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
function Enc([string]$s) { if ($null -eq $s) { return '' } ($s -replace '&','&amp;' -replace '<','&lt;' -replace '>','&gt;') }

# ── Resolve paths from .env if not supplied ──────────────────────────────
$envFile = Join-Path $repoRoot '.env'
$envMap = @{}
if (Test-Path $envFile) {
  Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([A-Z_]+)\s*=\s*(.+?)\s*(#.*)?$') { $envMap[$matches[1]] = $matches[2].Trim() }
  }
}
if (-not $OutputDir)  { $OutputDir  = $envMap['MASKBENCH_OUTPUT_DIR'] }
if (-not $DatasetDir) { $DatasetDir = $envMap['MASKBENCH_DATASET_DIR'] }
if (-not $OutputDir -or -not (Test-Path $OutputDir)) { throw "Output dir not found: '$OutputDir' (set MASKBENCH_OUTPUT_DIR in .env or pass -OutputDir)" }

# ── Pick the run folder (latest by write time unless -Run given) ──────────
if ($Run) {
  $runDir = Join-Path $OutputDir $Run
} else {
  $runDir = (Get-ChildItem -Path $OutputDir -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}
if (-not $runDir -or -not (Test-Path $runDir)) { throw "No run directory found under $OutputDir" }
$runName = Split-Path $runDir -Leaf
Write-Host "Reading run: $runName" -ForegroundColor Cyan

# ── inference_times.json → per-video timings ─────────────────────────────
$times = @()   # list of [pscustomobject]@{ video; sec; estimator }
$itPath = Join-Path $runDir 'inference_times.json'
if (Test-Path $itPath) {
  try {
    $it = Get-Content $itPath -Raw | ConvertFrom-Json
    foreach ($estProp in $it.PSObject.Properties) {
      foreach ($vidProp in $estProp.Value.PSObject.Properties) {
        $times += [pscustomobject]@{ video = $vidProp.Name; sec = [double]$vidProp.Value; estimator = $estProp.Name }
      }
    }
  } catch { Write-Warning "Could not parse inference_times.json (may be mid-write): $_" }
}
$timed   = $times.Count
$totalS  = ($times | Measure-Object sec -Sum).Sum
$avgS    = if ($timed) { $totalS / $timed } else { 0 }
$maxS    = ($times | Measure-Object sec -Maximum).Maximum
$minS    = ($times | Measure-Object sec -Minimum).Minimum
$slowest = $times | Sort-Object sec -Descending | Select-Object -First 8

# ── poses/ vs npz/ counts (the silent-drop check) ────────────────────────
function CountFiles($p) { if (Test-Path $p) { @(Get-ChildItem $p -Recurse -File -ErrorAction SilentlyContinue).Count } else { 0 } }
$posesN = CountFiles (Join-Path $runDir 'poses')
$npzN   = CountFiles (Join-Path $runDir 'npz')
$rendN  = CountFiles (Join-Path $runDir 'renderings')
$estimators = @(Get-ChildItem (Join-Path $runDir 'npz') -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name)

# ── parse the .log (this run only logs ERRORs to file) ───────────────────
$logEvents = @()   # @{ ts; lvl; stage; video; msg }
$stageMap = @{ 'inference_engine.py'='inference'; 'checkpointer.py'='npz'; 'pose_renderer.py'='render'; 'main.py'='config'; 'dataset.py'='dataset'; 'evaluator.py'='eval' }
$logFile = Get-ChildItem $runDir -Filter '*_maskbench.log' -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($logFile) {
  foreach ($line in (Get-Content $logFile.FullName)) {
    if ($line -match '^(?<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\s+-\s+\S+\s+-\s+(?<lvl>\w+)\s+-\s+(?<loc>\S+)\s+-\s+(?<msg>.*)$') {
      # Capture named groups BEFORE any further -match clobbers $matches
      $ts   = ($matches['ts'] -split ' ')[1]
      $lvl  = $matches['lvl'].ToLower()
      $file = ($matches['loc'] -split ':')[0]
      $msg  = $matches['msg']
      $stage = if ($stageMap.ContainsKey($file)) { $stageMap[$file] } else { 'other' }
      $video = if ($msg -match 'on Video:\s*(.+?)\s+with Estimator:') { $matches[1] } else { '' }
      $logEvents += [pscustomobject]@{ ts = $ts; lvl = $lvl; stage = $stage; video = $video; msg = $msg }
    }
  }
}
$errN  = @($logEvents | Where-Object lvl -eq 'error').Count
$warnN = @($logEvents | Where-Object lvl -eq 'warn').Count

# ── dataset denominator (optional, can be slow on 29k files) ─────────────
$discovered = $null
$byFolder = [ordered]@{}   # top-level dataset folder -> {disc, npz}
$byCorpus = @{}            # filename-prefix corpus     -> {disc, npz}
if (-not $SkipDatasetCount -and $DatasetDir -and (Test-Path $DatasetDir)) {
  try {
    # set of processed stems (npz basenames) for O(1) attribution back to folders
    $npzStems = [System.Collections.Generic.HashSet[string]]::new()
    Get-ChildItem (Join-Path $runDir 'npz') -Recurse -File -ErrorAction SilentlyContinue |
      ForEach-Object { [void]$npzStems.Add([IO.Path]::GetFileNameWithoutExtension($_.Name)) }

    $base = $DatasetDir.TrimEnd('\','/')
    $count = 0
    foreach ($f in [System.IO.Directory]::EnumerateFiles($base, '*.*', 'AllDirectories')) {
      if ($f -notmatch '\.(mp4|avi)$') { continue }
      $count++
      $stem = [IO.Path]::GetFileNameWithoutExtension($f)
      $rel  = $f.Substring($base.Length).TrimStart('\','/')
      $top  = ($rel -split '[\\/]')[0]            # generic: top-level folder (label dir, corpus dir, whatever)
      $corp = ($stem -split '_')[0]               # corpus encoded in filename prefix
      $done = $npzStems.Contains($stem)
      foreach ($pair in @(@($byFolder, $top), @($byCorpus, $corp))) {
        $d = $pair[0]; $k = $pair[1]
        if (-not $d.Contains($k)) { $d[$k] = [pscustomobject]@{ disc = 0; npz = 0 } }
        $d[$k].disc++; if ($done) { $d[$k].npz++ }
      }
    }
    $discovered = $count
  } catch { Write-Warning "Dataset scan failed: $_" }
}
$attempted = $posesN
$pct = if ($discovered -and $discovered -gt 0) { [math]::Round($attempted * 100.0 / $discovered, 1) } else { $null }
$etaTxt = if ($discovered -and $avgS -gt 0) {
  $remS = ($discovered - $attempted) * $avgS
  $h = [math]::Floor($remS/3600); $m = [math]::Round(($remS % 3600)/60)
  "~$h h $m m left at current rate"
} else { '' }

$config = if (Test-Path (Join-Path $runDir 'envision_gesture_challenge.yml')) { Get-Content (Join-Path $runDir 'envision_gesture_challenge.yml') -Raw }
          else { (Get-ChildItem $runDir -Filter '*.yml' | Select-Object -First 1 | Get-Content -Raw) }
$genStamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')

# ── build dynamic HTML fragments ─────────────────────────────────────────
$logRows = if ($logEvents.Count) {
  ($logEvents | ForEach-Object {
    $m = Enc $_.msg
    if ($_.video) { $m = $m -replace [regex]::Escape($_.video), ('<span class="hl">' + (Enc $_.video) + '</span>') }
    "<div class=`"log-line $($_.lvl)`"><span class=`"log-ts`">$($_.ts)</span><span class=`"log-lvl`">$($_.lvl.ToUpper())</span><span class=`"log-stage`">$($_.stage)</span><span class=`"log-msg`">$m</span></div>"
  }) -join "`n"
} else { '<div class="log-empty">No lines in the run logfile yet. (MaskBench only writes ERROR/WARN here; INFO goes to stdout.)</div>' }

$slowRows = ($slowest | ForEach-Object {
  $bw = if ($maxS) { [math]::Round($_.sec / $maxS * 100) } else { 0 }
  "<div class=`"vrow`"><div class=`"c-name`">$(Enc $_.video)</div><div class=`"c-time`">$([math]::Round($_.sec,1))s</div><div class=`"c-detect`"><span class=`"detect-bar`"><i style=`"width:$bw%`"></i></span></div><div class=`"c-status`"><span class=`"tag tag-success`"><i class=`"fas fa-check`"></i> npz written</span></div></div>"
}) -join "`n"

$failRows = ($logEvents | Where-Object lvl -eq 'error' | ForEach-Object {
  "<div class=`"vrow`"><div class=`"c-name`">$(Enc $_.video)</div><div class=`"c-time`">$($_.ts)</div><div class=`"c-detect`"></div><div class=`"c-status`"><span class=`"tag tag-error`"><i class=`"fas fa-xmark`"></i> ragged npz &mdash; skipped</span></div></div>"
}) -join "`n"

# generic per-group (folder / corpus) progress rows
function GroupRows($dict) {
  if (-not $dict -or $dict.Keys.Count -eq 0) {
    return '<div class="vrow"><div class="c-name text-secondary">Run without -SkipDatasetCount to populate this breakdown.</div></div>'
  }
  ($dict.GetEnumerator() | Where-Object { $_.Value.disc -gt 0 } | Sort-Object { $_.Value.disc } -Descending | ForEach-Object {
    $pc = if ($_.Value.disc) { [math]::Round($_.Value.npz * 100.0 / $_.Value.disc) } else { 0 }
    "<div class=`"vrow`"><div class=`"c-name`">$(Enc $_.Key)</div><div class=`"c-num`">$('{0:N0}' -f $_.Value.disc)</div><div class=`"c-num`">$('{0:N0}' -f $_.Value.npz)</div><div class=`"c-detect`"><span class=`"detect-bar`"><i style=`"width:$pc%`"></i></span> $pc%</div></div>"
  }) -join "`n"
}
$folderRows = GroupRows $byFolder
$corpusRows = GroupRows $byCorpus

$discTile = if ($discovered) { "<div class=`"recon-tile`"><div class=`"recon-v`">{0:N0}</div><div class=`"recon-l`">videos discovered</div></div>" -f $discovered }
            else { "<div class=`"recon-tile`"><div class=`"recon-v`">&mdash;</div><div class=`"recon-l`">videos discovered<br>(run with dataset scan)</div></div>" }
$progressHtml = if ($pct) {
@"
<div style="margin-bottom:14px;">
  <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:5px;">
    <span>Processing &middot; $attempted / $('{0:N0}' -f $discovered) videos</span><span class="mono text-secondary">$pct% &middot; $etaTxt</span>
  </div>
  <div class="progress-bar"><div class="progress-fill info" style="width:$([math]::Min($pct,100))%;"></div></div>
</div>
"@ } else { '' }

# ── page template ────────────────────────────────────────────────────────
$html = @"
<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MaskingOPS — Run Report — $runName</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--c-bg:#fff;--c-bg2:#f4f4f4;--c-bgd:#f5f5f5;--c-bd:#e0e0e0;--c-tx:#161616;--c-tx2:#393939;--c-blue:#0f62fe;--c-grn:#198038;--c-grnl:#defbe6;--c-yel:#f1c21b;--c-yell:#fff8e1;--c-red:#da1e28;--c-redl:#fff1f1;--c-bluel:#edf5ff}
body{font-family:'IBM Plex Sans',sans-serif;background:var(--c-bgd);color:var(--c-tx);line-height:1.5}
.accent-bar{height:3px;background:var(--c-tx)}
.app-header{display:flex;align-items:center;justify-content:space-between;padding:0 24px;height:48px;background:var(--c-tx);color:#f4f4f4}
.app-logo{font-size:14px;font-weight:600;display:flex;align-items:center;gap:10px;letter-spacing:.5px;color:#f4f4f4;text-decoration:none}
.header-nav{display:flex;height:100%}.header-nav a{display:flex;align-items:center;gap:6px;padding:0 16px;font-size:13px;color:#c6c6c6;text-decoration:none;border-bottom:2px solid transparent}
.header-nav a.active{color:#fff;border-bottom-color:#fff;font-weight:500}.header-nav a:hover{color:#fff;background:rgba(255,255,255,.08)}
.wrap{max-width:1180px;margin:0 auto;padding:22px 28px 40px}
.banner{display:flex;gap:12px;padding:12px 16px;border:1px solid var(--c-bd);border-left:3px solid var(--c-blue);background:var(--c-bluel);font-size:13px;margin-bottom:18px}
.page-label{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:var(--c-tx2);margin-bottom:4px}
.page-title{font-size:24px;font-weight:300}.page-title strong{font-weight:600}
.sub{font-size:13px;color:var(--c-tx2);margin-top:4px;display:flex;gap:14px;flex-wrap:wrap}.sub .mono{font-family:'IBM Plex Mono',monospace}
.recon{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:18px 0 14px}
.recon-tile{background:#fff;border:1px solid var(--c-bd);padding:14px 16px}
.recon-tile.err{border-left:3px solid var(--c-red)}.recon-tile.warn{border-left:3px solid var(--c-yel)}.recon-tile.ok{border-left:3px solid var(--c-grn)}
.recon-v{font-family:'IBM Plex Mono',monospace;font-size:24px;font-weight:600;line-height:1}.recon-l{font-size:11px;color:var(--c-tx2);margin-top:6px}
.progress-bar{height:4px;background:var(--c-bd);overflow:hidden}.progress-fill{height:100%}.progress-fill.info{background:var(--c-blue)}
.panel{background:#fff;border:1px solid var(--c-bd);margin-bottom:16px}
.panel-h{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid var(--c-bd);font-size:14px;font-weight:600}
.mono{font-family:'IBM Plex Mono',monospace}.text-secondary{color:var(--c-tx2)}
.tag{display:inline-flex;align-items:center;gap:4px;padding:2px 10px;font-size:11px;font-weight:500}
.tag-success{background:var(--c-grnl);color:var(--c-grn)}.tag-error{background:var(--c-redl);color:var(--c-red)}.tag-processing{background:var(--c-tx);color:#fff}
.log-console{max-height:340px;overflow:auto;background:#161616;color:#c6c6c6;font-family:'IBM Plex Mono',monospace;font-size:12px;line-height:1.7;padding:12px 0}
.log-line{display:flex;gap:14px;padding:1px 16px;white-space:nowrap}.log-line:hover{background:rgba(255,255,255,.05)}
.log-ts{color:#6f6f6f}.log-lvl{width:48px;font-weight:600}.log-stage{width:74px;color:#8d8d8d}.log-msg{white-space:pre-wrap}
.log-line.info .log-lvl{color:#78a9ff}.log-line.warn .log-lvl{color:#f1c21b}.log-line.error{background:rgba(218,30,40,.12)}.log-line.error .log-lvl{color:#ff8389}
.log-msg .hl{color:#ffd6a0}.log-empty{padding:18px 16px;color:#6f6f6f}
.vhead,.vrow{display:flex;align-items:center;padding:10px 16px;font-size:13px}
.vhead{background:var(--c-bg2);border-bottom:1px solid var(--c-bd);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:var(--c-tx2)}
.vrow{border-bottom:1px solid var(--c-bd)}.vrow:hover{background:var(--c-bg2)}
.c-name{flex:2;font-family:'IBM Plex Mono',monospace;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.c-time{width:110px;font-family:'IBM Plex Mono',monospace;color:var(--c-tx2)}.c-num{width:96px;font-family:'IBM Plex Mono',monospace;text-align:right;padding-right:16px}.c-detect{width:140px}.c-status{width:180px}
.detect-bar{height:6px;width:100px;background:var(--c-bd);display:inline-block;vertical-align:middle}.detect-bar i{display:block;height:100%;background:var(--c-grn)}
.code{background:#161616;color:#f4f4f4;font-family:'IBM Plex Mono',monospace;font-size:12px;line-height:1.7;padding:16px 20px;overflow:auto;white-space:pre}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
</style></head><body>
<div class="accent-bar"></div>
<div class="app-header">
  <a href="#" class="app-logo"><i class="fas fa-shield-alt"></i> MASKINGOPS</a>
  <nav class="header-nav"><a href="#"><i class="fas fa-th-large"></i> Dashboard</a><a href="#" class="active"><i class="fas fa-chart-bar"></i> MaskBench</a><a href="challenge-builder.html"><i class="fas fa-trophy"></i> Challenge Builder</a></nav>
  <span></span>
</div>
<div class="wrap">
  <div class="banner"><i class="fas fa-circle-info" style="color:var(--c-blue);margin-top:2px"></i><div><strong>Real run snapshot</strong> &mdash; read-only, generated $genStamp from <span class="mono">$runName</span>. Re-run <span class="mono">web/build_run_report.ps1</span> to refresh. Nothing here writes to the run.</div></div>

  <p class="page-label">MaskBench &middot; Run report</p>
  <h1 class="page-title"><strong>$runName</strong></h1>
  <div class="sub"><span class="mono">$runDir</span><span>$($estimators -join ', ')</span><span>avg $([math]::Round($avgS,2))s &middot; max $([math]::Round($maxS,1))s &middot; $([math]::Round($totalS/60))m compute</span></div>

  <div class="recon">
    $discTile
    <div class="recon-tile ok"><div class="recon-v" style="color:var(--c-grn)">$('{0:N0}' -f $npzN)</div><div class="recon-l">npz written</div></div>
    <div class="recon-tile"><div class="recon-v">$('{0:N0}' -f $posesN)</div><div class="recon-l">pose files (attempted)</div></div>
    <div class="recon-tile err"><div class="recon-v" style="color:var(--c-red)">$($posesN - $npzN)</div><div class="recon-l">npz dropped (poses&minus;npz)</div></div>
    <div class="recon-tile err"><div class="recon-v" style="color:var(--c-red)">$errN</div><div class="recon-l">errors in log</div></div>
    <div class="recon-tile"><div class="recon-v">$($estimators.Count)</div><div class="recon-l">estimators ran</div></div>
  </div>
  $progressHtml

  <div class="panel">
    <div class="panel-h"><span><i class="fas fa-stream"></i> &nbsp;Log stream &mdash; from $($logFile.Name)</span><span class="text-secondary mono" style="font-size:11px;font-weight:400">$($logEvents.Count) lines &middot; $errN error &middot; $warnN warn</span></div>
    <div class="log-console">$logRows</div>
  </div>

  <div class="cols">
    <div class="panel">
      <div class="panel-h"><span><i class="fas fa-circle-xmark" style="color:var(--c-red)"></i> &nbsp;Failed videos (silent drops)</span></div>
      <div class="vhead"><div class="c-name">Video</div><div class="c-time">When</div><div class="c-detect"></div><div class="c-status">Status</div></div>
      $failRows
    </div>
    <div class="panel">
      <div class="panel-h"><span><i class="fas fa-gauge-high"></i> &nbsp;Slowest videos</span></div>
      <div class="vhead"><div class="c-name">Video</div><div class="c-time">Time</div><div class="c-detect">rel.</div><div class="c-status">Status</div></div>
      $slowRows
    </div>
  </div>

  <div class="cols">
    <div class="panel">
      <div class="panel-h"><span><i class="fas fa-folder-tree"></i> &nbsp;By folder</span><span class="text-secondary mono" style="font-size:11px;font-weight:400">discovered &middot; npz &middot; %</span></div>
      <div class="vhead"><div class="c-name">Top-level folder</div><div class="c-num">Disc.</div><div class="c-num">npz</div><div class="c-detect">Progress</div></div>
      $folderRows
    </div>
    <div class="panel">
      <div class="panel-h"><span><i class="fas fa-layer-group"></i> &nbsp;By corpus</span><span class="text-secondary mono" style="font-size:11px;font-weight:400">filename prefix</span></div>
      <div class="vhead"><div class="c-name">Corpus</div><div class="c-num">Disc.</div><div class="c-num">npz</div><div class="c-detect">Progress</div></div>
      $corpusRows
    </div>
  </div>

  <div class="panel">
    <div class="panel-h"><span><i class="fas fa-file-code"></i> &nbsp;config.yml (run snapshot)</span></div>
    <div class="code">$(Enc $config)</div>
  </div>
</div></body></html>
"@

$outPath = Join-Path $PSScriptRoot 'run-report.html'
$html | Set-Content -Path $outPath -Encoding UTF8
Write-Host "Wrote $outPath" -ForegroundColor Green
Write-Host ("Discovered={0}  attempted(poses)={1}  npz={2}  dropped={3}  errors={4}  timed={5}" -f $discovered,$posesN,$npzN,($posesN-$npzN),$errN,$timed)
