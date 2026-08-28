# Why does the same configuration measure differently between batches? Repeat one identical
# measurement N times back to back, sampling GPU and CPU state around each run, so the
# throughput can be correlated against clocks, temperature, power and free memory.
#
# Candidates this is meant to separate:
#   - GPU boost clock sagging as the card heats up
#   - CPU frequency / hybrid-core scheduling varying per process launch
#   - memory pressure evicting the mmap'd CPU-side weights

[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string] $Model,
    [Parameter(Mandatory = $true)]
    [string] $Bench,                 # path to llama-bench.exe
    [int] $Rounds  = 8,
    [int] $Reps    = 10,
    [int] $NGen    = 128,
    [int] $NCpuMoe = 10,
    [int] $Threads = 10,
    [string] $Csv
)

$ErrorActionPreference = "Stop"
$repoRoot = if ($PSScriptRoot) { Split-Path (Split-Path $PSScriptRoot -Parent) -Parent } else { (Get-Location).Path }
if (-not $Csv) { $Csv = Join-Path $repoRoot "results\drift.csv" }
if (-not (Test-Path $Bench)) { throw "not found: $Bench" }

$env:CUDA_VISIBLE_DEVICES = "0"
$benchArgs = "-p 0 -n $NGen -ngl 99 -ncmoe $NCpuMoe -t $Threads -r $Reps -o json"

function Get-GpuState {
    $q = "temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu"
    $v = (nvidia-smi --query-gpu=$q --format=csv,noheader,nounits) -split ',' | ForEach-Object { $_.Trim() }
    [pscustomobject]@{ TempC = $v[0]; SmMHz = $v[1]; MemMHz = $v[2]; PowerW = $v[3]; UtilPct = $v[4] }
}

function Get-CpuMHz {
    # CurrentClockSpeed is the package-wide current frequency Windows reports.
    (Get-CimInstance Win32_Processor).CurrentClockSpeed
}

function Get-FreeGB {
    [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
}

$rows = foreach ($i in 1..$Rounds) {
    $before = Get-GpuState
    $cpuBefore = Get-CpuMHz
    $freeBefore = Get-FreeGB

    $json = cmd /c ('"{0}" -m "{1}" {2} 2>nul' -f $Bench, $Model, $benchArgs)
    $m = $json | Select-String -Pattern '"avg_ts":\s*([0-9.]+)' | Select-Object -First 1
    $ts = if ($m) { [double] $m.Matches.Groups[1].Value } else { $null }

    $after = Get-GpuState

    $row = [pscustomobject]@{
        Round       = $i
        TS          = if ($ts) { [math]::Round($ts, 2) } else { "FAIL" }
        GpuTempPre  = $before.TempC
        GpuTempPost = $after.TempC
        GpuSmPre    = $before.SmMHz
        GpuSmPost   = $after.SmMHz
        GpuPowPost  = $after.PowerW
        CpuMHzPre   = $cpuBefore
        FreeGBPre   = $freeBefore
    }
    # Write-Host so the progress line does not land in $rows alongside the objects.
    Write-Host ("  round {0,2}  ts {1,6}  gpu {2}C->{3}C  sm {4}->{5} MHz  {6} W  free {7} GB" -f `
        $i, $row.TS, $row.GpuTempPre, $row.GpuTempPost, $row.GpuSmPre, $row.GpuSmPost,
        $row.GpuPowPost, $row.FreeGBPre)
    $row
}

$rows | Export-Csv -Path $Csv -NoTypeInformation -Encoding utf8
""
"wrote $Csv"
""
$rows | Format-Table -AutoSize

$ok = $rows | Where-Object { $_.TS -ne "FAIL" }
if ($ok.Count -ge 2) {
    $ts = $ok | ForEach-Object { [double] $_.TS }
    $spread = ($ts | Measure-Object -Maximum).Maximum - ($ts | Measure-Object -Minimum).Minimum
    $mean = ($ts | Measure-Object -Average).Average
    "t/s  min {0:N2}  max {1:N2}  mean {2:N2}  spread {3:N1}%" -f `
        ($ts | Measure-Object -Minimum).Minimum, ($ts | Measure-Object -Maximum).Maximum, $mean, (100 * $spread / $mean)
}
