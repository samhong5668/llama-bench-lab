# Alternating N-way comparison on Windows - the counterpart of scripts/linux/bench3.sh, and the
# script that reproduces the batches in docs/windows/RESULTS.md.
#
# Every binary is warmed once with the measurement arguments before any number is taken, then
# the binaries are alternated round by round. Both matter: a binary's first run pays to fault in
# the mmap'd CPU-side weights, and whichever binary runs last in a block is penalised.
#
#   scripts/windows/bench.ps1 -Model <gguf> -Binaries a\llama-bench.exe,b\llama-bench.exe
#
# Defaults measure the deployment workload (experts on the CPU). To compare CPU backends
# instead, which is far less noisy, use a small model with -p 0 -n 64 -ngl 0: that benchmarks
# token generation on the CPU. (-ngl 0 alone still lets the GPU take prompt processing; pass
# --device none to rule the GPU out entirely.)
#
#   ... -Model qwen2.5-0.5b-instruct-q4_k_m.gguf -BenchArgs "-p 0 -n 64 -ngl 0 -t 6 -r 3"
#
# -ExtraPath prepends directories to PATH for the child process, which is how the run against
# PyTorch's bundled cuBLAS is done (point it at .venv\Lib\site-packages\torch\lib).

[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)][string]   $Model,
    [Parameter(Mandatory = $true)][string[]] $Binaries,
    [string[]] $Labels,
    [int]      $Rounds    = 4,
    [string]   $BenchArgs = "-p 0 -n 128 -ngl 99 -ncmoe 10 -t 10 -r 10",
    [string[]] $ExtraPath,
    [switch]   $NoToolkit          # clear CUDA_PATH/CUDA_HOME and drop the default PATH
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Model)) { throw "model not found: $Model" }
$Model = (Resolve-Path $Model).Path

$Binaries = $Binaries | ForEach-Object {
    if (-not (Test-Path $_)) { throw "binary not found: $_" }
    (Resolve-Path $_).Path
}
if (-not $Labels) { $Labels = $Binaries | ForEach-Object { Split-Path (Split-Path $_ -Parent) -Leaf } }
if ($Labels.Count -ne $Binaries.Count) { throw "-Labels has $($Labels.Count) entries for $($Binaries.Count) binaries" }

# Run a binary and return "avg +/- stddev" per result row. -o json is parsed rather than the md
# table so this does not depend on the model name or the column layout. ProcessStartInfo is used
# instead of the call operator because PowerShell 5.1 turns a native command's stderr into
# NativeCommandError records, which $ErrorActionPreference = "Stop" would make fatal.
function Invoke-Bench {
    param([string] $Exe, [string] $Arguments)

    # llama.app ships a single multi-tool `llama.exe` that takes a `bench` subcommand; the
    # llama-bench.exe from a release or a local build is the benchmark itself.
    $sub = if ((Split-Path $Exe -Leaf) -ieq "llama.exe") { "bench " } else { "" }

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $Exe
    $psi.Arguments              = "$sub-m `"$Model`" $Arguments -o json"
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true

    $base = if ($NoToolkit) { "$env:SystemRoot\system32;$env:SystemRoot" } else { $env:PATH }
    $psi.EnvironmentVariables["PATH"] = (@($ExtraPath) + @($base) | Where-Object { $_ }) -join ";"
    $psi.EnvironmentVariables["CUDA_VISIBLE_DEVICES"] = "0"
    if ($NoToolkit) {
        $psi.EnvironmentVariables.Remove("CUDA_PATH") | Out-Null
        $psi.EnvironmentVariables.Remove("CUDA_HOME") | Out-Null
    }

    $p   = [System.Diagnostics.Process]::Start($psi)
    $out = $p.StandardOutput.ReadToEnd()
    $err = $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    if ($p.ExitCode -ne 0) { throw "$Exe exited $($p.ExitCode)`n$err" }

    $avg    = [regex]::Matches($out, '"avg_ts":\s*([0-9.]+)')    | ForEach-Object { [double] $_.Groups[1].Value }
    $stddev = [regex]::Matches($out, '"stddev_ts":\s*([0-9.]+)') | ForEach-Object { [double] $_.Groups[1].Value }
    if (-not $avg) { throw "no avg_ts in output of $Exe`n$out" }
    0..($avg.Count - 1) | ForEach-Object { "{0:N2} +/- {1:N2}" -f $avg[$_], $stddev[$_] }
}

Write-Host "model: $Model"
Write-Host "args:  $BenchArgs"
foreach ($i in 0..($Binaries.Count - 1)) { Write-Host ("  {0,-12} {1}" -f $Labels[$i], $Binaries[$i]) }
Write-Host ""
Write-Host "warming each binary once, with the same arguments used for measurement"
foreach ($b in $Binaries) { Invoke-Bench -Exe $b -Arguments $BenchArgs | Out-Null }
Write-Host ""

$results = @{}
foreach ($l in $Labels) { $results[$l] = @() }

foreach ($r in 1..$Rounds) {
    $line = "round$r"
    foreach ($i in 0..($Binaries.Count - 1)) {
        $v = @(Invoke-Bench -Exe $Binaries[$i] -Arguments $BenchArgs)
        $results[$Labels[$i]] += [double] ($v[0] -split ' ')[0]
        $line += "   {0} {1}" -f $Labels[$i], ($v -join " / ").PadRight(18)
    }
    Write-Host $line
}

Write-Host ""
foreach ($l in $Labels) {
    $m = $results[$l] | Measure-Object -Average -Minimum -Maximum
    "{0,-12} mean {1,7:N2}   spread {2,5:N1}%" -f $l, $m.Average, (100 * ($m.Maximum - $m.Minimum) / $m.Average)
}
