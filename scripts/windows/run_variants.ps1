# Does llama.cpp's runtime CPU-variant dispatch pick the fastest variant available on this
# CPU? Force one variant at a time by leaving only its DLL in place. Same binary throughout,
# so only the CPU backend changes. Warm-up before each measured pass, two passes each.
#
# Expects the official llama.cpp release (cpu + cuda zips) extracted into -BinDir.
#
# Native commands are invoked through cmd /c so that cmd performs the redirection. Windows
# PowerShell 5.1 turns a native command's redirected stderr into ErrorRecords, which becomes
# fatal under ErrorActionPreference = Stop even when the exe exits 0.

[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string] $Model,
    [string] $BinDir,
    [string] $OutDir,
    [string[]] $Variants = @("AUTO", "alderlake", "haswell", "piledriver", "sandybridge", "x64"),
    [int] $Reps = 12,
    [int] $NGen = 128,
    [int] $NCpuMoe = 10,
    [int] $Threads = 10
)

$ErrorActionPreference = "Stop"

# The repo root is two levels up from scripts/windows/. $PSScriptRoot is empty under some
# invocation modes, so fall back to the working directory rather than using param defaults.
$repoRoot = if ($PSScriptRoot) { Split-Path (Split-Path $PSScriptRoot -Parent) -Parent } else { (Get-Location).Path }
if (-not $BinDir) { $BinDir = Join-Path $repoRoot "official-b10107\bin" }
if (-not $OutDir) { $OutDir = Join-Path $repoRoot "results\variants" }
if (-not (Test-Path $BinDir)) {
    throw "release binaries not found at $BinDir - extract the llama.cpp cpu + cuda zips there, or pass -BinDir"
}
$BinDir = (Resolve-Path $BinDir).Path
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
$stash = Join-Path $BinDir "_all"
New-Item -ItemType Directory -Force -Path $stash | Out-Null

$bench      = Join-Path $BinDir "llama-bench.exe"
$completion = Join-Path $BinDir "llama-completion.exe"
foreach ($exe in @($bench, $completion)) {
    if (-not (Test-Path $exe)) { throw "not found: $exe" }
}

# Stash every variant once, so each run can be given exactly one.
Get-ChildItem $BinDir -Filter "ggml-cpu-*.dll" -File | Move-Item -Destination $stash -Force
$stashed = Get-ChildItem $stash -Filter "ggml-cpu-*.dll" -File
if ($stashed.Count -eq 0) { throw "no ggml-cpu-*.dll found in $BinDir or $stash" }
"stashed $($stashed.Count) variants"

$env:CUDA_VISIBLE_DEVICES = "0"
$benchArgs = "-p 0 -n $NGen -ngl 99 -ncmoe $NCpuMoe -t $Threads -r $Reps"

try {
    foreach ($v in $Variants) {
        Get-ChildItem $BinDir -Filter "ggml-cpu-*.dll" -File | Remove-Item -Force
        if ($v -eq "AUTO") {
            Copy-Item "$stash\*.dll" $BinDir -Force          # all present, ggml chooses
        } else {
            $src = Join-Path $stash "ggml-cpu-$v.dll"
            if (-not (Test-Path $src)) { "=== $v === (not in this release, skipped)"; continue }
            Copy-Item $src $BinDir -Force
        }

        "=== $v ==="

        # system_info goes to stderr; let cmd merge it so PowerShell sees plain text.
        $line = cmd /c ('"{0}" -m "{1}" -p hi -n 1 --no-warmup 2>&1' -f $completion, $Model) |
                Select-String -Pattern "system_info" | Select-Object -First 1
        if ($line) { "  loaded: " + (($line.Line -split "CPU : ")[-1]) }

        cmd /c ('"{0}" -m "{1}" -p 0 -n 16 -ngl 99 -ncmoe {2} -r 1 -o md >nul 2>nul' -f $bench, $Model, $NCpuMoe) | Out-Null
        "  warm ok"

        foreach ($pass in 1, 2) {
            $json = Join-Path $OutDir "${v}_p${pass}.json"
            cmd /c ('"{0}" -m "{1}" {2} -o json 2>nul' -f $bench, $Model, $benchArgs) |
                Set-Content -Path $json -Encoding utf8
            $m = Select-String -Path $json -Pattern '"avg_ts":\s*([0-9.]+)' | Select-Object -First 1
            if ($m) { "  pass{0}  {1}" -f $pass, $m.Matches.Groups[1].Value }
            else    { "  pass{0}  (no result - see {1})" -f $pass, $json }
        }
    }
}
finally {
    # Always put the release back the way it was found.
    Get-ChildItem $BinDir -Filter "ggml-cpu-*.dll" -File | Remove-Item -Force -ErrorAction SilentlyContinue
    Copy-Item "$stash\*.dll" $BinDir -Force
    "restored $((Get-ChildItem $BinDir -Filter 'ggml-cpu-*.dll' -File).Count) variants"
}
