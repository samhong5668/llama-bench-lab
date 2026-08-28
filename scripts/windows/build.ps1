# Build llama-bench from llama.cpp source in one of the configurations this investigation
# compared. Clones the source if it is not already there.
#
#   native          GGML_NATIVE=ON - tuned for THIS machine's CPU. Fast, but not
#                   distributable: it will fault on a host without the same ISA.
#   replica         llama-install.sh's forced settings. Reproduces the llama.app prebuilt.
#   replica-shared  as replica, but shared linking - isolates the linking factor.
#   noomp           GGML_NATIVE=OFF (which leaves the ISA options ON) with OpenMP off -
#                   isolates the OpenMP factor.
#
# The MSVC environment is imported from vcvars64.bat, located through vswhere.

[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("native", "replica", "replica-shared", "noomp")]
    [string] $Config,

    [string] $Tag      = "b10107",
    [string] $CudaArch = "120",
    [string] $SrcDir,
    [string] $BuildDir,
    # The published numbers were produced with MSVC 19.44 (VS 2022 Build Tools). vswhere
    # -latest may resolve to a newer toolset; pass -VsPath to pin it if that matters.
    [string] $VsPath,
    [switch] $ConfigureOnly
)

$ErrorActionPreference = "Stop"

# The repo root is two levels up from scripts/windows/. $PSScriptRoot is empty under some
# invocation modes, so fall back to the working directory rather than using param defaults.
$repoRoot = if ($PSScriptRoot) { Split-Path (Split-Path $PSScriptRoot -Parent) -Parent } else { (Get-Location).Path }
if (-not $SrcDir)   { $SrcDir   = Join-Path $repoRoot "src" }
if (-not $BuildDir) { $BuildDir = Join-Path $repoRoot "build-$Config" }

# --- source ------------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $SrcDir "CMakeLists.txt"))) {
    "cloning llama.cpp $Tag ..."
    git clone --depth 1 --branch $Tag https://github.com/ggml-org/llama.cpp.git $SrcDir
}
$SrcDir = (Resolve-Path $SrcDir).Path
"source: $SrcDir @ $(git -C $SrcDir rev-parse HEAD)"

# --- MSVC environment --------------------------------------------------------------------
if ($VsPath) {
    $vsRoot = $VsPath
} else {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "vswhere not found - install Visual Studio or the Build Tools" }
    $vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath |
              Select-Object -First 1
}
if (-not $vsRoot) { throw "no Visual Studio installation with the C++ toolset was found" }
$vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found under $vsRoot" }

# cmd performs the `call`; we import the resulting environment into this session.
cmd /c ('call "{0}" >nul 2>nul && set' -f $vcvars) | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}
"toolset: $(cmd /c 'where cl' | Select-Object -First 1)"

# Ninja ships with Visual Studio; prefer it over generating an MSBuild project.
$ninja = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if (Test-Path $ninja) { $env:PATH = "$ninja;$env:PATH" }

# --- configuration -----------------------------------------------------------------------
$common = @(
    "-DCMAKE_BUILD_TYPE=Release"
    "-DGGML_CUDA=ON"
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
    "-DLLAMA_BUILD_TESTS=OFF"
    "-DLLAMA_BUILD_EXAMPLES=OFF"
    "-DLLAMA_BUILD_SERVER=OFF"
    "-DLLAMA_BUILD_APP=OFF"
    "-DLLAMA_BUILD_UI=OFF"
    "-DLLAMA_CURL=OFF"
)

# llama-install.sh's CMakeLists.txt forces these for every preset. The ISA options have to
# be listed explicitly: GGML_NATIVE=OFF alone leaves ggml's INS_ENB enabling them.
$installShForced = @(
    "-DGGML_NATIVE=OFF"
    "-DGGML_OPENMP=OFF"
    "-DGGML_LTO=OFF"
    "-DGGML_CCACHE=OFF"
    "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF"
    "-DGGML_SSE42=OFF"
    "-DGGML_AVX=OFF"
    "-DGGML_AVX2=OFF"
    "-DGGML_FMA=OFF"
    "-DGGML_F16C=OFF"
    "-DGGML_BMI2=OFF"
)

$specific = switch ($Config) {
    "native"         { @() }                                     # GGML_NATIVE defaults to ON
    "replica"        { $installShForced + @("-DBUILD_SHARED_LIBS=OFF", "-DGGML_STATIC=ON") }
    "replica-shared" { $installShForced + @("-DBUILD_SHARED_LIBS=ON",  "-DGGML_STATIC=OFF") }
    "noomp"          { @("-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF") }
}

"config: $Config"

# cmake and ninja write progress to stderr. Under PowerShell 5.1, if the caller redirects
# stderr into the pipeline (`... 2>&1 > log.txt`, or a `2>&1` on the invocation), each such
# line arrives as a NativeCommandError, which $ErrorActionPreference = "Stop" turns into a
# terminating error even on a successful build. Exit codes are checked explicitly instead.
function Invoke-Native {
    # $Args would shadow PowerShell's automatic variable, hence $Command.
    param(
        [Parameter(Mandatory = $true)][string] $What,
        [Parameter(Mandatory = $true)][string] $Exe,
        [string[]] $Command = @()
    )
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try     { & $Exe @Command }
    finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE) { throw "$What failed ($LASTEXITCODE)" }
}

Invoke-Native -What "cmake configure" -Exe "cmake" `
    -Command (@("-S", $SrcDir, "-B", $BuildDir, "-G", "Ninja") + $common + $specific)
if ($ConfigureOnly) { "configure only - stopping here"; return }

Invoke-Native -What "build" -Exe "cmake" `
    -Command @("--build", $BuildDir, "--target", "llama-bench", "llama-completion", "--parallel")

Get-ChildItem (Join-Path $BuildDir "bin") -Filter "llama-*.exe" |
    Select-Object Name, @{n = 'KB'; e = { [math]::Round($_.Length / 1KB) } } |
    Format-Table -AutoSize
