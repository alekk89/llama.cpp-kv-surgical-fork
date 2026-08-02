param(
    [string]$Upstream = "upstream/master",
    [string]$Branch = "HEAD",
    [switch]$Fetch
)

$ErrorActionPreference = "Stop"

if ($Fetch) {
    git fetch upstream
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

git rev-parse --verify $Branch 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Unknown branch or commit: $Branch"
}
git rev-parse --verify $Upstream 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Unknown upstream ref: $Upstream. Run with -Fetch or fetch it manually."
}

$status = git status --porcelain
if ($status) {
    Write-Warning "The working tree is dirty. This audit checks committed revisions only."
}

$base = git merge-base $Branch $Upstream
$forkFiles = @(git diff --name-only "$base..$Branch")
$upstreamFiles = @(git diff --name-only "$base..$Upstream")
$upstreamSet = [System.Collections.Generic.HashSet[string]]::new(
    [string[]]$upstreamFiles,
    [System.StringComparer]::Ordinal
)
$overlap = @($forkFiles | Where-Object { $upstreamSet.Contains($_) } | Sort-Object -Unique)

Write-Output "Merge base: $base"
Write-Output "Fork files changed: $($forkFiles.Count)"
Write-Output "Upstream files changed: $($upstreamFiles.Count)"
Write-Output "Overlapping files: $($overlap.Count)"
$overlap | ForEach-Object { Write-Output "  $_" }

$mergeOutput = git merge-tree --write-tree --messages $Branch $Upstream 2>&1
$mergeExit = $LASTEXITCODE
$mergeOutput | ForEach-Object { Write-Output $_ }
if ($mergeExit -ne 0) {
    $conflicts = @(
        @($mergeOutput | ForEach-Object {
            $line = [string]$_
            if ($line -match '^[0-9]{6} [0-9a-f]+ [123]\t(.+)$') {
                $Matches[1]
            }
        }) | Sort-Object -Unique
    )
    if ($conflicts.Count -eq 1 -and $conflicts[0] -eq "README.md") {
        Write-Output "Only README.md conflicts. This is expected because the fork intentionally owns its root README."
    } else {
        Write-Error "The committed fork does not merge cleanly with $Upstream."
        exit $mergeExit
    }
}

Write-Output "Committed code merge audit passed. Build and run the managed-slot tests on the merged result before updating the fork."
exit 0
