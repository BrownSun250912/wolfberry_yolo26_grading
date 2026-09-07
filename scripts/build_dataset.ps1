[CmdletBinding()]
param(
    [string]$SourceImageJson = (Join-Path $PSScriptRoot '..\..\new'),
    [string]$SourceYoloLabels = (Join-Path $PSScriptRoot '..\..\labels'),
    [string]$ProjectRoot = (Join-Path $PSScriptRoot '..'),
    [int]$Seed = 20260823,
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$SourceImageJson = (Resolve-Path -LiteralPath $SourceImageJson).Path
$SourceYoloLabels = (Resolve-Path -LiteralPath $SourceYoloLabels).Path
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$SplitRoot = Join-Path $ProjectRoot 'data\split_v1'
$ManifestRoot = Join-Path $ProjectRoot 'data\manifests'

$images = @(Get-ChildItem -LiteralPath $SourceImageJson -File -Filter '*.png' | ForEach-Object {
    if ($_.BaseName -notmatch '^(\d+)_Color$') {
        throw "Unexpected image name: $($_.Name)"
    }
    [PSCustomObject]@{
        OriginalNumber = [int]$Matches[1]
        OriginalStem = $_.BaseName
        ImagePath = $_.FullName
    }
} | Sort-Object OriginalNumber)

if ($images.Count -ne 1285) {
    throw "Expected 1285 PNG files, found $($images.Count)."
}

$records = [System.Collections.Generic.List[object]]::new()
$seenClassIds = [System.Collections.Generic.HashSet[int]]::new()

foreach ($item in $images) {
    $jsonPath = Join-Path $SourceImageJson ($item.OriginalStem + '.json')
    $txtPath = Join-Path $SourceYoloLabels ($item.OriginalStem + '.txt')
    if (-not (Test-Path -LiteralPath $jsonPath -PathType Leaf)) { throw "Missing JSON: $jsonPath" }
    if (-not (Test-Path -LiteralPath $txtPath -PathType Leaf)) { throw "Missing TXT: $txtPath" }

    $annotation = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
    # Some legacy JSON files retain a relative prefix such as
    # "..\新建文件夹\901_Color.png". The leaf name is still the correct image.
    if ([System.IO.Path]::GetFileName([string]$annotation.imagePath) -ne ($item.OriginalStem + '.png')) {
        throw "JSON imagePath mismatch: $jsonPath"
    }
    if (@($annotation.shapes).Count -lt 1) { throw "No instances in JSON: $jsonPath" }
    foreach ($shape in @($annotation.shapes)) {
        if ($shape.label -ne 'wolfberry' -or $shape.shape_type -ne 'polygon') {
            throw "Unexpected JSON label/type in $jsonPath"
        }
    }

    $txtLines = @(Get-Content -LiteralPath $txtPath | Where-Object { $_.Trim().Length -gt 0 })
    if ($txtLines.Count -ne @($annotation.shapes).Count) {
        throw "Instance count differs between JSON and TXT: $($item.OriginalStem)"
    }
    foreach ($line in $txtLines) {
        $tokens = @($line.Trim() -split '\s+')
        if ($tokens.Count -lt 7 -or (($tokens.Count - 1) % 2) -ne 0) {
            throw "Invalid YOLO polygon token count: $txtPath"
        }
        $classId = [int]$tokens[0]
        [void]$seenClassIds.Add($classId)
        foreach ($coordinate in $tokens[1..($tokens.Count - 1)]) {
            $value = [double]::Parse($coordinate, [Globalization.CultureInfo]::InvariantCulture)
            if ($value -lt 0.0 -or $value -gt 1.0) { throw "Out-of-range coordinate: $txtPath" }
        }
    }

    $sourceGroup = if ($item.OriginalNumber -le 150) {'001-150'}
        elseif ($item.OriginalNumber -le 500) {'151-500'}
        elseif ($item.OriginalNumber -le 900) {'501-900'}
        else {'901-1300'}
    $count = @($annotation.shapes).Count
    $countBin = if ($count -le 3) {'01-03'} elseif ($count -le 6) {'04-06'} else {'07+'}

    $records.Add([PSCustomObject]@{
        OriginalNumber = $item.OriginalNumber
        OriginalStem = $item.OriginalStem
        ImagePath = $item.ImagePath
        JsonPath = $jsonPath
        TxtPath = $txtPath
        ImageWidth = [int]$annotation.imageWidth
        ImageHeight = [int]$annotation.imageHeight
        InstanceCount = $count
        SourceGroup = $sourceGroup
        CountBin = $countBin
        Stratum = "$sourceGroup|$countBin"
    })
}

if (($seenClassIds | Sort-Object) -join ',' -ne '0') {
    throw "Expected only class ID 0, found: $(($seenClassIds | Sort-Object) -join ',')"
}

# Shuffle and allocate separately inside every stratum. This balances source-number
# ranges and low/medium/high instance-count images across all three fixed splits.
$rng = [System.Random]::new($Seed)
$trainTarget = [int][math]::Round($records.Count * 0.80)
$valTarget = [int][math]::Round($records.Count * 0.10)
$testTarget = $records.Count - $trainTarget - $valTarget

$byOriginal = @($records | Sort-Object OriginalNumber)
$workIdByOriginal = @{}
for ($i = 0; $i -lt $byOriginal.Count; $i++) {
    $workIdByOriginal[$byOriginal[$i].OriginalNumber] = $i + 1
}

$assignments = [System.Collections.Generic.List[object]]::new()
foreach ($group in @($records | Group-Object Stratum | Sort-Object Name)) {
    $shuffled = @($group.Group | Sort-Object { $rng.Next() })
    $groupTrain = [int][math]::Floor($shuffled.Count * 0.80)
    $groupVal = [int][math]::Floor($shuffled.Count * 0.10)
    for ($i = 0; $i -lt $shuffled.Count; $i++) {
        $split = if ($i -lt $groupTrain) {'train'} elseif ($i -lt ($groupTrain + $groupVal)) {'val'} else {'test'}
        $assignments.Add([PSCustomObject]@{ Record = $shuffled[$i]; Split = $split; Tie = $rng.Next() })
    }
}

# Floors leave a small surplus in test. Move deterministic candidates until the
# global counts are exactly 1028/128/129.
$trainNeed = $trainTarget - @($assignments | Where-Object Split -eq 'train').Count
$valNeed = $valTarget - @($assignments | Where-Object Split -eq 'val').Count
$testCandidates = @($assignments | Where-Object Split -eq 'test' | Sort-Object Tie)
if ($trainNeed -lt 0 -or $valNeed -lt 0 -or ($trainNeed + $valNeed) -gt $testCandidates.Count) {
    throw 'Unexpected split quota state while rebalancing strata.'
}
for ($i = 0; $i -lt $trainNeed; $i++) { $testCandidates[$i].Split = 'train' }
for ($i = $trainNeed; $i -lt ($trainNeed + $valNeed); $i++) { $testCandidates[$i].Split = 'val' }

# The 7+ instance bin is rare (10 images). Per-source floors can otherwise leave
# one split without any hard crowded example, so enforce an exact 8/1/1 balance
# by swapping with non-rare records while preserving global split counts.
$rare = @($assignments | Where-Object { $_.Record.CountBin -eq '07+' })
if ($rare.Count -ge 3) {
    $rareTargets = @{
        train = [int][math]::Round($rare.Count * 0.80)
        val = [int][math]::Round($rare.Count * 0.10)
    }
    $rareTargets.test = $rare.Count - $rareTargets.train - $rareTargets.val
    foreach ($donorSplit in @('train','val','test')) {
        while (@($rare | Where-Object Split -eq $donorSplit).Count -gt $rareTargets[$donorSplit]) {
            $receiverSplit = @('train','val','test') | Where-Object {
                @($rare | Where-Object Split -eq $_).Count -lt $rareTargets[$_]
            } | Select-Object -First 1
            if (-not $receiverSplit) { throw 'Could not rebalance rare crowded-image bin.' }
            $rareMove = $rare | Where-Object Split -eq $donorSplit | Sort-Object Tie | Select-Object -First 1
            $nonRareSwap = $assignments | Where-Object {
                $_.Split -eq $receiverSplit -and $_.Record.CountBin -ne '07+' -and
                $_.Record.SourceGroup -eq $rareMove.Record.SourceGroup
            } | Sort-Object Tie | Select-Object -First 1
            if (-not $nonRareSwap) {
                $nonRareSwap = $assignments | Where-Object {
                    $_.Split -eq $receiverSplit -and $_.Record.CountBin -ne '07+'
                } | Sort-Object Tie | Select-Object -First 1
            }
            if (-not $nonRareSwap) { throw 'Could not find a non-rare swap candidate.' }
            $rareMove.Split = $receiverSplit
            $nonRareSwap.Split = $donorSplit
        }
    }
}

$splitByOriginal = @{}
foreach ($assignment in $assignments) {
    $splitByOriginal[$assignment.Record.OriginalNumber] = $assignment.Split
}

$manifest = @($byOriginal | ForEach-Object {
    $newStem = 'wb_{0:D6}' -f $workIdByOriginal[$_.OriginalNumber]
    [PSCustomObject]@{
        work_id = $workIdByOriginal[$_.OriginalNumber]
        new_stem = $newStem
        split = $splitByOriginal[$_.OriginalNumber]
        original_number = $_.OriginalNumber
        original_stem = $_.OriginalStem
        source_group = $_.SourceGroup
        image_width = $_.ImageWidth
        image_height = $_.ImageHeight
        instance_count = $_.InstanceCount
        original_image = $_.ImagePath
        original_json = $_.JsonPath
        original_txt = $_.TxtPath
    }
})

Write-Host "Validated $($manifest.Count) complete PNG/JSON/TXT triplets."
$manifest | Group-Object split | Sort-Object Name | Select-Object Name,Count | Format-Table -AutoSize
Write-Host "Targets: train=$trainTarget val=$valTarget test=$testTarget seed=$Seed"

if (-not $Execute) {
    Write-Host 'Dry run only. Re-run with -Execute to create the split dataset.'
    return
}

if (Test-Path -LiteralPath $SplitRoot) {
    $existingFiles = @(Get-ChildItem -LiteralPath $SplitRoot -File -Recurse -ErrorAction SilentlyContinue)
    if ($existingFiles.Count -gt 0) {
        throw "Refusing to overwrite non-empty dataset: $SplitRoot"
    }
}

foreach ($kind in @('images','labels','labelme_json')) {
    foreach ($split in @('train','val','test')) {
        [void](New-Item -ItemType Directory -Force -Path (Join-Path $SplitRoot "$kind\$split"))
    }
}
[void](New-Item -ItemType Directory -Force -Path $ManifestRoot)

foreach ($row in $manifest) {
    $record = $records | Where-Object OriginalNumber -eq $row.original_number | Select-Object -First 1
    $destImage = Join-Path $SplitRoot "images\$($row.split)\$($row.new_stem).png"
    $destTxt = Join-Path $SplitRoot "labels\$($row.split)\$($row.new_stem).txt"
    $destJson = Join-Path $SplitRoot "labelme_json\$($row.split)\$($row.new_stem).json"
    Copy-Item -LiteralPath $record.ImagePath -Destination $destImage
    Copy-Item -LiteralPath $record.TxtPath -Destination $destTxt
    $annotation = Get-Content -LiteralPath $record.JsonPath -Raw | ConvertFrom-Json
    $annotation.imagePath = "$($row.new_stem).png"
    $annotation | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $destJson -Encoding utf8
}

$manifestPath = Join-Path $ManifestRoot 'dataset_manifest.csv'
$manifest | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding utf8
foreach ($split in @('train','val','test')) {
    $manifest | Where-Object split -eq $split | ForEach-Object new_stem |
        Set-Content -LiteralPath (Join-Path $ManifestRoot "$split.txt") -Encoding utf8
}

$yamlPath = $SplitRoot.Replace('\','/')
$yaml = @"
# Ultralytics YOLO instance-segmentation dataset
path: $yamlPath
train: images/train
val: images/val
test: images/test

names:
  0: wolfberry
"@
$yaml | Set-Content -LiteralPath (Join-Path $SplitRoot 'wolfberry_seg.yaml') -Encoding utf8

foreach ($split in @('train','val','test')) {
    $expected = @($manifest | Where-Object split -eq $split).Count
    foreach ($kind in @('images','labels','labelme_json')) {
        $actual = @(Get-ChildItem -LiteralPath (Join-Path $SplitRoot "$kind\$split") -File).Count
        if ($actual -ne $expected) { throw "Post-build count mismatch: $kind/$split expected=$expected actual=$actual" }
    }
}

Write-Host "Dataset created successfully: $SplitRoot"
Write-Host "Manifest: $manifestPath"
