$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$archive = Join-Path $PSScriptRoot 'data/raw/genres.tar.gz'
$expectedHash = 'b28cc067ff6199bd826f5d1a6931458586d64acae7f44b2e882b7a97af057531'
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create .venv and install requirements.txt first.'
}
New-Item -ItemType Directory -Force data/raw | Out-Null
if (-not (Test-Path -LiteralPath $archive)) {
    $partial = $archive + '.part'
    curl.exe -L --fail --retry 2 -C - -o $partial 'https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz?download=true'
    if ($LASTEXITCODE -ne 0) { throw 'Download failed. Run this script again to resume.' }
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $expectedHash) {
        throw 'Archive checksum mismatch. Inspect the partial download before retrying.'
    }
    Move-Item -LiteralPath $partial -Destination $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedHash) {
    throw 'Archive checksum mismatch.'
}
& $python -c "import tarfile; archive=tarfile.open('data/raw/genres.tar.gz'); archive.extractall('data/raw', filter='data')"
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed.' }
& $python -u -m audio.batch data/raw/genres --output-folder data/processed
if ($LASTEXITCODE -ne 0) { throw 'Batch extraction failed. Inspect the audit if present.' }
