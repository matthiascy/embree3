$env:WORK_DIR=(get-location)
$env:EMBREE_DIR="${env:WORK_DIR}\embree-${env:EMBREE_VERSION}.x64.vc14.windows\"

# embree3.dll (and the bundled TBB DLLs) live in the embree `bin/` directory;
# put it on PATH so the test executables can load them at runtime (otherwise
# they fail with STATUS_DLL_NOT_FOUND / exit code 0xc0000135).
$env:PATH="${env:EMBREE_DIR}bin;${env:PATH}"

Write-Output "Building embree-rs"
cargo build
if (!$?) {
    exit 1
}

Write-Output "Running embree-rs Tests"
cargo test
if (!$?) {
    exit 1
}

# build the examples
cd examples
Get-ChildItem .\ -Directory | Where-Object { $_.Name -ne "todos" } | ForEach-Object {
	Write-Output $_
	cd $_
	cargo build
	if (!$?) {
		exit 1
	}
	cd ..
}

