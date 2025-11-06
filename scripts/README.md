# KaMPIng Scripts

This directory contains various utility scripts for the KaMPIng project.

## Release Management

### prepare_release.sh

Prepares all files for a new release by updating version numbers in source files.

**Usage:**
```bash
./scripts/prepare_release.sh <version>
```

**Example:**
```bash
./scripts/prepare_release.sh 0.3.0
```

This script will:
1. Verify you're on the `main` branch with a clean working directory
2. Create a new branch for the release preparation
3. Update version numbers in:
   - `CMakeLists.txt`
   - `CITATION.cff` (including release date)
   - `README.md`
4. Commit the changes
5. Provide instructions for creating a PR and pushing the release tag

See [Release Guidelines](../docs/guidelines/release_guidelines.md) for the complete release process.

## Code Formatting

### get_clang_format.sh

Downloads the appropriate version of clang-format for code formatting.

**Usage:**
```bash
./scripts/get_clang_format.sh
```

### get_cmake_format.sh

Downloads cmake-format for CMake file formatting.

**Usage:**
```bash
./scripts/get_cmake_format.sh
```

### run_clang_format.sh

Runs clang-format on all C++ source files in the project.

**Usage:**
```bash
./scripts/run_clang_format.sh
```

## Notes

- All scripts should be run from the repository root directory
- Scripts with `.sh` extension are executable bash scripts
- Make sure scripts are executable: `chmod +x scripts/*.sh`
