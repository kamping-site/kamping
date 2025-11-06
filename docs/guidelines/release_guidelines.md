Release Guidelines for KaMPIng {#release_guidelines}
===============================================

This document describes the automated release process for the KaMPIng project.

## Overview

KaMPIng uses an automated release workflow that creates GitHub releases when version tags are pushed to the repository. The workflow handles:

- Automatic generation of release notes from merged pull requests
- Creation of GitHub releases
- Optional creation of a PR to update version numbers in source files

## Release Process

### 1. Prepare for Release

Before creating a release:

1. **Ensure all desired features/fixes are merged** into the `main` branch
2. **Review and test** the current state of `main`
3. **Decide on the version number** following [semantic versioning](https://semver.org/):
   - **Major version (X.0.0)**: Breaking changes, incompatible API changes
   - **Minor version (0.X.0)**: New features, backwards-compatible changes
   - **Patch version (0.0.X)**: Bug fixes, backwards-compatible fixes

### 2. Create and Push a Version Tag

The automated release workflow is triggered when you push a tag matching the pattern `v*.*.*` (e.g., `v0.3.0`).

```bash
# Ensure you're on the latest main branch
git checkout main
git pull origin main

# Create a version tag (replace X.Y.Z with your version)
git tag -a vX.Y.Z -m "Release vX.Y.Z"

# Push the tag to trigger the release workflow
git push origin vX.Y.Z
```

**Example:**
```bash
git tag -a v0.3.0 -m "Release v0.3.0"
git push origin v0.3.0
```

### 3. Automated Workflow Steps

Once the tag is pushed, the release workflow automatically:

1. **Validates the tag format**: Ensures it follows semantic versioning (e.g., `v1.0.0` or `v1.0.0-beta.1`)
2. **Checks for existing release**: Skips creation if the release already exists
3. **Generates release notes**: Uses GitHub's automatic release notes generation based on merged pull requests since the last release
4. **Creates the GitHub release**: Publishes the release with the generated notes
5. **Creates a version update PR**: Opens a pull request to update version numbers in:
   - `CMakeLists.txt`
   - `CITATION.cff`
   - `README.md`

### 4. Review and Merge Version Update PR

After the release is created, review the automatically generated PR that updates version numbers:

1. **Review the changes** in the PR
2. **Ensure all version numbers are correct**
3. **Merge the PR** to keep version numbers synchronized

### 5. Verify the Release

1. Go to the [releases page](https://github.com/kamping-site/kamping/releases)
2. Verify the release notes are accurate
3. Edit the release notes if needed to add additional context or highlights

## Manual Release Creation (Alternative)

If you need to create a release manually or re-run the workflow:

1. Go to the [Actions tab](https://github.com/kamping-site/kamping/actions)
2. Select the "Create Release" workflow
3. Click "Run workflow"
4. Enter the tag name (e.g., `v0.3.0`)
5. Click "Run workflow"

## Pre-release Versions

For beta, alpha, or release candidate versions, use the standard semantic versioning pre-release syntax:

```bash
git tag -a v0.3.0-beta.1 -m "Release v0.3.0-beta.1"
git push origin v0.3.0-beta.1
```

You can then mark the release as a pre-release in the GitHub UI after it's created.

## Version Number Locations

Version numbers are maintained in the following files:

1. **CMakeLists.txt**: `VERSION X.Y.Z` in the `project()` command
2. **CITATION.cff**: `version: X.Y.Z` and `date-released: YYYY-MM-DD`
3. **README.md**: `GIT_TAG vX.Y.Z` in the FetchContent example

The automated workflow updates these files after creating a release.

## Troubleshooting

### Release workflow fails

- **Check the Actions tab** for detailed error messages
- **Ensure the tag format is correct** (must match `v*.*.*`)
- **Verify you have the necessary permissions** to create releases

### Release already exists

If you push a tag for an existing release, the workflow will detect it and skip creation. To create a new release:

1. Delete the existing release in GitHub
2. Delete the tag locally and remotely:
   ```bash
   git tag -d vX.Y.Z
   git push --delete origin vX.Y.Z
   ```
3. Create and push the tag again

### Version update PR not created

The version update PR is only created for tag pushes (not manual workflow runs). If it's not created:

- Check the workflow logs for errors
- The PR might already exist
- There might be no version changes needed

## Best Practices

1. **Use clear commit messages** - They appear in the automated release notes
2. **Label pull requests appropriately** - This helps organize release notes
3. **Review release notes** - Edit them after creation if needed to add highlights or context
4. **Test before tagging** - Ensure the `main` branch is in a good state
5. **Follow semantic versioning** - This helps users understand the impact of updates
6. **Coordinate with team** - Ensure everyone knows a release is being created

## Example Release Workflow

Here's a complete example of creating version 0.3.0:

```bash
# 1. Update local repository
git checkout main
git pull origin main

# 2. Verify everything is working
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build

# 3. Create and push the tag
git tag -a v0.3.0 -m "Release v0.3.0"
git push origin v0.3.0

# 4. Wait for the automated workflow to complete
# Monitor at: https://github.com/kamping-site/kamping/actions

# 5. Review and merge the version update PR
# Find it at: https://github.com/kamping-site/kamping/pulls

# 6. Review the release
# View at: https://github.com/kamping-site/kamping/releases/tag/v0.3.0
```

## Additional Notes

- The release workflow requires the `contents: write` and `pull-requests: write` permissions
- Release notes are generated from the commit history between the current and previous tags
- The workflow uses the GitHub CLI (`gh`) and API for most operations
- All commits are made by `github-actions[bot]`
