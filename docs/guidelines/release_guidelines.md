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

**Important workflow:** 
1. You create and push a tag from the current HEAD
2. The workflow automatically updates version numbers in source files
3. The workflow commits these changes and moves the tag to point to the new commit
4. The tag ends up pointing to a commit that has the correct version numbers

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

**What happens next:**
1. Workflow triggers on the tag push
2. Workflow creates the GitHub release
3. Workflow updates version numbers in CMakeLists.txt, CITATION.cff, and README.md
4. Workflow commits these changes to main
5. Workflow moves the tag to point to this new commit (with updated versions)

This ensures the tag always points to a commit where version numbers match the tag.

### 3. Automated Workflow Steps

Once the tag is pushed, the release workflow automatically:

1. **Validates the tag format**: Ensures it follows semantic versioning (e.g., `v1.0.0` or `v1.0.0-beta.1`)
2. **Checks for existing release**: Skips creation if the release already exists
3. **Generates release notes**: Uses GitHub's automatic release notes generation based on merged pull requests since the last release
4. **Creates the GitHub release**: Publishes the release with the generated notes
5. **Updates version numbers**: Updates the following files to match the release version:
   - `CMakeLists.txt` → updated to release version (e.g., v0.3.0)
   - `CITATION.cff` → updated to release version with current date
   - `README.md` → updated to reference the release tag
6. **Commits and moves tag**: Commits the version changes to main and moves the tag to point to this commit

### 4. Verify the Release

After the workflow completes:

1. **Pull the latest changes** from main (the version update commit was pushed automatically)
   ```bash
   git pull origin main
   ```

2. **Verify the tag points to the correct commit**:
   ```bash
   git show vX.Y.Z
   ```
   This should show the commit with the updated version numbers.

3. **Check the GitHub release page** to ensure the release was created successfully

### 5. Review and Edit Release Notes (Optional)

1. Go to the [releases page](https://github.com/kamping-site/kamping/releases)
2. Review the automatically generated release notes
3. Edit the release notes if needed to add additional context, highlights, or breaking changes
4. You can use the [release template](.github/release-template.md) as a guide for structuring release notes

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

## Version Number Locations and Workflow

Version numbers are maintained in the following files:

1. **CMakeLists.txt**: `VERSION X.Y.Z` in the `project()` command
2. **CITATION.cff**: `version: X.Y.Z` and `date-released: YYYY-MM-DD`
3. **README.md**: `GIT_TAG vX.Y.Z` in the FetchContent example

### Version Number Workflow

The release process follows this pattern:

1. **Before release**: Version numbers in source files reflect the previous release (e.g., v0.2.0)
2. **Create tag**: Tag is initially created pointing to the current HEAD (e.g., `git tag v0.3.0`)
3. **Workflow triggers**: The release workflow is triggered by the tag push
4. **Release created**: GitHub release is created
5. **Versions updated**: Workflow updates version numbers to v0.3.0 in all files
6. **Tag moved**: Workflow commits the changes and moves the tag to point to this new commit
7. **Result**: The tag v0.3.0 now points to a commit where all version numbers are v0.3.0

**Visual representation:**
```
Before:
  main: ... → commit A (v0.2.0 in files) → commit B → commit C
                                                        ↑
                                                    v0.3.0 tag

After workflow:
  main: ... → commit A → commit B → commit C → commit D (v0.3.0 in files)
                                                ↑
                                            v0.3.0 tag (moved)
  
  Commit D message: "Prepare v0.3.0 release"
```

This ensures that:
- The tag always points to code with consistent version numbers
- Users can clone at any tag and get the correct version
- Version numbers in source files always match the git tag
- The process is fully automated after pushing the tag

**Note:** If version numbers are already correct (matching the tag), the workflow will skip the update step. This can happen if you manually updated versions before tagging.

## Troubleshooting

### Release workflow fails

- **Check the Actions tab** for detailed error messages
- **Ensure the tag format is correct** (must match `v*.*.*`)
- **Verify you have the necessary permissions** to create releases
- **Check if the version numbers were already at the release version** (workflow will skip updates if no changes needed)

### Release already exists

If you push a tag for an existing release, the workflow will detect it and skip creation. To create a new release:

1. Delete the existing release in GitHub
2. Delete the tag locally and remotely:
   ```bash
   git tag -d vX.Y.Z
   git push --delete origin vX.Y.Z
   ```
3. Create and push the tag again

### Tag not moved / Version not updated

If the workflow completes but the tag wasn't moved:

- Check the workflow logs for the "Update version numbers and move tag" step
- The version numbers might have already been correct (no update needed)
- Ensure the workflow has permission to force-push tags
- You can manually verify: `git show vX.Y.Z` should show the version update commit

### Merge conflicts after release

If someone pushes to main while the release workflow is running:

1. The workflow might fail to push the version update commit
2. You'll need to manually update version numbers and push
3. Then manually move the tag:
   ```bash
   git pull origin main
   # Update version numbers manually if needed
   git add CMakeLists.txt CITATION.cff README.md
   git commit -m "Prepare vX.Y.Z release"
   git tag -d vX.Y.Z
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin main
   git push origin vX.Y.Z --force
   ```

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
# The workflow will:
#   - Create the GitHub release
#   - Update version numbers in source files
#   - Commit the changes to main
#   - Move the tag to point to the version update commit

# 5. Pull the updated main branch
git pull origin main

# 6. Verify the tag points to the version update commit
git show v0.3.0
# Should show commit message "Prepare v0.3.0 release"

# 7. Review and optionally edit the release notes
# View at: https://github.com/kamping-site/kamping/releases/tag/v0.3.0
```

## Additional Notes

- The release workflow requires the `contents: write` and `pull-requests: write` permissions
- Release notes are generated from the commit history between the current and previous tags
- The workflow uses the GitHub CLI (`gh`) and API for most operations
- All commits are made by `github-actions[bot]`
