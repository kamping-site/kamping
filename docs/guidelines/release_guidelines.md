Release Guidelines for KaMPIng {#release_guidelines}
===============================================

This document describes the automated release process for the KaMPIng project.

## Overview

KaMPIng uses an automated release workflow that creates GitHub releases when version tags are pushed to the repository. The workflow handles:

- Automatic generation of release notes from merged pull requests
- Creation of GitHub releases
- Optional creation of a PR to update version numbers in source files

## Release Process

The KaMPIng release process is a **two-step workflow** that ensures releases are reviewed before finalization:

1. **Preparation**: Push a pre-release tag (`v*.*.*-pre`) → Automated PR is created
2. **Finalization**: Review and merge the PR → Release is automatically created

### 1. Prepare for Release

Before creating a release:

1. **Ensure all desired features/fixes are merged** into the `main` branch
2. **Review and test** the current state of `main`
3. **Decide on the version number** following [semantic versioning](https://semver.org/):
   - **Major version (X.0.0)**: Breaking changes, incompatible API changes
   - **Minor version (0.X.0)**: New features, backwards-compatible changes
   - **Patch version (0.0.X)**: Bug fixes, backwards-compatible fixes

### 2. Initiate Release Preparation

You can initiate the release process in two ways:

#### Option A: Push a Pre-Release Tag (Recommended)

Push a **pre-release tag** with the `-pre` suffix:

```bash
# Ensure you're on the latest main branch
git checkout main
git pull origin main

# Create a pre-release tag (replace X.Y.Z with your version)
git tag -a vX.Y.Z-pre -m "Prepare release vX.Y.Z"

# Push the tag to trigger the release preparation workflow
git push origin vX.Y.Z-pre
```

**Example:**
```bash
git tag -a v0.3.0-pre -m "Prepare release v0.3.0"
git push origin v0.3.0-pre
```

#### Option B: Trigger Workflow Manually

Alternatively, trigger the workflow manually without creating a tag:

1. Go to the [Actions tab](https://github.com/kamping-site/kamping/actions/workflows/release.yml)
2. Click "Run workflow"
3. Select the `main` branch
4. Enter the version number (e.g., `0.3.0` without the `v` prefix)
5. Click "Run workflow"

**What happens automatically (both options):**
1. Workflow validates the version format
2. Checks that the release doesn't already exist
3. Creates a new branch `release-v0.3.0` with version updates:
   - Updates `CMakeLists.txt` to version 0.3.0
   - Updates `CITATION.cff` to version 0.3.0 with current date
   - Updates `README.md` to reference v0.3.0
4. Generates preview release notes from merged PRs
5. Creates a Pull Request titled "Release v0.3.0" with all changes and preview notes

### 3. Review the Release PR

After the workflow completes, you'll have a PR to review:

1. **Go to the Pull Requests page** of your repository
2. **Find the "Release v0.3.0" PR** (it will be labeled with `release`)
3. **Review the changes**:
   - Verify version numbers are correct in all files
   - Check the commit message
4. **Review the release notes** in the PR description:
   - The PR includes auto-generated release notes from merged PRs
   - Edit the PR description to refine the release notes if needed
   - Add highlights, breaking changes, or upgrade instructions
5. **Request reviews** from team members if desired

### 4. Finalize the Release

When you're satisfied with the release preparation:

1. **Merge the PR** (use "Squash and merge" or "Create a merge commit")
2. **Automated finalization** happens immediately:
   - Workflow creates the final tag `v0.3.0` at the merge commit
   - GitHub release is created with the release notes from the PR
   - Pre-release tag `v0.3.0-pre` is automatically deleted

### 5. Verify the Release

After the workflow completes:

1. **Pull the latest changes**:
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Verify the tag was created**:
   ```bash
   git tag -l v0.3.0
   git show v0.3.0
   ```

3. **Check the GitHub release page**: Visit https://github.com/kamping-site/kamping/releases/tag/v0.3.0

### 6. Edit Release Notes (Optional)

After the release is published, you can further refine it:

1. Go to the [releases page](https://github.com/kamping-site/kamping/releases)
2. Click "Edit" on the release
3. Refine the release notes if needed
4. You can use the [release template](https://github.com/kamping-site/kamping/blob/main/.github/release-template.md) as a guide for structuring release notes

## Canceling a Release

If you pushed a pre-release tag by mistake or want to cancel the release:

1. **Close the release PR** without merging
2. **Delete the pre-release tag**:
   ```bash
   git tag -d v0.3.0-pre
   git push --delete origin v0.3.0-pre
   ```
3. **Delete the release branch** (optional):
   ```bash
   git push --delete origin release-v0.3.0
   ```

## Creating Pre-release Versions (Beta, RC, etc.)

For beta, alpha, or release candidate versions, the process is currently manual. The automated workflow is designed for stable releases only.

To create a pre-release version manually:

1. Update version numbers in CMakeLists.txt, CITATION.cff, and README.md
2. Commit and push to main
3. Create and push the tag:
   ```bash
   git tag -a v0.3.0-beta.1 -m "Release v0.3.0-beta.1"
   git push origin v0.3.0-beta.1
   ```
4. Create the GitHub release manually and mark it as a pre-release

**Note**: Using the `-pre` suffix (e.g., `v0.3.0-beta.1-pre`) will trigger the automated workflow, but it's designed for stable releases.

## Version Number Locations and Workflow

Version numbers are maintained in the following files:

1. **CMakeLists.txt**: `VERSION X.Y.Z` in the `project()` command
2. **CITATION.cff**: `version: X.Y.Z` and `date-released: YYYY-MM-DD`
3. **README.md**: `GIT_TAG vX.Y.Z` in the FetchContent example

### Version Number Workflow

The release process follows this two-step pattern:

**Step 1: Preparation (Pre-Release Tag)**
```
1. Push v0.3.0-pre tag
   main: ... → commit C (v0.2.0 in files)
                       ↑
                   v0.3.0-pre

2. Workflow creates release-v0.3.0 branch with version updates
   release-v0.3.0: commit C → commit D (v0.3.0 in files)
                              "Prepare v0.3.0 release"

3. PR created: release-v0.3.0 → main
```

**Step 2: Finalization (PR Merge)**
```
4. PR is reviewed and merged
   main: ... → commit C → commit M (merge, v0.3.0 in files)

5. Workflow creates final tag at merge commit
   main: ... → commit C → commit M (v0.3.0 in files)
                          ↑
                      v0.3.0 tag

6. v0.3.0-pre tag is deleted
```

### Key Benefits of This Workflow

✅ **Manual Review**: Release preparation is reviewed before finalization  
✅ **Clear History**: Clean commit showing "Prepare v0.3.0 release"  
✅ **Consistency**: Tag points to commit with matching version numbers  
✅ **Flexibility**: Can edit release notes in PR before publishing  
✅ **Safety**: Can cancel by closing PR without merging  
✅ **Automation**: After approval, release creation is automatic

## Troubleshooting

### PR creation fails

If the workflow fails to create the release PR:

- **Check if a PR already exists**: Look for open PRs with `release-v` branch names
- **Check if the release already exists**: The workflow will fail if the final tag/release exists
- **Verify the tag format**: Must be `vX.Y.Z-pre` (e.g., `v0.3.0-pre`)
- **Check the Actions tab** for detailed error messages

To retry:
```bash
# Delete the pre-release tag
git tag -d v0.3.0-pre
git push --delete origin v0.3.0-pre

# Fix any issues and try again
git tag -a v0.3.0-pre -m "Prepare release v0.3.0"
git push origin v0.3.0-pre
```

### Version numbers were already updated

If you manually updated version numbers before pushing the pre-release tag:

- The workflow will fail because there are no changes to commit
- **Solution**: Revert the version numbers to the previous release, then push the pre-release tag

### Merge conflicts in release PR

If the release PR has merge conflicts (someone pushed to main after PR creation):

1. **Update the release branch**:
   ```bash
   git checkout release-v0.3.0
   git pull origin main
   # Resolve conflicts
   git add .
   git commit -m "Resolve merge conflicts"
   git push origin release-v0.3.0
   ```

2. **Continue with normal PR review and merge**

### Release finalization fails after PR merge

If the finalization workflow fails after merging the PR:

- **Check the Actions tab** for error messages
- **Manually create the tag**:
  ```bash
  git checkout main
  git pull origin main
  git tag -a v0.3.0 -m "Release v0.3.0"
  git push origin v0.3.0
  ```
- **Manually create the release** in GitHub UI if needed

### Accidentally pushed wrong pre-release tag

If you pushed `v0.4.0-pre` but meant `v0.3.0-pre`:

1. **Close the PR** without merging
2. **Delete the wrong tag and branch**:
   ```bash
   git push --delete origin v0.4.0-pre
   git push --delete origin release-v0.4.0
   ```
3. **Push the correct tag**:
   ```bash
   git tag -a v0.3.0-pre -m "Prepare release v0.3.0"
   git push origin v0.3.0-pre
   ```

## Best Practices

1. **Test before tagging** - Ensure the `main` branch is in a good state before pushing the pre-release tag
2. **Use clear commit messages** - They appear in the automated release notes
3. **Label pull requests** - Use labels like `feature`, `bug`, `documentation` to help organize release notes
4. **Review the release PR thoroughly** - Check version numbers and release notes before merging
5. **Edit release notes in the PR** - Refine the auto-generated notes before finalizing
6. **Follow semantic versioning** - This helps users understand the impact of updates
7. **Coordinate with team** - Ensure everyone knows a release is being prepared
8. **Don't push to main during release** - Wait for the release PR to merge to avoid conflicts

## Example Release Workflow

Here's a complete example of creating version 0.3.0:

```bash
# 1. Update local repository
git checkout main
git pull origin main

# 2. Verify everything is working (recommended)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build

# 3a. Option A: Push the pre-release tag
git tag -a v0.3.0-pre -m "Prepare release v0.3.0"
git push origin v0.3.0-pre

# OR

# 3b. Option B: Trigger workflow manually
# Go to https://github.com/kamping-site/kamping/actions/workflows/release.yml
# Click "Run workflow", enter version "0.3.0", and run

# 4. Wait for the PR to be created (usually takes 1-2 minutes)
# Monitor at: https://github.com/kamping-site/kamping/actions

# 5. Review the release PR
# Go to: https://github.com/kamping-site/kamping/pulls
# - Check version numbers in files
# - Review the auto-generated release notes
# - Edit the PR description to refine release notes if needed
# - Request team reviews if desired

# 6. Merge the PR
# Click "Merge pull request" in the GitHub UI
# (or use gh CLI: gh pr merge --squash)

# 7. Wait for final release creation (automatic, takes ~30 seconds)
# The workflow will:
#   - Create the v0.3.0 tag at the merge commit
#   - Create the GitHub release with the notes from the PR
#   - Delete the v0.3.0-pre tag

# 8. Pull the latest changes
git checkout main
git pull origin main

# 9. Verify the release
git show v0.3.0  # Should show the merge commit with version updates
# View release at: https://github.com/kamping-site/kamping/releases/tag/v0.3.0

# 10. (Optional) Further edit release notes in GitHub UI if needed
```

### Quick Reference

**Start release (Option A - Tag):**
```bash
git tag -a v0.3.0-pre -m "Prepare release v0.3.0" && git push origin v0.3.0-pre
```

**Start release (Option B - Manual):**
- Go to Actions → Run workflow → Enter version

**Cancel release:**
```bash
# Close the PR in GitHub, then (only if using tag method):
git push --delete origin v0.3.0-pre
```

**After merge:**
```bash
git pull origin main && git show v0.3.0
```

## Additional Notes

- The release workflow requires the `contents: write` and `pull-requests: write` permissions
- Release notes are generated from the commit history between the current and previous tags
- The workflow uses the GitHub CLI (`gh`) and API for most operations
- All commits are made by `github-actions[bot]`
