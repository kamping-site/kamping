# Release Process Quick Start

This is a quick reference for creating releases in the KaMPIng repository. For detailed documentation, see [Release Guidelines](../docs/guidelines/release_guidelines.md).

## TL;DR

```bash
# 1. Start release
git tag -a v0.3.0-pre -m "Prepare release v0.3.0"
git push origin v0.3.0-pre

# 2. Review the auto-created PR at github.com/kamping-site/kamping/pulls

# 3. Merge the PR → Release is automatically created!

# 4. Pull latest
git pull origin main && git show v0.3.0
```

## Two-Step Process

### Step 1: Push Pre-Release Tag → Creates PR

```bash
git checkout main
git pull origin main
git tag -a vX.Y.Z-pre -m "Prepare release vX.Y.Z"
git push origin vX.Y.Z-pre
```

**What happens:**
- Automated workflow creates a release PR
- PR includes version updates in CMakeLists.txt, CITATION.cff, README.md
- PR includes preview of release notes

### Step 2: Review & Merge PR → Creates Release

1. Go to https://github.com/kamping-site/kamping/pulls
2. Review the "Release vX.Y.Z" PR
3. Edit PR description to refine release notes if needed
4. Merge the PR

**What happens:**
- Final tag `vX.Y.Z` is created at merge commit
- GitHub release is published with notes from PR
- Pre-release tag `vX.Y.Z-pre` is deleted

## Cancel a Release

```bash
# Close the PR in GitHub UI, then:
git push --delete origin vX.Y.Z-pre
```

## Version Numbers

- **Major (X.0.0)**: Breaking changes
- **Minor (0.X.0)**: New features, backwards-compatible
- **Patch (0.0.X)**: Bug fixes, backwards-compatible

## Tag Format

✅ Correct: `v0.3.0-pre` → starts release process  
✅ Correct: `v1.0.0-pre` → starts release process  
❌ Wrong: `v0.3.0` → won't trigger (final tags are auto-created)  
❌ Wrong: `0.3.0-pre` → missing 'v' prefix  

## Troubleshooting

**PR not created?**
- Check Actions tab for errors
- Verify tag format: `vX.Y.Z-pre`
- Ensure release doesn't already exist

**Can't merge PR?**
- Resolve any merge conflicts
- Ensure all checks pass

**Release not created after merge?**
- Check Actions tab for errors
- Manually create tag if needed: `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push origin vX.Y.Z`

## See Also

- [Complete Release Guidelines](../docs/guidelines/release_guidelines.md)
- [Release Notes Template](release-template.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
