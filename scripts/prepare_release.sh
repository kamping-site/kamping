#!/bin/bash
# Script to prepare a new release for KaMPIng
# This script helps prepare all files for a new release before pushing the tag

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print functions
print_error() {
    echo -e "${RED}Error: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if version is provided
if [ $# -eq 0 ]; then
    print_error "No version specified"
    echo "Usage: $0 <version>"
    echo "Example: $0 0.3.0"
    exit 1
fi

VERSION=$1
TAG="v$VERSION"

# Validate version format
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    print_error "Invalid version format: $VERSION"
    echo "Version must follow semantic versioning (e.g., 0.3.0 or 0.3.0-beta.1)"
    exit 1
fi

print_info "Preparing release $TAG"

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    print_error "You must be on the main branch to prepare a release"
    echo "Current branch: $CURRENT_BRANCH"
    exit 1
fi

# Check if working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    print_error "Working directory is not clean"
    echo "Please commit or stash your changes before preparing a release"
    git status --short
    exit 1
fi

# Pull latest changes
print_info "Pulling latest changes from origin..."
git pull origin main

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    print_error "Tag $TAG already exists"
    exit 1
fi

# Create a new branch for version updates
BRANCH="prepare-release-$VERSION"
print_info "Creating branch $BRANCH..."
git checkout -b "$BRANCH"

# Update CMakeLists.txt
if [ -f CMakeLists.txt ]; then
    if grep -q "VERSION [0-9]" CMakeLists.txt; then
        sed -i "s/VERSION [0-9][0-9.]*/VERSION $VERSION/" CMakeLists.txt
        print_success "Updated CMakeLists.txt"
    else
        print_error "Could not find version in CMakeLists.txt"
        git checkout main
        git branch -D "$BRANCH"
        exit 1
    fi
fi

# Update CITATION.cff
if [ -f CITATION.cff ]; then
    sed -i "s/^version: .*/version: $VERSION/" CITATION.cff
    # Update date-released to today
    TODAY=$(date +%Y-%m-%d)
    sed -i "s/^date-released: .*/date-released: $TODAY/" CITATION.cff
    print_success "Updated CITATION.cff"
else
    print_info "CITATION.cff not found, skipping"
fi

# Update README.md
if [ -f README.md ]; then
    if grep -q "GIT_TAG v[0-9]" README.md; then
        sed -i "s/GIT_TAG v[0-9][0-9.]*/GIT_TAG $TAG/" README.md
        print_success "Updated README.md"
    else
        print_info "No version reference found in README.md, skipping"
    fi
fi

# Show changes
print_info "Changes to be committed:"
git diff --stat

echo ""
print_info "Review the changes above. Do you want to continue? (y/n)"
read -r response

if [[ ! "$response" =~ ^[Yy]$ ]]; then
    print_info "Aborting release preparation"
    git checkout main
    git branch -D "$BRANCH"
    exit 0
fi

# Commit changes
git add CMakeLists.txt CITATION.cff README.md
git commit -m "Prepare $TAG release"
print_success "Changes committed"

# Show next steps
echo ""
print_success "Release preparation complete!"
echo ""
echo "Next steps:"
echo "  1. Review the changes:"
echo "     git show HEAD"
echo ""
echo "  2. Push the branch and create a PR:"
echo "     git push origin $BRANCH"
echo "     gh pr create --title \"Prepare $TAG release\" --body \"Prepare version files for $TAG release\""
echo ""
echo "  3. After the PR is merged to main, create and push the tag:"
echo "     git checkout main"
echo "     git pull origin main"
echo "     git tag -a $TAG -m \"Release $TAG\""
echo "     git push origin $TAG"
echo ""
echo "  4. The automated workflow will create the release and a follow-up PR"
echo ""
print_info "Or, if you want to undo these changes:"
echo "     git checkout main"
echo "     git branch -D $BRANCH"
