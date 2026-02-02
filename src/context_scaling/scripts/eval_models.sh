#!/usr/bin/env bash
set -euo pipefail

# ========= USER CONFIG =========
# Repo you want to create a NEW branch in (destination repo)
DEST_REPO_URL="git@github.com:ORG/dest-repo.git"
DEST_BASE_BRANCH="main"          # base branch to branch off from in dest repo
NEW_BRANCH="rewrite-from-other"  # new branch name to create in dest repo

# Repo whose code should overwrite everything in dest repo (source repo)
SRC_REPO_URL="git@github.com:ORG/source-repo.git"
SRC_BRANCH="main"               # branch to take code from in source repo

# A file from your CURRENT repo/branch to copy into the dest repo after overwrite
# (path relative to where you run this script)
FILE_FROM_CURRENT_REPO="path/to/local_file.py"
# where to place it inside the dest repo (relative path)
DEST_PATH_IN_DEST_REPO="scripts/local_file.py"

# Commit message in dest repo
COMMIT_MSG="Rewrite code from source repo + add local file"
# =================================

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing command: $1" >&2; exit 1; }; }
need_cmd git
need_cmd rsync
need_cmd mktemp

# Capture current repo root (for copying FILE_FROM_CURRENT_REPO)
CUR_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${CUR_ROOT}" ]]; then
  echo "ERROR: Must run from inside a git repo (so I can locate FILE_FROM_CURRENT_REPO)." >&2
  exit 1
fi

LOCAL_FILE_ABS="${CUR_ROOT}/${FILE_FROM_CURRENT_REPO}"
if [[ ! -f "${LOCAL_FILE_ABS}" ]]; then
  echo "ERROR: Local file not found: ${LOCAL_FILE_ABS}" >&2
  exit 1
fi

WORKDIR="$(mktemp -d)"
cleanup() { rm -rf "${WORKDIR}"; }
trap cleanup EXIT

DEST_DIR="${WORKDIR}/dest"
SRC_DIR="${WORKDIR}/src"

echo "Workdir: ${WORKDIR}"
echo "Current repo root: ${CUR_ROOT}"

# 1) Clone destination repo and create new branch
echo "Cloning dest repo..."
git clone "${DEST_REPO_URL}" "${DEST_DIR}"
cd "${DEST_DIR}"
git fetch --all --prune

echo "Checking out base branch: ${DEST_BASE_BRANCH}"
git checkout "${DEST_BASE_BRANCH}"
git pull --ff-only origin "${DEST_BASE_BRANCH}"

echo "Creating new branch: ${NEW_BRANCH}"
git checkout -b "${NEW_BRANCH}"

# 2) Clone source repo (the one whose code becomes the new content)
echo "Cloning source repo..."
git clone --branch "${SRC_BRANCH}" --single-branch "${SRC_REPO_URL}" "${SRC_DIR}"

# 3) Wipe dest repo working tree (but keep .git), then copy source content in
echo "Overwriting dest repo contents with source repo contents..."
# Remove everything except .git
find "${DEST_DIR}" -mindepth 1 -maxdepth 1 ! -name ".git" -exec rm -rf {} +

# Copy source -> dest, excluding source .git
rsync -a --delete \
  --exclude ".git" \
  "${SRC_DIR}/" "${DEST_DIR}/"

# 4) Copy the local file from CURRENT repo into the dest repo
DEST_FILE_ABS="${DEST_DIR}/${DEST_PATH_IN_DEST_REPO}"
mkdir -p "$(dirname "${DEST_FILE_ABS}")"
cp -f "${LOCAL_FILE_ABS}" "${DEST_FILE_ABS}"
echo "Copied local file: ${LOCAL_FILE_ABS} -> ${DEST_FILE_ABS}"

# 5) Commit everything in dest repo
cd "${DEST_DIR}"

# If you want to keep dest repo's own .gitignore, comment out the next line.
# (As written, source repo's .gitignore will be used if it exists in source.)
git add -A

if git diff --cached --quiet; then
  echo "No changes to commit. (Did source repo match dest already?)"
else
  git commit -m "${COMMIT_MSG}"
  echo "Committed on branch ${NEW_BRANCH}"
fi

# 6) Push new branch
echo "Pushing branch to origin..."
git push -u origin "${NEW_BRANCH}"

echo "Done."
echo "Created branch: ${NEW_BRANCH} in ${DEST_REPO_URL}"
