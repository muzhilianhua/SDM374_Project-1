#!/usr/bin/env bash
# Build the offline course-computer kit (T10, temporary version for the 2026-11-12 test).
#
#   deploy/course-pc/build_kit.sh <python-3.11-windows-install_only.tar.gz> <wheels dir> <output dir> [bat|script]
#
# bat (default): numbered .bat steps. script: a single run.py in the shape of the course solver/run.py.
#
# The Python archive is python-build-standalone's cpython-3.11.x-x86_64-pc-windows-msvc-install_only
# build (the one `uv python install` uses). Wheels: on any machine,
#   uv export --format requirements-txt --no-hashes --group dev --no-emit-project > req.txt
#   uvx pip download --only-binary=:all: --platform win_amd64 --python-version 3.11 \
#       --implementation cp -r req.txt -d wheels
#   uvx pip download --only-binary=:all: --platform win_amd64 --python-version 3.11 colorama pip -d wheels
# (pip evaluates `sys_platform == 'win32'` markers on the host, so colorama is fetched by name.)
# Packs the committed HEAD only; uncommitted changes are not in the kit.
set -euo pipefail

python_archive=$1
wheels=$2
out=$3
mode=${4:-bat}
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
commit=$(git -C "$repo" rev-parse HEAD)
suffix=""
[ "$mode" = script ] && suffix="-script"
name="sdm374-course-pc${suffix}-${commit:0:7}"
kit="$out/$name"

if [ -e "$kit" ]; then
  echo "$kit already exists; remove it or pick another output dir" >&2
  exit 1
fi
mkdir -p "$kit/repo" "$kit/results"
git -C "$repo" archive --format=tar HEAD | tar -x -C "$kit/repo"
printf '%s %s\n' "$commit" "$(git -C "$repo" log -1 --format=%cI HEAD)" > "$kit/repo/VERSION"
tar -xzf "$python_archive" -C "$kit"            # creates python/ with python.exe
test -f "$kit/python/python.exe"
mkdir -p "$kit/wheels"
cp "$wheels"/*.whl "$kit/wheels/"
if [ "$mode" = script ]; then
  cp "$repo/deploy/course-pc/run.py" "$kit/run.py"
else
  cp "$repo"/deploy/course-pc/bat/*.bat "$kit/"
fi
cp "$repo/deploy/course-pc/README.md" "$kit/README.md"
(cd "$out" && rm -f "$name.zip" && zip -qr -X "$name.zip" "$name")
echo "kit: $kit"
echo "zip: $out/$name.zip ($(du -h "$out/$name.zip" | cut -f1))"
