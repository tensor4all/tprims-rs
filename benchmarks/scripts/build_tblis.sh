#!/usr/bin/env bash
# Build the pinned TBLIS baseline into an install prefix.
#
#   build_tblis.sh <prefix>
#
# The prefix is what `TBLIS_ROOT` points at: it must end up with
# `lib/libtblis.so` and `include/tblis.h`, which is exactly what
# `benchmarks/build.rs` links under `--features tblis`, and a `PROVENANCE` the
# campaign's `scripts/record_run.py` reads to say which TBLIS a published cell
# measured.
#
# Pinned by release *tag*, never by a branch, and never by a local revision:
#
#   tblis      tag v2.0-beta2, which resolves to
#              b16a732939d8454021e0a0f0097cf1cc1dd3ab19; `tblis-version` reads 2.0
#   BLIS       the commit in the checkout's `blis-git-tag` file, read from the
#              checkout rather than hard-coded here. TBLIS 2.x fetches BLIS
#              itself with `FetchContent` from https://github.com/flame/blis.git
#              at that tag, so the *build* needs the network for BLIS; this
#              script performs no other network access.
#
# **TBLIS 2.x builds through CMake.** Its `configure` is a checked-in autoconf
# wrapper that invokes `cmake` (>= 3.23, CMakeLists.txt) and leaves a Makefile
# behind, so `cmake` and a C/C++ toolchain must be on PATH; `autoreconf` is not
# needed. The clone must be recursive: TBLIS's CMake refuses to configure
# without its `marray`, `tci` and `stl_ext` submodules.
#
# The BLIS configuration family is explicit and never `auto`, and nothing is
# compiled with `-march=native`: the prefix then depends on the pinned sources
# and the chosen family rather than on the host that built it, so a release of
# the same class can be rebuilt with the same kernels. The cost is that a newer
# family's kernels go unused. `config=` in the written PROVENANCE is the BLIS
# configuration actually resolved inside that family.
#
# The build tree is kept: it is where the resolved BLIS configuration is read
# from, and a failed build is only diagnosable in place. Set `TBLIS_BUILD_DIR`
# to choose it.
set -euo pipefail

TAG=v2.0-beta2
FAMILY="${TBLIS_BLIS_CONFIG_FAMILY:-zen3}"
URL=https://github.com/MatthewsResearchGroup/tblis.git

prefix_in="${1:?usage: build_tblis.sh <prefix>}"
mkdir -p "$prefix_in"
prefix="$(cd "$prefix_in" && pwd -P)"
work="${TBLIS_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/tblis-build.XXXXXX")}"
jobs="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)}"

echo "tblis: $TAG, BLIS config family $FAMILY"
echo "prefix: $prefix"
echo "build tree: $work"

if [[ -d "$work/tblis/.git" ]]; then
    # Re-running with the same TBLIS_BUILD_DIR reuses the tree: a failed build is
    # worth retrying in place, and the tree is where the resolved BLIS
    # configuration is read from.
    git -C "$work/tblis" fetch --tags --quiet
    git -C "$work/tblis" checkout "$TAG"
    git -C "$work/tblis" submodule update --init --recursive
else
    git clone --recursive "$URL" "$work/tblis"
    git -C "$work/tblis" checkout "$TAG"
    git -C "$work/tblis" submodule update --init --recursive
fi
cd "$work/tblis"

commit="$(git rev-parse HEAD)"
tag="$(git describe --tags --exact-match)"
version="$(cat tblis-version)"
blis_commit="$(cat blis-git-tag)"
echo "checked out tag $tag -> commit $commit"
echo "tblis version $version, pinned BLIS commit $blis_commit"

./configure --prefix="$prefix" --with-blis-config-family="$FAMILY"
make -j"$jobs"
make install

# TBLIS's CMake looks for a BLIS already installed on the host (`find_package`,
# then pkgconfig) *before* it fetches the pinned one, and its `--with-blis-*`
# options only add search paths. On a host with a system BLIS the pin would be
# ignored and the PROVENANCE below would name a revision the prefix does not
# contain, which is worse than not building at all.
# The directory is looked up by *name*, not by a path glob: a build directory
# called `tblis-build-<tag>` contains the substring "blis-build" and made a
# `-path '*blis-build*'` match TBLIS's own `tblis/plugin/config.mk`, whose
# CONFIG_NAME does not exist.
blis_build="$(find "$work/tblis" -maxdepth 5 -type d -name blis-build -print -quit || true)"
config_mk="${blis_build:+$blis_build/config.mk}"
if [[ -z "$blis_build" || ! -f "$config_mk" ]]; then
    echo "ERROR: no vendored BLIS build under $work/tblis: the build found a BLIS" >&2
    echo "already installed on this host and used that instead of the pinned" >&2
    echo "$blis_commit. Hide it from pkg-config/CMake (or build elsewhere) so" >&2
    echo "that the recorded BLIS revision is the one the prefix was built with." >&2
    exit 1
fi

# What BLIS resolved inside the requested family; a family may have
# sub-configurations and `zen3` need not be what its `CONFIG_NAME` ends up as.
# BLIS writes a make assignment (`CONFIG_NAME := zen3`), so the value is what
# follows the `=`; taking the first token after the name would record `:=`.
config="$(sed -n 's/^CONFIG_NAME[[:space:]]*:*=[[:space:]]*\([A-Za-z0-9_.-][A-Za-z0-9_.-]*\).*/\1/p' "$config_mk" | head -1)"
if [[ -z "$config" ]]; then
    echo "ERROR: $config_mk has no readable CONFIG_NAME line; cannot record the BLIS configuration" >&2
    echo "       (looked for a make assignment such as 'CONFIG_NAME := zen3')" >&2
    exit 1
fi

so="$prefix/lib/libtblis.so"
if [[ ! -f "$so" ]]; then
    echo "ERROR: no $so after install; build.rs links the shared library" >&2
    exit 1
fi
header="$prefix/include/tblis.h"
if [[ ! -f "$header" ]]; then
    echo "ERROR: no $header after install; TBLIS_ROOT must be an install prefix" >&2
    exit 1
fi
sha="$(sha256sum "$so" | awk '{print $1}')"
cc_version="$(cc --version 2>&1 | head -1 || true)"

{
    printf 'version=%s\n' "$version"
    printf 'tag=%s\n' "$tag"
    printf 'commit=%s\n' "$commit"
    printf 'blis_commit=%s\n' "$blis_commit"
    printf 'config=%s\n' "$config"
    printf 'config_family=%s\n' "$FAMILY"
    printf 'build=%s\n' "./configure --prefix=$prefix --with-blis-config-family=$FAMILY (cmake, Unix Makefiles, Release)"
    printf 'compiler=%s\n' "$cc_version"
    # `sha256=` is the field the campaign's recorder reads into the manifest's
    # `providers[].sha256`; it is the hash of the artifact that gets linked.
    printf 'sha256=%s\n' "$sha"
} > "$prefix/PROVENANCE"

echo "installed $so (sha256 $sha)"
cat "$prefix/PROVENANCE"
