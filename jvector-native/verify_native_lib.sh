#!/bin/bash
# Copyright DataStax, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Verify a native-module or assembled multi-release jar. Default: both Linux
# libraries. --all is the four-platform release gate; --host is development only.
set -euo pipefail
MODULE_ROOT="$(cd "$(dirname "$0")" && pwd)"
PLATFORMS="linux-x86_64,linux-aarch64"
JAR=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --all) PLATFORMS="linux-x86_64,linux-aarch64,macos-x86_64,macos-aarch64"; shift ;;
    --host)
      case "$(uname -s)" in Linux) OS=linux ;; Darwin) OS=macos ;; *) echo 'Unsupported host OS' >&2; exit 1 ;; esac
      case "$(uname -m)" in x86_64|amd64) ARCH=x86_64 ;; arm64|aarch64) ARCH=aarch64 ;; *) echo 'Unsupported host architecture' >&2; exit 1 ;; esac
      PLATFORMS="${OS}-${ARCH}"; shift ;;
    --platforms) PLATFORMS="${2:?Specify a comma-separated platform list}"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) [ -z "$JAR" ] || { echo 'Specify one jar' >&2; exit 1; }; JAR="$1"; shift ;;
  esac
done
if [ -z "$JAR" ]; then
  for candidate in "$MODULE_ROOT"/target/jvector-native-*.jar; do
    case "$candidate" in *-sources.jar|*-javadoc.jar) continue ;; esac
    if [ -f "$candidate" ]; then JAR="$candidate"; break; fi
  done
fi
[ -f "$JAR" ] || { echo "Native jar missing: $JAR" >&2; exit 1; }
if command -v llvm-nm >/dev/null; then NM=llvm-nm; else NM=nm; fi
command -v "$NM" >/dev/null || { echo 'nm is required for release symbol verification' >&2; exit 1; }
# Keep this gate synchronized with the exported kernel list, including optional
# runtime kernels that must be present in a new release.
KERNEL_HEADER="$MODULE_ROOT/src/main/native/src/jvector_simd_kernel_list.h"
[ -f "$KERNEL_HEADER" ] || { echo "Kernel header missing: $KERNEL_HEADER" >&2; exit 1; }
REQUIRED_SYMBOLS="$(sed -nE 's/^[[:space:]]*KERNEL_ENTRY\([^,]+,[[:space:]]*([^,]+),.*/\1/p' "$KERNEL_HEADER")
jvector_supports_abi
jvector_simd_get_active_isa
jvector_simd_get_max_isa_env"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
ENTRIES="$(unzip -Z1 "$JAR")"
IFS=',' read -r -a TARGETS <<< "$PLATFORMS"
[ "${#TARGETS[@]}" -gt 0 ] || { echo 'No platforms selected' >&2; exit 1; }
FAIL=0
for PLATFORM in "${TARGETS[@]}"; do
  case "$PLATFORM" in
    linux-x86_64|linux-aarch64) EXT=so ;;
    macos-x86_64|macos-aarch64) EXT=dylib ;;
    *) echo "Unsupported platform: $PLATFORM" >&2; exit 1 ;;
  esac
  ARCH="${PLATFORM#*-}"
  LIBNAME="libjvector-${ARCH}.${EXT}"
  # An assembled jar must use the JDK-22 overlay, rather than a stray root copy.
  if [[ "$ENTRIES" == *'META-INF/versions/22/'* ]]; then
    ENTRY="META-INF/versions/22/$LIBNAME"
  else
    ENTRY="$LIBNAME"
  fi
  LIB="$WORKDIR/$LIBNAME"
  if ! unzip -p "$JAR" "$ENTRY" > "$LIB" 2>/dev/null || [ ! -s "$LIB" ]; then
    echo "MISSING $PLATFORM: $ENTRY" >&2; FAIL=1; continue
  fi
  DESCRIPTION="$(file "$LIB")"
  echo "Checking $PLATFORM: $DESCRIPTION"
  case "$PLATFORM" in
    linux-*) [[ "$DESCRIPTION" == *ELF* ]] || { echo 'Expected ELF' >&2; FAIL=1; } ;;
    macos-*) [[ "$DESCRIPTION" == *Mach-O* ]] || { echo 'Expected Mach-O' >&2; FAIL=1; } ;;
  esac
  case "$ARCH" in
    x86_64) [[ "$DESCRIPTION" == *x86-64* || "$DESCRIPTION" == *x86_64* ]] || { echo 'Wrong architecture' >&2; FAIL=1; } ;;
    aarch64) [[ "$DESCRIPTION" == *aarch64* || "$DESCRIPTION" == *ARM64* || "$DESCRIPTION" == *arm64* ]] || { echo 'Wrong architecture' >&2; FAIL=1; } ;;
  esac
  if [ "$EXT" = so ]; then
    SYMBOLS="$("$NM" -D --defined-only "$LIB" 2>/dev/null || "$NM" -g "$LIB" 2>/dev/null || true)"
  else
    SYMBOLS="$("$NM" -g -U "$LIB" 2>/dev/null || "$NM" -g "$LIB" 2>/dev/null || true)"
  fi
  for SYMBOL in $REQUIRED_SYMBOLS; do
    # Require a definition, not an undefined reference that happens to have
    # the expected name. Mach-O C definitions carry a leading underscore.
    if ! grep -Eq "[[:space:]][TtWw][[:space:]]_?${SYMBOL}$" <<< "$SYMBOLS"; then
      echo "MISSING $PLATFORM symbol: $SYMBOL" >&2; FAIL=1
    fi
  done
  if [ "$EXT" = dylib ] && [ "$(uname -s)" = Darwin ]; then
    otool -L "$LIB"
  elif [ "$EXT" = so ] && [ "$(uname -s)" = Linux ] && command -v ldd >/dev/null; then
    case "$(uname -m):$ARCH" in
      x86_64:x86_64|aarch64:aarch64)
        DEPENDENCIES="$(ldd "$LIB" 2>&1 || true)"
        echo "$DEPENDENCIES"
        if echo "$DEPENDENCIES" | grep -qi 'not found'; then FAIL=1; fi ;;
      *) echo "Foreign architecture: runtime dependencies require target-host CI" ;;
    esac
  else
    echo "Foreign OS: runtime dependencies require target-host CI"
  fi
done
[ "$FAIL" -eq 0 ] || { echo 'Native verification FAILED' >&2; exit 1; }
echo "Native verification passed: $PLATFORMS"
