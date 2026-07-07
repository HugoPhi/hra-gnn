#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "提示：该兼容入口现在会编译仓库中的全部 LaTeX 表格。"
exec "${ROOT_DIR}/scripts/build_all_tex.sh"
