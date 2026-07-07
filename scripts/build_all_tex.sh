#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "未找到 ${PYTHON}，请先创建项目虚拟环境并安装 requirements.txt。" >&2
  exit 1
fi

cd "${ROOT_DIR}"
"${PYTHON}" scripts/render_all_tex.py

echo
echo "全部 LaTeX 表格已生成："
echo "  output/pdf/all_latex_tables.pdf"
echo "  output/pdf/tables/*.pdf"
echo "  doc/assets/tables/*.svg"
echo "  doc/assets/tables/*.png"
