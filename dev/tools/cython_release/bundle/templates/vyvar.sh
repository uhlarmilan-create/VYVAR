#!/usr/bin/env bash
set -euo pipefail
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VYVAR_RELEASE_BUNDLE=1
export PYTHONHOME="${INSTALL_DIR}/python"
export PYTHONPATH="${INSTALL_DIR}/src_py:${PYTHONPATH:-}"
export PATH="${INSTALL_DIR}/python/bin:${PATH}"
PY="${INSTALL_DIR}/python/bin/python3"
if [[ "${1:-}" == "--selftest" ]]; then
  exec "${PY}" "${INSTALL_DIR}/vyvar_selftest.py"
fi
exec "${PY}" -m streamlit run "${INSTALL_DIR}/app.py" --server.headless true "$@"
