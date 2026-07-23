#!/usr/bin/env bash
set -euo pipefail
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VYVAR_RELEASE_BUNDLE=1
if [[ -x "${INSTALL_DIR}/python/bin/python3" ]]; then
  PY_HOME="${INSTALL_DIR}/python"
elif [[ -x "${INSTALL_DIR}/python/python/bin/python3" ]]; then
  PY_HOME="${INSTALL_DIR}/python/python"
else
  PY_BIN="$(find "${INSTALL_DIR}/python" -name python3 -type f 2>/dev/null | head -1)"
  PY_HOME="$(dirname "${PY_BIN}")"
fi
export PYTHONHOME="${PY_HOME}"
export PYTHONPATH="${INSTALL_DIR}/src_py:${PYTHONPATH:-}"
export PATH="${PY_HOME}/bin:${PATH}"
PY="${PY_HOME}/bin/python3"
if [[ "${1:-}" == "--selftest" ]]; then
  exec "${PY}" "${INSTALL_DIR}/vyvar_selftest.py"
fi
exec "${PY}" -m streamlit run "${INSTALL_DIR}/app.py" --server.headless true "$@"
