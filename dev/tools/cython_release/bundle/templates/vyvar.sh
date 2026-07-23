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
PY="${PY_HOME}/bin/python3"
# Isolated mode: ignore host PYTHONPATH/PYTHONHOME/user-site (bundle contamination fix).
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PATH="${PY_HOME}/bin:${PATH}"
if [[ "${1:-}" == "--selftest" ]]; then
  exec "${PY}" -I "${INSTALL_DIR}/vyvar_selftest.py"
fi
exec "${PY}" -I -m streamlit run "${INSTALL_DIR}/app.py" --server.headless true "$@"
