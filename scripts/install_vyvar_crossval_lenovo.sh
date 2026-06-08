#!/bin/bash
# VYVAR Cross-Validation Setup — Kubuntu 24.04 / Python 3.12
# Run: bash install_vyvar_crossval_lenovo.sh
# Milan Lenovo T460 — /home/milan/Public/vyvar_catalogs

set -e
echo "=== VYVAR Cross-Validation Setup ==="

# 1. System dependencies
sudo apt-get update
sudo apt-get install -y \
    source-extractor \
    python3-pip \
    python3-venv \
    python3-dev \
    libcfitsio-dev \
    wcslib-dev \
    git \
    curl

# 2. Python venv
cd ~
python3 -m venv vyvar_crossval
source ~/vyvar_crossval/bin/activate

# 3. Python packages
pip install --upgrade pip
pip install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    astropy \
    photutils \
    astroquery \
    sep \
    regions

# 4. IRAF via AstroConda (lightweight — only iraf.noao.digiphot.apphot)
# Modern approach: use pyraf or ccdproc instead of full IRAF
pip install pyraf || echo "pyraf optional — skipping if fails"
pip install ccdproc

# 5. Verify SExtractor
which sex || which source-extractor || echo "WARNING: SExtractor not found"
sex --version 2>/dev/null || source-extractor --version 2>/dev/null || true

# 6. Test imports
python3 -c "
import numpy, scipy, matplotlib, pandas, astropy, photutils
print('numpy:', numpy.__version__)
print('astropy:', astropy.__version__)
print('photutils:', photutils.__version__)
print('ALL OK')
"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chmod +x "${SCRIPT_DIR}/validate_photometry_crossval.py" 2>/dev/null || true

echo ""
echo "=== Setup complete ==="
echo "Activate with: source ~/vyvar_crossval/bin/activate"
echo "DB path: /home/milan/Public/vyvar_catalogs"
