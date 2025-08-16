#!/usr/bin/env bash
set -e
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

python -c "import sys,platform; print('Python',sys.version); print('Platform',platform.platform())"
python -c "import ccxt, pandas; print('ccxt', ccxt.__version__, 'pandas', pandas.__version__)"