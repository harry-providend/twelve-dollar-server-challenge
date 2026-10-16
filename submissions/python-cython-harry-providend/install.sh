#!/usr/bin/env bash
# Runs once as root on a clean Ubuntu 24.04: Python 3.12 with headers and venv, and a C toolchain.
# SQLite itself is built from source by build.sh.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential \
  ca-certificates curl xz-utils patch
python3 --version
