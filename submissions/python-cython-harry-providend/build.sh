#!/usr/bin/env bash
# Builds the app as a normal user:
#   1. SQLite 3.53.4 from source (Ubuntu's pinned orig tarball, sha256-checked), patched with
#      sqlite-seek.patch, amalgamated;
#   2. Cython (pinned, build-time only) turns fastserver.pyx into C;
#   3. both are compiled with -O3 and linked with LTO into one CPython extension next to server.py,
#      twice: the second build uses the profile of a training run of the first (PGO).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

SQLITE_VERSION=3.53.4
SQLITE_TARBALL="sqlite3_${SQLITE_VERSION}.orig.tar.xz"
SQLITE_SHA256=05366adc8ee80c6b1faa71285b49d133c2121238ca58934b077fdeb95934abd4
SQLITE_URLS=(
  "https://archive.ubuntu.com/ubuntu/pool/main/s/sqlite3/$SQLITE_TARBALL"
  "http://archive.ubuntu.com/ubuntu/pool/main/s/sqlite3/$SQLITE_TARBALL"
  "http://mirrors.digitalocean.com/ubuntu/pool/main/s/sqlite3/$SQLITE_TARBALL"
)
# The build options SQLite recommends for speed, minus the features this server does not use.
# THREADSAFE=0: every process here uses SQLite from a single thread, so all mutexes compile away.
SQLITE_DEFS=(
  -DSQLITE_THREADSAFE=0 -DSQLITE_DEFAULT_MEMSTATUS=0 -DSQLITE_DEFAULT_WAL_SYNCHRONOUS=1
  -DSQLITE_LIKE_DOESNT_MATCH_BLOBS -DSQLITE_MAX_EXPR_DEPTH=0 -DSQLITE_OMIT_DECLTYPE
  -DSQLITE_OMIT_DEPRECATED -DSQLITE_OMIT_PROGRESS_CALLBACK -DSQLITE_OMIT_SHARED_CACHE
  -DSQLITE_USE_ALLOCA -DSQLITE_OMIT_AUTOINIT -DSQLITE_DQS=0 -DSQLITE_OMIT_LOAD_EXTENSION
  -DSQLITE_OMIT_JSON -DSQLITE_OMIT_UTF16 -DSQLITE_OMIT_TRACE -DSQLITE_OMIT_AUTHORIZATION
  -DHAVE_FDATASYNC=1
)
CFLAGS=(-O3 -march=native -fPIC -flto=auto -fvisibility=hidden -fno-strict-aliasing)

SQLITE_DIR=".deps/sqlite3-$SQLITE_VERSION"
PATCH_SHA256="$(sha256sum sqlite-seek.patch | cut -d' ' -f1)"

mkdir -p .deps build
if [ ! -f "$SQLITE_DIR/sqlite3.c" ] || [ "$(cat "$SQLITE_DIR/.patched" 2>/dev/null)" != "$PATCH_SHA256" ]; then
  if [ ! -f ".deps/$SQLITE_TARBALL" ]; then
    for url in "${SQLITE_URLS[@]}"; do
      curl -fsSL --retry 3 -o ".deps/$SQLITE_TARBALL.part" "$url" && break || rm -f ".deps/$SQLITE_TARBALL.part"
    done
    mv ".deps/$SQLITE_TARBALL.part" ".deps/$SQLITE_TARBALL"
  fi
  echo "$SQLITE_SHA256  .deps/$SQLITE_TARBALL" | sha256sum -c -
  rm -rf "$SQLITE_DIR"
  tar -xJf ".deps/$SQLITE_TARBALL" -C .deps
  patch -d "$SQLITE_DIR" -p1 --quiet < sqlite-seek.patch
  # The amalgamation (one sqlite3.c) is generated with the Jim Tcl bundled in the source tree.
  (cd "$SQLITE_DIR" && ./configure --disable-tcl >/dev/null && make sqlite3.c >/dev/null)
  echo "$PATCH_SHA256" > "$SQLITE_DIR/.patched"
fi

"$PY" -m venv --clear .venv
.venv/bin/pip install --disable-pip-version-check --no-input --quiet --only-binary=:all: -r requirements.txt
.venv/bin/cython -3 fastserver.pyx -o build/fastserver.c
INC="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
EXT="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

compile() {  # SQLite + the extension, linked with LTO; extra arguments go to every step
  rm -f fastserver.*.so
  cc "${CFLAGS[@]}" "$@" -w "${SQLITE_DEFS[@]}" -c "$SQLITE_DIR/sqlite3.c" -o build/sqlite3.o
  cc "${CFLAGS[@]}" "$@" -Wno-unused-function -I"$SQLITE_DIR" -I"$INC" -c build/fastserver.c -o build/fastserver.o
  cc "${CFLAGS[@]}" "$@" -Wno-stringop-overread -shared build/fastserver.o build/sqlite3.o -lm -o "fastserver$EXT"
}

# Profile-guided optimization: build an instrumented server, run it on a synthetic database with a
# load-test-like request mix (pgo_train.py), then rebuild using the measured branch and call counts.
PROFILE="$PWD/build/pgo"
rm -rf "$PROFILE"
compile -fprofile-generate -fprofile-dir="$PROFILE"
if .venv/bin/python pgo_train.py && compgen -G "$PROFILE/*.gcda" >/dev/null; then
  compile -fprofile-use -fprofile-dir="$PROFILE" -fprofile-partial-training -Wno-missing-profile
else
  echo "PGO training failed: building without a profile"
  compile
fi
.venv/bin/python -c 'import fastserver; print("built fastserver with SQLite", fastserver.sqlite_version())'
