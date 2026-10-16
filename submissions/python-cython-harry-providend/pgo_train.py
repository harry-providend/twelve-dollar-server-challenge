"""Profile-guided optimization training run, used by build.sh (standard library only).

Creates a small synthetic database with the challenge schema in a temporary directory, starts the
instrumented server on it, sends it a request mix like the load test's (feed, single posts, likes,
new posts, a few errors) over keep-alive connections, then stops it with SIGTERM so that GCC's
profile is written. The real database is never touched.

    .venv/bin/python pgo_train.py
"""
import base64
import hashlib
import hmac
import json
import os
import random
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SECRET = b"pgo-training-secret"
USERS, POSTS, REQUESTS_PER_CLIENT, CLIENTS = 5_000, 60_000, 5_000, 16
SCHEMA = """
CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')));
CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
  body TEXT NOT NULL CHECK (length(body) BETWEEN 1 AND 500),
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')));
CREATE TABLE likes (user_id INTEGER NOT NULL REFERENCES users(id), post_id INTEGER NOT NULL REFERENCES posts(id),
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')), PRIMARY KEY (user_id, post_id));
CREATE INDEX posts_created_at_id_idx ON posts (created_at DESC, id DESC);
CREATE INDEX posts_user_id_idx ON posts (user_id);
CREATE INDEX likes_post_id_idx ON likes (post_id);
"""
WORDS = ("server database request latency cache index query benchmark droplet cpu memory disk network "
         "coffee morning project deploy release bug feature test rewrite commit branch merge review").split()


def make_db(path, rnd):
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO users VALUES (?, ?, '2024-01-01T00:00:00.000Z')",
                   ((i, f"user_{i}") for i in range(1, USERS + 1)))
    t0 = time.mktime((2025, 1, 1, 0, 0, 0, 0, 0, 0))
    db.executemany("INSERT INTO posts VALUES (?, ?, ?, ?)", (
        (i, 1 + int(USERS * rnd.random() ** 2), " ".join(rnd.choices(WORDS, k=rnd.randint(4, 30))).capitalize() + ".",
         time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t0 + i * 500)) + ".%03dZ" % (i % 1000))
        for i in range(1, POSTS + 1)))
    db.executemany("INSERT OR IGNORE INTO likes (user_id, post_id, created_at) VALUES (?, ?, '2025-06-01T00:00:00.000Z')",
                   ((rnd.randint(1, USERS), rnd.randint(1, POSTS)) for _ in range(4 * POSTS)))
    db.execute("ANALYZE")
    db.commit()
    db.execute("PRAGMA journal_mode = WAL")
    db.close()


def token(uid, rnd):
    enc = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()
    head = enc(b'{"alg":"HS256","typ":"JWT"}')
    body = enc(json.dumps({"sub": str(uid), "username": f"user_{uid}", "iat": int(time.time()),
                           "exp": int(time.time()) + 3600}).encode())
    sig = enc(hmac.new(SECRET, f"{head}.{body}".encode(), hashlib.sha256).digest())
    return f"{head}.{body}.{sig}"


def client(port, seed, errors):
    rnd = random.Random(seed)
    tok = token(rnd.randint(1, USERS), rnd)
    s = socket.create_connection(("127.0.0.1", port))
    f = s.makefile("rb")
    ids = [POSTS]
    for n in range(REQUESTS_PER_CLIENT):
        r = rnd.random()
        pid = rnd.choice(ids)
        if r < 0.45:
            req = b"GET /feed HTTP/1.1\r\nHost: x\r\n\r\n"
        elif r < 0.90:
            req = f"GET /posts/{pid} HTTP/1.1\r\nHost: x\r\n\r\n".encode()
        elif r < 0.97:
            req = (f"POST /posts/{pid}/like HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {tok}\r\n"
                   "Content-Length: 0\r\n\r\n").encode()
        elif r < 0.99:
            body = json.dumps({"body": f"training post {seed}-{n} " + " ".join(rnd.choices(WORDS, k=8))}).encode()
            req = (f"POST /posts HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {tok}\r\n"
                   f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
        else:
            req = rnd.choice([b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n", b"GET /posts/abc HTTP/1.1\r\n\r\n",
                              b"GET /nope HTTP/1.1\r\n\r\n", b"POST /posts HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}"])
        s.sendall(req)
        status = f.readline()
        length = 0
        while (line := f.readline()) not in (b"\r\n", b""):
            if line[:15].lower() == b"content-length:":
                length = int(line[15:])
        data = f.read(length)
        if not status.startswith(b"HTTP/1.1 "):
            errors.append(status)
            return
        if data.startswith(b'{"posts":['):
            ids = [p["id"] for p in json.loads(data)["posts"]]
    s.close()


def main():
    rnd = random.Random(42)
    tmp = tempfile.mkdtemp(prefix="pgo-train-")
    try:
        path = os.path.join(tmp, "train.db")
        make_db(path, rnd)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = dict(os.environ, SQLITE_PATH=path, JWT_SECRET=SECRET.decode(), HOST="127.0.0.1", PORT=str(port))
        server = subprocess.Popen([sys.executable, os.path.join(HERE, "server.py")], env=env, cwd=HERE,
                                  stdout=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        errors = []
        threads = [threading.Thread(target=client, args=(port, i, errors)) for i in range(CLIENTS)]
        started = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        server.send_signal(signal.SIGTERM)  # a clean exit writes the profile
        server.wait(timeout=30)
        total = CLIENTS * REQUESTS_PER_CLIENT
        print(f"pgo training: {total} requests in {time.time() - started:.1f}s, server exit {server.returncode}")
        return 1 if errors or server.returncode else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
