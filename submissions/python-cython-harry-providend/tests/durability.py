"""Acknowledged writes survive SIGKILL, and the WAL stays bounded under sustained writes.

    python3 tests/durability.py <submission-dir> <seed feed.db>

Starts the server on a temporary copy of the database, writes from 32 threads for a while (enough
to run many checkpoints), SIGKILLs the server and its checkpointer mid-load, restarts it on the same
file and checks that every post and like that got a 201 is there.
"""
import base64, hashlib, hmac, json, os, shutil, signal, socket, subprocess, sys, tempfile, threading, time
import urllib.request

DIR, SEED = sys.argv[1], sys.argv[2]
PORT = 3500
SECONDS = float(os.environ.get("SECONDS_OF_LOAD", "20"))

def b64(b): return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
def token(uid):
    h = b64(b'{"alg":"HS256","typ":"JWT"}')
    p = b64(json.dumps({"sub": str(uid), "username": f"u{uid}", "exp": time.time() + 3600}).encode())
    return f"{h}.{p}." + b64(hmac.new(b"twelve-dollar-challenge", f"{h}.{p}".encode(), hashlib.sha256).digest())

def start(db):
    env = dict(os.environ, SQLITE_PATH=db, JWT_SECRET="twelve-dollar-challenge", HOST="127.0.0.1", PORT=str(PORT))
    log = open(db + ".log", "ab")
    p = subprocess.Popen(["bash", os.path.join(DIR, "start.sh")], env=env, stdout=log, stderr=log)
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1).read(); return p
        except OSError: time.sleep(0.1)
    p.kill()
    raise SystemExit("server did not start:\n" + open(db + ".log").read())

def worker(uid, stop, acked):
    tok = token(uid)
    s = socket.create_connection(("127.0.0.1", PORT)); f = s.makefile("rb")
    n = 0
    try:
        while not stop.is_set():
            n += 1
            if n % 3:
                body = json.dumps({"body": f"durable {uid}-{n}"}).encode()
                s.sendall(f"POST /posts HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {tok}\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
            else:
                s.sendall(f"POST /posts/{1 + (uid * 7919 + n) % 499999}/like HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {tok}\r\nContent-Length: 0\r\n\r\n".encode())
            line = f.readline()
            if not line: break
            st = int(line.split()[1]); length = 0
            while (h := f.readline()) not in (b"\r\n", b""):
                if h.lower().startswith(b"content-length:"): length = int(h.split(b":")[1])
            b = json.loads(f.read(length))
            if st == 201:
                acked.append(("post", b["post"]["id"], b["post"]["body"]) if "post" in b else ("like", uid, b["post_id"]))
    except OSError:
        pass

tmp = tempfile.mkdtemp(); db = os.path.join(tmp, "feed.db"); shutil.copy(SEED, db)
server = start(db)
stop = threading.Event(); acked = []
threads = [threading.Thread(target=worker, args=(1000 + i, stop, acked)) for i in range(32)]
for t in threads: t.start()
max_wal = 0; t0 = time.time()
while time.time() - t0 < SECONDS:
    time.sleep(0.25)
    try: max_wal = max(max_wal, os.path.getsize(db + "-wal"))
    except OSError: pass
pid = int(subprocess.check_output(["ss", "-Hltnp", f"sport = :{PORT}"]).decode().split("pid=")[1].split(",")[0])
children = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
for p in [pid] + [int(c) for c in children]: os.kill(p, signal.SIGKILL)   # mid-load, no shutdown
stop.set()
for t in threads: t.join()
server.wait()
snapshot = list(acked)
print(f"acknowledged before SIGKILL: {sum(a[0]=='post' for a in snapshot)} posts, {sum(a[0]=='like' for a in snapshot)} likes; "
      f"max WAL {max_wal/1048576:.1f} MiB")

server = start(db)
import sqlite3
con = sqlite3.connect(db)
missing = 0
for kind, a, b in snapshot:
    if kind == "post":
        row = con.execute("SELECT body FROM posts WHERE id = ?", (a,)).fetchone()
        missing += row is None or row[0] != b
    else:
        missing += con.execute("SELECT 1 FROM likes WHERE user_id = ? AND post_id = ?", (a, b)).fetchone() is None
print("missing after restart:", missing)
urllib.request.urlopen(f"http://127.0.0.1:{PORT}/feed", timeout=2).read()
server.send_signal(signal.SIGTERM); server.wait()
shutil.rmtree(tmp)
ok = missing == 0 and max_wal < 256 * 1048576
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
