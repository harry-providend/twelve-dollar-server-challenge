"""Opens N idle keep-alive connections (each sends one request first), waits, then checks that every
one of them still answers. Run load against the server meanwhile to see that it is unaffected.

    python3 tests/idle_connections.py <host:port> <connections> <hold-seconds>

Uses source addresses 127.0.0.1-127.0.0.9 so that one machine can open more than ~28k connections.
Raise the client's `ulimit -n` first.
"""
import selectors, socket, sys, time

host, port = sys.argv[1].split(":"); port = int(port)
N, HOLD = int(sys.argv[2]), float(sys.argv[3])
REQ = b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n"

def open_all():
    socks = []
    for i in range(N):
        s = socket.socket(); s.bind((f"127.0.0.{1 + i % 9}", 0)); s.connect((host, port)); socks.append(s)
    return socks

def roundtrip_all(socks):
    """Sends one request on every socket and waits for every response; returns the number answered."""
    sel = selectors.DefaultSelector(); bufs = {}
    for s in socks:
        s.setblocking(False); s.send(REQ); sel.register(s, selectors.EVENT_READ); bufs[s] = b""
    done, deadline = 0, time.time() + 60
    while done < len(socks) and time.time() < deadline:
        for key, _ in sel.select(1):
            s = key.fileobj
            try: data = s.recv(4096)
            except OSError: data = b""
            bufs[s] += data
            if not data or bufs[s].endswith(b"}"):
                sel.unregister(s); done += bool(data) and bufs[s].startswith(b"HTTP/1.1 200")
    return done

t0 = time.time(); socks = open_all()
print(f"opened {N} connections in {time.time() - t0:.1f}s; first round answered: {roundtrip_all(socks)}")
time.sleep(HOLD)
ok = roundtrip_all(socks)
print(f"after {HOLD:.0f}s idle: {ok}/{N} answered")
sys.exit(0 if ok == N else 1)
