"""HTTP and API edge cases beyond test/test.sh, over raw sockets.

    python3 tests/edge_cases.py [host:port]      # default 127.0.0.1:3000, server on a fresh seed copy

Writes posts and likes. Needs only the standard library.
"""
import base64
import hashlib
import hmac
import json
import socket
import sys
import threading
import time

HOST, PORT = (sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:3000").split(":")
PORT = int(PORT)
SECRET = b"twelve-dollar-challenge"
failures = 0


def check(name, ok, detail=""):
    global failures
    print(("  ok   " if ok else "  FAIL ") + name + ("" if ok else f"\n       {detail}"))
    failures += not ok


def b64(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def sign(payload, header=None, secret=SECRET):
    h = b64(json.dumps(header or {"alg": "HS256", "typ": "JWT"}).encode())
    p = b64(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
    s = b64(hmac.new(secret, f"{h}.{p}".encode(), hashlib.sha256).digest())
    return f"{h}.{p}.{s}"


NOW = int(time.time())
TOKEN = sign({"sub": "7", "username": "edge_user", "iat": NOW, "exp": NOW + 3600})


def connect():
    s = socket.create_connection((HOST, PORT), timeout=10)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s


def read_response(f):
    """Reads one response from a socket file; returns (status, headers, body)."""
    line = f.readline()
    if not line:
        return None, {}, b""
    status = int(line.split()[1])
    headers = {}
    while True:
        h = f.readline()
        if h in (b"\r\n", b"\n", b""):
            break
        k, v = h.decode().split(":", 1)
        headers[k.strip().lower()] = v.strip()
    body = f.read(int(headers.get("content-length", 0)))
    return status, headers, body


def request(method, path, body=None, token=None, extra=""):
    data = body.encode() if isinstance(body, str) else body
    head = f"{method} {path} HTTP/1.1\r\nHost: x\r\n"
    if token:
        head += f"Authorization: Bearer {token}\r\n"
    if data is not None:
        head += f"Content-Type: application/json\r\nContent-Length: {len(data)}\r\n"
    return head.encode() + extra.encode() + b"\r\n" + (data or b"")


def roundtrip(raw, n=1):
    with connect() as s:
        s.sendall(raw)
        f = s.makefile("rb")
        return [read_response(f) for _ in range(n)]


def api(method, path, body=None, token=None):
    status, _, b = roundtrip(request(method, path, body, token))[0]
    return status, json.loads(b) if b else None


def closed(s):
    s.settimeout(3)
    try:
        return s.recv(1) == b""
    except (ConnectionResetError, socket.timeout):
        return False


print(f"Edge cases against {HOST}:{PORT}")

print("-- framing")
r = roundtrip(request("GET", "/health") + request("GET", "/posts/1") + request("GET", "/feed"), 3)
check("3 pipelined requests answered in order", [x[0] for x in r] == [200, 200, 200]
      and b'"status"' in r[0][2] and b'"post"' in r[1][2] and b'"posts"' in r[2][2], r)

with connect() as s:
    for byte in request("POST", "/posts", '{"body":"sent one byte at a time"}', TOKEN):
        s.send(bytes([byte]))
        time.sleep(0.001)
    st, _, b = read_response(s.makefile("rb"))
    check("request fragmented into single bytes", st == 201 and b"one byte at a time" in b, (st, b))

chunked = (f"POST /posts HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {TOKEN}\r\n"
           "Transfer-Encoding: chunked\r\n\r\n6\r\n{\"body\r\n10\r\n\":\"chunked body\"\r\n1\r\n}\r\n0\r\n\r\n").encode()
st, _, b = roundtrip(chunked)[0]
check("chunked request body", st == 201 and json.loads(b)["post"]["body"] == "chunked body", (st, b))

with connect() as s:
    body = b'{"body":"after 100-continue"}'
    s.sendall(f"POST /posts HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {TOKEN}\r\nExpect: 100-continue\r\n"
              f"Content-Length: {len(body)}\r\n\r\n".encode())
    f = s.makefile("rb")
    interim = f.readline()
    f.readline()
    s.sendall(body)
    st, _, b = read_response(f)
    check("Expect: 100-continue", interim.startswith(b"HTTP/1.1 100") and st == 201, (interim, st, b))

with connect() as s:
    s.sendall(request("GET", "/health", extra="Connection: close\r\n"))
    st, h, _ = read_response(s.makefile("rb"))
    check("Connection: close is honoured", st == 200 and h.get("connection") == "close" and closed(s), h)

with connect() as s:
    s.sendall(b"GET /health HTTP/1.0\r\n\r\n")
    st, _, _ = read_response(s.makefile("rb"))
    check("HTTP/1.0 closes by default", st == 200 and closed(s))

with connect() as s:
    s.sendall(b"GET /health HTTP/1.0\r\nConnection: keep-alive\r\n\r\n")
    f = s.makefile("rb")
    st, h, _ = read_response(f)
    s.sendall(b"GET /health HTTP/1.0\r\nConnection: keep-alive\r\n\r\n")
    st2, _, _ = read_response(f)
    check("HTTP/1.0 keep-alive", st == st2 == 200 and h.get("connection") == "keep-alive")

with connect() as s:
    s.sendall(b"GET /health HTTP/1.1\r\nX-Big: " + b"a" * 20000 + b"\r\n\r\n")
    st, _, _ = read_response(s.makefile("rb"))
    check("oversized header block -> 400 and close", st == 400 and closed(s), st)

with connect() as s:
    s.sendall(b"POST /posts HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n")
    st, _, _ = read_response(s.makefile("rb"))
    check("huge Content-Length -> 413", st == 413, st)

st = roundtrip(b"NONSENSE\r\n\r\n")[0][0]
check("malformed request line -> 400", st == 400, st)

with connect() as s:
    s.sendall(b"GET /posts/1 HT")
    s.shutdown(socket.SHUT_WR)
    check("half request then EOF: connection closed", closed(s))

with connect() as s:
    s.sendall(request("GET", "/health"))
    s.shutdown(socket.SHUT_WR)  # half-close after the request: the response still arrives
    st, _, _ = read_response(s.makefile("rb"))
    check("response delivered after client half-close", st == 200, st)

st, b = api("GET", "/feed?limit=5")
check("query string ignored for routing", st == 200 and len(b["posts"]) == 20, st)
check("unknown method -> 404", roundtrip(request("DELETE", "/posts/1"))[0][0] == 404)
check("/posts/ with empty id -> 404", api("GET", "/posts/")[0] == 404)
check("/posts/1/x -> 404", api("GET", "/posts/1/x")[0] == 404)
check("leading zeros in id", api("GET", "/posts/0001")[1]["post"]["id"] == 1)
check("id beyond int64 -> 404", api("GET", "/posts/99999999999999999999999")[0] == 404)

print("-- JWT")
def auth_status(token):
    return api("POST", "/posts", '{"body":"jwt"}', token)

check("exp as float in the future", auth_status(sign({"sub": "3", "username": "u", "exp": NOW + 99.5}))[0] == 201)
check("no exp claim accepted", auth_status(sign({"sub": "3", "username": "u"}))[0] == 201)
check("exp as string rejected", auth_status(sign({"sub": "3", "username": "u", "exp": "9999999999"}))[1]
      == {"error": "invalid or expired token"})
check("nbf in the future rejected", auth_status(sign({"sub": "3", "username": "u", "nbf": NOW + 600}))[1]
      == {"error": "invalid or expired token"})
check("alg written with escapes", auth_status(sign({"sub": "3", "username": "u"},
      header={"alg": "HS256"}))[0] == 201)
check("HS512 header rejected", auth_status(sign({"sub": "3", "username": "u"}, header={"alg": "HS512"}))[0] == 401)
check("wrong secret rejected", auth_status(sign({"sub": "3", "username": "u"}, secret=b"nope"))[0] == 401)
check("payload not an object", auth_status(sign(b"[1,2]"))[1] == {"error": "invalid or expired token"})
check("payload not JSON", auth_status(sign(b"{nope"))[1] == {"error": "invalid or expired token"})
check("four segments rejected", auth_status(TOKEN + ".x")[0] == 401)
check("sub = 0 -> invalid payload", auth_status(sign({"sub": "0", "username": "u"}))[1]
      == {"error": "invalid token payload"})
check("sub as number -> invalid payload", auth_status(sign({"sub": 3, "username": "u"}))[1]
      == {"error": "invalid token payload"})
check("username as number -> invalid payload", auth_status(sign({"sub": "3", "username": 5}))[1]
      == {"error": "invalid token payload"})
st, b = auth_status(sign({"sub": "3", "username": 'qu"o\\te é\U0001F600'}))
check("username with escapes echoed as author", st == 201 and b["post"]["author"] == 'qu"o\\te é\U0001F600', b)
st, _, b = roundtrip(request("POST", "/posts", '{"body":"x"}', extra=f"Authorization: bearer {TOKEN}\r\n"))[0]
check("lowercase 'bearer' scheme -> missing", st == 401 and json.loads(b) == {"error": "missing bearer token"}, b)

print("-- bodies")
def create(body_json):
    return api("POST", "/posts", body_json, TOKEN)

st, b = create(json.dumps({"body": "\U0001F600" * 500}))
check("500 emoji (500 characters) accepted", st == 201 and b["post"]["body"] == "\U0001F600" * 500, st)
st, b = create(json.dumps({"body": "\U0001F600" * 501}))
check("501 emoji rejected", st == 400 and b == {"error": "body must be at most 500 characters"}, b)
st, b = create('{"body":"\\u00e9\\ud83d\\ude00\\n\\t\\"x\\""}')
check("escapes and surrogate pairs decoded", st == 201 and b["post"]["body"] == 'é😀\n\t"x"', b)
st, b = create('{"body":"\\ud83d"}')
check("lone surrogate -> malformed", st == 400 and b == {"error": "malformed JSON body"}, b)
st, b = create('{"body":"　 ﻿ trimmed   "}')
check("JavaScript whitespace trimmed", st == 201 and b["post"]["body"] == "trimmed", b)
st, b = create('{"body":"　 ﻿"}')
check("only unicode whitespace -> required", b == {"error": "body is required"}, b)
check("duplicate key: last wins", create('{"body":1,"body":"second"}')[1]["post"]["body"] == "second")
check("trailing garbage -> malformed", create('{"body":"x"} x')[1] == {"error": "malformed JSON body"})
check("array body -> required", create('["body"]')[1] == {"error": "body is required"})
check("empty request body -> malformed", create("")[1] == {"error": "malformed JSON body"})
check("invalid UTF-8 -> malformed", create(b'{"body":"\xff"}')[1] == {"error": "malformed JSON body"})
check("control char in string -> malformed", create('{"body":"a\x01"}')[1] == {"error": "malformed JSON body"})
st, b = create('{"body":"ctl \\u0001 <b>&</b>"}')
check("control characters escaped on output", st == 201 and b["post"]["body"] == "ctl \x01 <b>&</b>", b)
st, b = create('{"x":{"y":[1,2,{"z":null}]},"body":"nested ok","n":-1.5e3}')
check("other members of any type skipped", st == 201 and b["post"]["body"] == "nested ok", b)

print("-- likes")
pid = create('{"body":"like race"}')[1]["post"]["id"]
tok = sign({"sub": "4242", "username": "racer", "exp": NOW + 3600})
results = []
def like():
    results.append(api("POST", f"/posts/{pid}/like", None, tok)[0])
threads = [threading.Thread(target=like) for _ in range(16)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check("16 simultaneous duplicate likes: one 201, fifteen 200", sorted(results) == [200] * 15 + [201], results)
check("like_count is 1", api("GET", f"/posts/{pid}")[1]["post"]["like_count"] == 1)

print(f"failures={failures}")
sys.exit(1 if failures else 0)
