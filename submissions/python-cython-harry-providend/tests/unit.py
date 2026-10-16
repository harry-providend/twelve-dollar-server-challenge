"""Fuzzes the extension's own JSON string writer and HMAC-SHA256 against the standard library.

    .venv/bin/python tests/unit.py
"""
import hashlib
import hmac
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastserver  # noqa: E402

rnd = random.Random(1)
alphabet = [chr(c) for c in range(128)] + ["é", "✓", " ", "😀", '"', "\\"]
for _ in range(20000):
    s = "".join(rnd.choice(alphabet) for _ in range(rnd.randrange(40)))
    out = fastserver.json_string_selftest(s.encode())
    assert json.loads(out) == s, (s, out)
    assert out == json.dumps(s, ensure_ascii=False).encode() or b"\\u00" in out, (s, out)
for _ in range(2000):
    key = bytes(rnd.randrange(256) for _ in range(rnd.randrange(150)))
    msg = bytes(rnd.randrange(256) for _ in range(rnd.randrange(300)))
    assert fastserver.hmac_selftest(key, msg) == hmac.new(key, msg, hashlib.sha256).digest()
print("unit: JSON string writer and HMAC-SHA256 match the standard library")
