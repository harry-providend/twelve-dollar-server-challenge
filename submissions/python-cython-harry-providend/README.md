# Python + Cython (own epoll HTTP server)

| | |
|---|---|
| Language | Python 3.12.3 (Ubuntu 24.04's `python3`) hosting a Cython 3.1.4 extension module |
| Framework | none: an epoll HTTP/1.1 server written in Cython (`fastserver.pyx`) |
| SQLite | 3.53.4, built from source with a small seek patch, linked into the extension (C API, prepared statements) |
| JSON, JWT | written in Cython: strict JSON parser and writer, SHA-256/HMAC, base64url |
| **Nginx or direct** | **Direct**: serves `0.0.0.0:80` itself |

`server.py` is plain Python: it reads the environment, opens the listening socket, raises the file
limit and forks the checkpointer process. Then it calls `fastserver.serve()`, and from there every
request is handled by Cython code compiled to C that never touches a Python object or the GIL.
Cython is a build-time dependency only (`requirements.txt`); at run time the server imports nothing
outside the standard library and its own extension.

## Running it

```bash
sudo bash install.sh   # apt: python3, python3-venv, python3-dev, build-essential, curl, xz-utils
bash build.sh          # SQLite from source, Cython -> C, PGO build (about 40 s)
SQLITE_PATH=... JWT_SECRET=... HOST=0.0.0.0 PORT=80 bash start.sh
```

`build.sh` downloads SQLite's source from the Ubuntu archive (`sqlite3_3.53.4.orig.tar.xz`, checked
against a pinned SHA-256), applies `sqlite-seek.patch` and generates the amalgamation with the Jim Tcl
bundled in the tree.

## Measured

On a 4-core Intel Xeon 2.1 GHz VM, the server pinned to one core with `taskset` and the load generator
on the other three. Fresh seed copy for each run, 2 s warm-up, 15 s runs. `mixed` is the mix used in
the Fortran submission's comparison (`wrk -t2 -c64`, 100 feeds, 100 post reads, 15 likes and 2 new
posts per 217 requests, reading ids from the latest feed). The FastAPI row is
`python-fastapi-cknutson12`, measured the same way on the same machine.

| | mixed req/s | `GET /feed` req/s | `GET /posts/:id` req/s |
|---|---:|---:|---:|
| Python / FastAPI + uvicorn (#1) | 5,936 | 5,952 | 7,339 |
| **Python / Cython (this)** | **79,182** | **60,130** | **159,554** |

Mixed is the median of three runs (76.7k–80.0k). With `wrk -t3 -c128` the mixed load reaches 82–84k
req/s; with `-c64` wrk itself is close to its limit.
Writes alone (likes and posts) run at about 64k/s, held for 60 s with the WAL staying at ~26 MB.

`bench/load.js` with k6 at 15,000 users (the most open connections this test machine allows), 60 s
ramp and 2 min hold, direct mode: p95 0.62 ms, p99 2.9 ms, 0 errors, with the server using about
7% of its core and 33 MB of RSS.

## Optimizations, and why

**No Python on the request path.** `fastserver.pyx` is written against C types only (`cdef ...
noexcept nogil`): epoll loop, HTTP parsing, routing, JSON, JWT and SQLite calls all compile to plain
C. Per request that leaves one `recv`, one `send` and its share of one `epoll_wait`.

**One process, one thread, level-triggered epoll.** The box has one vCPU. Each readable socket gets
exactly one `recv` (no extra call to hit `EAGAIN`). Requests are parsed in a shared 64 KiB buffer; a
connection only owns heap memory while a request is incomplete, so 15,000 idle keep-alive connections
cost about 100 bytes each in user space (15 MB RSS in total).

**Group commit by epoll round.** The first database access of a round opens a transaction, the round
ends with one `COMMIT`, and only then are that round's responses written. Writes are committed in
batches (~20 requests per round under load), every 201 is sent after its commit, and reads share the
transaction instead of each starting their own. If a commit failed, every response of that round
would be replaced by a 500.

**Checkpoints in a separate process, in two phases.** WAL with `synchronous=NORMAL` (rule 6).
Automatic checkpoints are off; once the WAL passes 4,000 frames a forked checkpointer process copies
it back (`PASSIVE`), so the event loop never waits on an `fsync`. Bulk passes run while writes go on.
When only a short tail is left, writes are held back for one final short pass, after which the WAL is
fully copied and SQLite restarts it from the beginning. Without that pause a WAL under constant
writes is never fully checkpointed and grows without bound. Reads are never held back.

**SQLite built for this job.** `SQLITE_THREADSAFE=0` (each process uses SQLite from one thread, so all
mutexes compile away), no memory statistics, and unused features omitted. It is compiled together
with the extension using `-O3 -march=native` and link-time optimization, so calls like
`sqlite3_column_int64` can be inlined into the handlers.

**Profile-guided optimization.** `build.sh` builds an instrumented server, runs it for a few seconds
against a synthetic database with a load-test-like request mix (`pgo_train.py`, which never touches
the real database), and rebuilds with the recorded profile. This alone is +10–20%; SQLite's bytecode
interpreter benefits most. If training fails, the build falls back to a normal optimized build.

**SQLite patched so seeks continue from where the cursor is** (`sqlite-seek.patch`, about 70 lines).
All SQL is unchanged; the patch makes SQLite's own engine do less for the same statements. The
feed's like counts are a correlated subquery, and SQLite closes and reopens its cursor for each of
the 20 rows and searches the likes index from its root every time, although all 20 posts' likes are
next to each other. The patch (1) keeps a read cursor that is reopened on the same b-tree, and its
position, (2) lets an index seek search the current leaf page when that page's first and last
entries bracket the key (stock SQLite does this only on the last page), and (3) does the same for
rowid seeks (stock SQLite only special-cases `rowid + 1`; the feed walks posts newest first). The
feed query goes from 15.7 to 11.0 µs in SQLite. Verified with SQLite's own test suite built from the
patched source with assertions on (`veryquick`, 330,400 tests: the same 28 failures as unpatched
SQLite, all permission tests that fail when run as root) and a differential fuzzer that runs random
seeks, joins, correlated subqueries and writes on 512-byte pages against unpatched SQLite: identical
results, with all three new paths taken tens of thousands of times per run.

**No mmap, a large page cache.** With `mmap_size > 0` in WAL mode, SQLite looks up every page fetch
in the WAL index, even for pages it has cached, and hot pages have many versions in the WAL: that
lookup was 12% of the profile. With mmap off and a 256 MiB page cache, cache hits skip it entirely
(+15% on the mixed load).

**Tried and not kept.** Stripping SQLite with ~35 `SQLITE_OMIT_*` options at amalgamation time (the
grammar is regenerated, so this goes beyond compile flags) removes 23% of its code but measured no
runtime gain: those features live in prepare-time code, and statements are prepared once. Several
option combinations also do not build or cannot read this schema (`OMIT_ATTACH` with
`OMIT_VACUUM`, `OMIT_FOREIGN_KEY` rejects `REFERENCES`), so it adds risk for nothing. Rewriting the
feed SQL (ascending scans, one grouped range count, `IN` lists) was slower or no faster than the
reference query.

**JSON.** Responses are written straight into an output buffer in key order. String escaping checks 8
bytes at a time for `"`, `\` and control characters. The request parser is a strict RFC 8259
validator (malformed JSON is a 400) that decodes escapes and surrogate pairs and only extracts the
members it needs. Bodies are trimmed with JavaScript's `String.prototype.trim()` whitespace set and
measured in characters.

**JWT.** HS256 only, verified on every request (rule 5): the HMAC key's inner and outer padded
blocks are hashed once at startup, so a token costs about four SHA-256 compressions. Signatures are
compared in constant time, `exp`/`nbf` are checked, and `sub` must be a positive integer string.

**SQL.** The reference queries as prepared statements. A like is one statement,
`INSERT ... SELECT ... WHERE EXISTS (post) ON CONFLICT DO NOTHING`; the post-existence check only
runs when it inserted nothing, to tell "already liked" from "no such post".

**Many connections.** The file limit is raised to the hard limit. When descriptors run out, the least
recently active idle connection is closed to admit the new one, and idle connections are closed after
300 s. Clients that pipeline get at most 16 requests served per round, and a client that does not
read its responses stops being served at 4 MiB of unsent output, so one connection cannot starve the
others or grow memory without bound.

## Tests

From the repository root, each against a server started on a fresh seed copy:

```bash
bash test/run.sh submissions/python-cython-harry-providend          # the official 42 checks
python3 submissions/python-cython-harry-providend/tests/edge_cases.py 127.0.0.1:3000
python3 submissions/python-cython-harry-providend/tests/durability.py submissions/python-cython-harry-providend seed/feed.db
python3 submissions/python-cython-harry-providend/tests/idle_connections.py 127.0.0.1:3000 15000 30
submissions/python-cython-harry-providend/.venv/bin/python submissions/python-cython-harry-providend/tests/unit.py
```

- `edge_cases.py` (49 checks): pipelining, a request sent one byte at a time, chunked bodies,
  `Expect: 100-continue`, `Connection: close`, HTTP/1.0, oversized headers and bodies, half-closed
  clients, JWT corner cases (`exp` as float or string, `nbf`, escaped `alg`, wrong secret), Unicode
  length limits and trimming, invalid UTF-8, lone surrogates, and 16 simultaneous duplicate likes.
- `durability.py`: 32 clients write for a while, the server and its checkpointer are `SIGKILL`ed
  mid-load, and after a restart every post and like that got a 201 is in the database (about 375,000
  in a 20 s run); also checks that the WAL stayed bounded.
- `idle_connections.py`: holds N idle keep-alive connections and checks they are all still served.
- `unit.py`: fuzzes the JSON string writer and the HMAC against the standard library.

## License

MIT, under the repo's [license](../../LICENSE).
