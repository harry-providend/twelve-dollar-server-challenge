# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, overflowcheck=False, infer_types=True
"""The hot path of the server, compiled by Cython to C.

Python (server.py) sets the process up: environment, listening socket, file limits, the checkpointer
process. It then calls serve(), which never returns. Everything a request touches runs below without
the GIL and without Python objects: an epoll loop, an HTTP/1.1 parser, the router, HS256 JWT checks
(SHA-256/HMAC written here), a strict JSON parser, a JSON writer and SQLite's C API with prepared
statements.

One event-loop round = one epoll_wait() batch. The first database access of a round opens a
transaction, the round ends with COMMIT, and only then are the round's responses written to the
sockets. So writes are group-committed, every 201 is sent after its row is committed, and no response
leaves before the data it shows is durable.
"""

from libc.stdint cimport uint8_t, uint32_t, uint64_t, int64_t
from libc.string cimport memcpy, memmove, memcmp, memchr, memset, strlen
from libc.stdlib cimport malloc, calloc, realloc, free, strtod, abort
from libc.errno cimport errno, EAGAIN, EINTR, EMFILE, ENFILE, ENOBUFS, ENOMEM, ECONNABORTED

cdef extern from "<unistd.h>" nogil:
    ssize_t read(int, void*, size_t)
    ssize_t write(int, const void*, size_t)
    int close(int)

cdef extern from "<time.h>" nogil:
    ctypedef long time_t
    ctypedef int clockid_t
    cdef struct timespec:
        time_t tv_sec
        long tv_nsec
    cdef struct tm:
        int tm_sec
    int clock_gettime(clockid_t, timespec*)
    tm* gmtime_r(const time_t*, tm*)
    size_t strftime(char*, size_t, const char*, const tm*)
    enum:
        CLOCK_MONOTONIC
        CLOCK_MONOTONIC_COARSE
        CLOCK_REALTIME_COARSE

cdef extern from "<sys/socket.h>" nogil:
    ctypedef uint32_t socklen_t
    cdef struct sockaddr:
        pass
    int accept4(int, sockaddr*, socklen_t*, int)
    int setsockopt(int, int, int, const void*, socklen_t)
    ssize_t send(int, const void*, size_t, int)
    ssize_t recv(int, void*, size_t, int)
    enum:
        SOCK_NONBLOCK
        SOCK_CLOEXEC
        MSG_NOSIGNAL

cdef extern from "<netinet/in.h>" nogil:
    enum:
        IPPROTO_TCP

cdef extern from "<netinet/tcp.h>" nogil:
    enum:
        TCP_NODELAY

cdef extern from "<sys/epoll.h>" nogil:
    ctypedef union epoll_data_t:
        void* ptr
        int fd
        uint32_t u32
        uint64_t u64
    cdef struct epoll_event:
        uint32_t events
        epoll_data_t data
    int epoll_create1(int)
    int epoll_ctl(int, int, int, epoll_event*)
    int epoll_wait(int, epoll_event*, int, int)
    enum:
        EPOLLIN
        EPOLLOUT
        EPOLLERR
        EPOLLHUP
        EPOLL_CTL_ADD
        EPOLL_CTL_MOD
        EPOLL_CTL_DEL
        EPOLL_CLOEXEC

cdef extern from "<sys/prctl.h>" nogil:
    int prctl(int, ...)
    enum:
        PR_SET_PDEATHSIG

cdef extern from "<signal.h>" nogil:
    ctypedef void (*sighandler_t)(int) noexcept nogil
    sighandler_t signal(int, sighandler_t)
    enum:
        SIGKILL
        SIGTERM
        SIGINT

cdef extern from "sqlite3.h" nogil:
    ctypedef struct sqlite3
    ctypedef struct sqlite3_stmt
    ctypedef long long sqlite3_int64
    ctypedef void (*sqlite3_destructor_type)(void*)
    ctypedef int (*wal_hook_t)(void*, sqlite3*, const char*, int) noexcept nogil
    sqlite3_destructor_type SQLITE_STATIC
    int sqlite3_config(int, ...)
    int sqlite3_initialize()
    const char* sqlite3_libversion()
    int sqlite3_open_v2(const char*, sqlite3**, int, const char*)
    int sqlite3_close(sqlite3*)
    int sqlite3_exec(sqlite3*, const char*, void*, void*, char**)
    int sqlite3_prepare_v3(sqlite3*, const char*, int, unsigned int, sqlite3_stmt**, const char**)
    int sqlite3_step(sqlite3_stmt*)
    int sqlite3_reset(sqlite3_stmt*)
    int sqlite3_bind_int64(sqlite3_stmt*, int, sqlite3_int64)
    int sqlite3_bind_text(sqlite3_stmt*, int, const char*, int, sqlite3_destructor_type)
    sqlite3_int64 sqlite3_column_int64(sqlite3_stmt*, int)
    const unsigned char* sqlite3_column_text(sqlite3_stmt*, int)
    int sqlite3_column_bytes(sqlite3_stmt*, int)
    int sqlite3_changes(sqlite3*)
    const char* sqlite3_errmsg(sqlite3*)
    int sqlite3_get_autocommit(sqlite3*)
    int sqlite3_busy_timeout(sqlite3*, int)
    int sqlite3_wal_autocheckpoint(sqlite3*, int)
    int sqlite3_wal_checkpoint_v2(sqlite3*, const char*, int, int*, int*)
    void* sqlite3_wal_hook(sqlite3*, wal_hook_t, void*)
    enum:
        SQLITE_OK
        SQLITE_ROW
        SQLITE_DONE
        SQLITE_OPEN_READWRITE
        SQLITE_OPEN_NOMUTEX
        SQLITE_PREPARE_PERSISTENT
        SQLITE_CONFIG_SINGLETHREAD
        SQLITE_CONFIG_MEMSTATUS
        SQLITE_CHECKPOINT_PASSIVE


# --- tuning -----------------------------------------------------------------------------------------

cdef enum:
    RDBUF_SIZE = 65536          # shared receive buffer; a connection only owns memory while a request is partial
    MAX_EVENTS = 256            # epoll batch = one round = one transaction
    MAX_HEADER = 16384
    MAX_BODY = 65536            # a 500-character body is at most ~6 KiB of JSON
    RBUF_LIMIT = MAX_HEADER + MAX_BODY + 65536
    MAX_PIPELINE = 16           # requests served per connection per round; the rest wait a round
    BACKPRESSURE = 4194304      # stop serving a client that is not reading its responses
    IDLE_TIMEOUT = 300          # seconds; Nginx needs >= 65 s, k6 users think for at most 15 s
    CKPT_FRAMES = 4000          # WAL frames (4 KiB pages) before the checkpointer process is asked to run
    CKPT_TAIL = 400             # frames left for the final pass, the only one that holds writes back
    CKPT_PASSES = 4             # bulk passes before the final one regardless of the tail
    USERNAME_MAX = 4096

# checkpoint states
cdef enum:
    CK_IDLE = 0
    CK_BULK = 1                 # bulk passes run while writes continue
    CK_FINAL_NEXT = 2           # writes are held; the final pass starts after this round's COMMIT
    CK_FINAL = 3                # final pass running; when it ends the WAL is fully copied back

# connection flags
cdef enum:
    CF_CLOSE = 1                # close once the pending responses are written
    CF_DEFERRED = 2             # holds a write request until the running checkpoint finishes
    CF_IN_ROUND = 4             # has responses in this round's output arena
    CF_CLOSE_QUEUED = 8
    CF_DEAD = 16                # the socket failed; drop pending output
    CF_KEEPALIVE10 = 32         # HTTP/1.0 request that asked for keep-alive
    CF_CONTINUE_SENT = 64       # "100 Continue" already sent for the current request
    CF_PENDING = 128            # has buffered requests left for the next round (MAX_PIPELINE)

# JSON value types
cdef enum:
    JT_NONE = 0
    JT_STR = 1
    JT_NUM = 2
    JT_TRUE = 3
    JT_FALSE = 4
    JT_NULL = 5
    JT_OBJ = 6
    JT_ARR = 7

cdef struct Conn:
    int fd
    uint32_t flags
    uint32_t ev                 # events currently registered with epoll (0 = not registered)
    int64_t last_active         # monotonic seconds
    char* rbuf                  # unprocessed input (partial or deferred request)
    size_t rlen
    size_t rcap
    char* wbuf                  # output the socket did not accept yet
    size_t wlen
    size_t woff
    size_t wcap
    Conn* prev                  # LRU list, least recently active first
    Conn* next

cdef struct JField:
    const char* key
    int klen
    int type
    const char* vs              # value start (the opening quote for strings)
    const char* ve              # value end (one past the closing quote for strings)
    int esc                     # string contains backslash escapes


# --- global state -----------------------------------------------------------------------------------

cdef int g_ep = -1
cdef int g_stop = 0             # set by SIGTERM/SIGINT: leave the loop between rounds
cdef int g_lfd = -1
cdef int g_ckreq_fd = -1
cdef int g_ckdone_fd = -1
cdef bint g_ckpt_disabled = False
cdef bint g_defer_writes = False # the final checkpoint pass is running (or about to): hold writes back
cdef int g_ck_state = 0         # CK_IDLE, CK_BULK, CK_FINAL_NEXT, CK_FINAL
cdef int g_ck_passes = 0
cdef int g_wal_frames = 0       # frames in the WAL after the last commit
cdef bint g_listen_paused = False
cdef int64_t g_listen_resume_ms = 0

cdef Conn** g_conns = NULL
cdef int g_max_fds = 0
cdef int g_nconns = 0
cdef int g_max_conns = 0
cdef Conn* g_lru_head = NULL
cdef Conn* g_lru_tail = NULL

cdef Conn** g_deferred = NULL
cdef int g_ndeferred = 0
cdef int g_deferred_cap = 0
cdef Conn** g_pending = NULL
cdef Conn** g_pending_spare = NULL
cdef int g_npending = 0
cdef int g_pending_cap = 0

cdef int64_t g_now = 0          # monotonic seconds
cdef int64_t g_now_ms = 0
cdef int64_t g_start = 0
cdef int64_t g_last_sweep = 0
cdef int64_t g_wall = 0         # unix time, seconds
cdef char g_date[64]
cdef size_t g_date_len = 0

cdef char* g_rdbuf = NULL

# output arena: every response of the round, written to the sockets after COMMIT
cdef char* g_out = NULL
cdef size_t g_out_len = 0
cdef size_t g_out_cap = 0
cdef Conn** g_seg_conn = NULL
cdef size_t* g_seg_start = NULL
cdef size_t* g_seg_end = NULL
cdef int* g_seg_nreq = NULL
cdef int g_nseg = 0
cdef int g_seg_cap = 0
cdef Conn** g_closeq = NULL
cdef int g_nclose = 0

cdef char* g_bb = NULL          # response body scratch
cdef size_t g_bb_cap = 0
cdef char* g_tmp = NULL         # unescaped POST body
cdef char* g_chunk = NULL       # de-chunked request body
cdef uint8_t* g_jwt = NULL      # decoded JWT parts
cdef char g_uname[USERNAME_MAX]

cdef sqlite3* g_db = NULL
cdef sqlite3_stmt* st_begin = NULL
cdef sqlite3_stmt* st_commit = NULL
cdef sqlite3_stmt* st_rollback = NULL
cdef sqlite3_stmt* st_one = NULL
cdef sqlite3_stmt* st_feed = NULL
cdef sqlite3_stmt* st_post = NULL
cdef sqlite3_stmt* st_ins_post = NULL
cdef sqlite3_stmt* st_ins_like = NULL
cdef sqlite3_stmt* st_exists = NULL
cdef bint g_in_txn = False
cdef bint g_round_failed = False

cdef uint32_t K256[64]
cdef uint32_t g_istate[8]
cdef uint32_t g_ostate[8]
cdef uint8_t B64D[256]
cdef uint8_t ESC[256]
cdef const char* HEXD = b"0123456789abcdef"
cdef const char* B64E = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

cdef const char* SQL_FEED = (
    b"SELECT p.id, p.body, p.created_at, u.username, "
    b"(SELECT count(*) FROM likes l WHERE l.post_id = p.id) "
    b"FROM posts p JOIN users u ON u.id = p.user_id "
    b"ORDER BY p.created_at DESC, p.id DESC LIMIT 20")
cdef const char* SQL_POST = (
    b"SELECT p.id, p.body, p.created_at, u.username, "
    b"(SELECT count(*) FROM likes l WHERE l.post_id = p.id) "
    b"FROM posts p JOIN users u ON u.id = p.user_id WHERE p.id = ?1")
cdef const char* SQL_INS_POST = b"INSERT INTO posts (user_id, body) VALUES (?1, ?2) RETURNING id, created_at"
# Inserts only when the post exists; no change means "already liked" or "no such post".
cdef const char* SQL_INS_LIKE = (
    b"INSERT INTO likes (user_id, post_id) SELECT ?1, ?2 "
    b"WHERE EXISTS (SELECT 1 FROM posts WHERE id = ?2) ON CONFLICT (user_id, post_id) DO NOTHING")
cdef const char* SQL_EXISTS = b"SELECT 1 FROM posts WHERE id = ?1"


# --- small helpers ----------------------------------------------------------------------------------

cdef inline char* cpy(char* d, const char* s, size_t n) noexcept nogil:
    memcpy(d, s, n)
    return d + n

cdef inline char* cps(char* d, const char* s) noexcept nogil:
    cdef size_t n = strlen(s)
    memcpy(d, s, n)
    return d + n

cdef inline char* put_u64(char* d, uint64_t v) noexcept nogil:
    cdef char tmp[24]
    cdef int i = 24
    if v == 0:
        d[0] = c'0'
        return d + 1
    while v:
        i -= 1
        tmp[i] = <char>(48 + v % 10)
        v //= 10
    memcpy(d, tmp + i, 24 - i)
    return d + (24 - i)

cdef inline char* put_i64(char* d, int64_t v) noexcept nogil:
    if v < 0:
        d[0] = c'-'
        return put_u64(d + 1, <uint64_t>(-(v + 1)) + 1)
    return put_u64(d, <uint64_t>v)

cdef inline void* xrealloc(void* p, size_t n) noexcept nogil:
    cdef void* q = realloc(p, n)
    if q == NULL:
        abort()
    return q

cdef inline bint word_plain(uint64_t v) noexcept nogil:
    """True if none of the 8 bytes in v needs escaping in a JSON string (control, '"' or '\\')."""
    cdef uint64_t ones = 0x0101010101010101ULL, high = 0x8080808080808080ULL
    cdef uint64_t q = v ^ (ones * 0x22)
    cdef uint64_t b = v ^ (ones * 0x5C)
    return (((v - ones * 0x20) & ~v) | ((q - ones) & ~q) | ((b - ones) & ~b)) & high == 0

cdef inline char* put_json_str(char* d, const uint8_t* s, size_t n) noexcept nogil:
    """Writes s as a JSON string. d needs room for 6 * n + 2 bytes."""
    cdef size_t i = 0, j
    cdef uint64_t v
    cdef uint8_t ch, e
    d[0] = c'"'
    d += 1
    while True:
        while i + 8 <= n:          # 8 bytes at a time while nothing needs escaping
            memcpy(&v, s + i, 8)
            if not word_plain(v):
                break
            memcpy(d, &v, 8)
            d += 8
            i += 8
        j = i + 8 if i + 8 <= n else n
        while i < j:
            ch = s[i]
            e = ESC[ch]
            if e == 0:
                d[0] = <char>ch
                d += 1
            elif e == c'u':
                d[0] = c'\\'
                d[1] = c'u'
                d[2] = c'0'
                d[3] = c'0'
                d[4] = HEXD[ch >> 4]
                d[5] = HEXD[ch & 15]
                d += 6
            else:
                d[0] = c'\\'
                d[1] = <char>e
                d += 2
            i += 1
        if i >= n:
            break
    d[0] = c'"'
    return d + 1

cdef inline char* bb_ensure(char* p, size_t need) noexcept nogil:
    """Grows the body scratch buffer so that `need` bytes fit after p; returns p's new address."""
    global g_bb, g_bb_cap
    cdef size_t used = p - g_bb
    cdef size_t cap
    if used + need > g_bb_cap:
        cap = g_bb_cap * 2
        while cap < used + need:
            cap *= 2
        g_bb = <char*>xrealloc(g_bb, cap)
        g_bb_cap = cap
        return g_bb + used
    return p


# --- SHA-256, HMAC, base64url -----------------------------------------------------------------------

cdef inline uint32_t ror(uint32_t x, uint32_t n) noexcept nogil:
    return (x >> n) | (x << (32 - n))

cdef void sha256_block(uint32_t* st, const uint8_t* b) noexcept nogil:
    cdef uint32_t w[64]
    cdef uint32_t a, bb, c, d, e, f, g, h, s0, s1, t1, t2
    cdef int i
    for i in range(16):
        w[i] = ((<uint32_t>b[4 * i] << 24) | (<uint32_t>b[4 * i + 1] << 16)
                | (<uint32_t>b[4 * i + 2] << 8) | <uint32_t>b[4 * i + 3])
    for i in range(16, 64):
        s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3)
        s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10)
        w[i] = w[i - 16] + s0 + w[i - 7] + s1
    a = st[0]; bb = st[1]; c = st[2]; d = st[3]
    e = st[4]; f = st[5]; g = st[6]; h = st[7]
    for i in range(64):
        s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)
        t1 = h + s1 + ((e & f) ^ (~e & g)) + K256[i] + w[i]
        s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)
        t2 = s0 + ((a & bb) ^ (a & c) ^ (bb & c))
        h = g; g = f; f = e; e = d + t1
        d = c; c = bb; bb = a; a = t1 + t2
    st[0] += a; st[1] += bb; st[2] += c; st[3] += d
    st[4] += e; st[5] += f; st[6] += g; st[7] += h

cdef inline void store_be(const uint32_t* st, uint8_t* out) noexcept nogil:
    cdef int i
    for i in range(8):
        out[4 * i] = <uint8_t>(st[i] >> 24)
        out[4 * i + 1] = <uint8_t>(st[i] >> 16)
        out[4 * i + 2] = <uint8_t>(st[i] >> 8)
        out[4 * i + 3] = <uint8_t>st[i]

cdef inline void put_bitlen(uint8_t* blk, uint64_t bits) noexcept nogil:
    cdef int i
    for i in range(8):
        blk[63 - i] = <uint8_t>(bits >> (8 * i))

cdef void sha256_tail(uint32_t* st, const uint8_t* msg, size_t n, uint64_t prefix_len) noexcept nogil:
    """Hashes msg into st (which already absorbed prefix_len bytes) and applies the final padding."""
    cdef uint8_t blk[64]
    cdef size_t full = n // 64, rem, i
    for i in range(full):
        sha256_block(st, msg + 64 * i)
    rem = n - full * 64
    memcpy(blk, msg + full * 64, rem)
    blk[rem] = 0x80
    if rem + 1 > 56:
        memset(blk + rem + 1, 0, 63 - rem)
        sha256_block(st, blk)
        memset(blk, 0, 56)
    else:
        memset(blk + rem + 1, 0, 55 - rem)
    put_bitlen(blk, (prefix_len + n) * 8)
    sha256_block(st, blk)

cdef void hmac_sha256(const uint8_t* msg, size_t n, uint8_t* out) noexcept nogil:
    cdef uint32_t st[8]
    cdef uint8_t inner[32]
    memcpy(st, g_istate, 32)
    sha256_tail(st, msg, n, 64)
    store_be(st, inner)
    memcpy(st, g_ostate, 32)
    sha256_tail(st, inner, 32, 64)
    store_be(st, out)

cdef void hmac_init(const uint8_t* key, size_t klen) noexcept nogil:
    """Precomputes the HMAC inner/outer key blocks (every token is still fully verified)."""
    cdef uint8_t k[64]
    cdef uint8_t pad[64]
    cdef uint32_t st[8]
    cdef int i
    memset(k, 0, 64)
    if klen > 64:
        sha256_init(st)
        sha256_tail(st, key, klen, 0)
        store_be(st, k)
    else:
        memcpy(k, key, klen)
    for i in range(64):
        pad[i] = k[i] ^ 0x36
    sha256_init(g_istate)
    sha256_block(g_istate, pad)
    for i in range(64):
        pad[i] = k[i] ^ 0x5c
    sha256_init(g_ostate)
    sha256_block(g_ostate, pad)

cdef inline void sha256_init(uint32_t* st) noexcept nogil:
    st[0] = 0x6a09e667; st[1] = 0xbb67ae85; st[2] = 0x3c6ef372; st[3] = 0xa54ff53a
    st[4] = 0x510e527f; st[5] = 0x9b05688c; st[6] = 0x1f83d9ab; st[7] = 0x5be0cd19

cdef inline ssize_t b64url_decode(const char* s, size_t n, uint8_t* out) noexcept nogil:
    """Unpadded base64url; -1 on any other character or an impossible length."""
    cdef uint32_t acc = 0
    cdef int bits = 0
    cdef size_t i, o = 0
    cdef uint8_t v
    if n % 4 == 1:
        return -1
    for i in range(n):
        v = B64D[<uint8_t>s[i]]
        if v == 255:
            return -1
        acc = ((acc << 6) | v) & 0xFFFFFF
        bits += 6
        if bits >= 8:
            bits -= 8
            out[o] = <uint8_t>(acc >> bits)
            o += 1
    return <ssize_t>o

cdef inline void b64url_encode32(const uint8_t* d, char* out) noexcept nogil:
    """32 bytes -> 43 characters."""
    cdef int i, o = 0
    cdef uint32_t v
    for i in range(0, 30, 3):
        v = (<uint32_t>d[i] << 16) | (<uint32_t>d[i + 1] << 8) | d[i + 2]
        out[o] = B64E[(v >> 18) & 63]
        out[o + 1] = B64E[(v >> 12) & 63]
        out[o + 2] = B64E[(v >> 6) & 63]
        out[o + 3] = B64E[v & 63]
        o += 4
    v = (<uint32_t>d[30] << 16) | (<uint32_t>d[31] << 8)
    out[o] = B64E[(v >> 18) & 63]
    out[o + 1] = B64E[(v >> 12) & 63]
    out[o + 2] = B64E[(v >> 6) & 63]


# --- JSON parsing (RFC 8259, strict) ----------------------------------------------------------------

cdef inline const char* j_ws(const char* p, const char* e) noexcept nogil:
    while p < e and (p[0] == c' ' or p[0] == c'\n' or p[0] == c'\r' or p[0] == c'\t'):
        p += 1
    return p

cdef inline int hexv(char ch) noexcept nogil:
    if ch >= c'0' and ch <= c'9':
        return ch - 48
    if ch >= c'a' and ch <= c'f':
        return ch - 87
    if ch >= c'A' and ch <= c'F':
        return ch - 55
    return -1

cdef inline int hex4(const char* p) noexcept nogil:
    cdef int a = hexv(p[0]), b = hexv(p[1]), c = hexv(p[2]), d = hexv(p[3])
    if a < 0 or b < 0 or c < 0 or d < 0:
        return -1
    return (a << 12) | (b << 8) | (c << 4) | d

cdef inline int utf8_seq(const uint8_t* p, const uint8_t* e) noexcept nogil:
    """Length of the valid UTF-8 sequence at p (lead byte >= 0x80), 0 if invalid."""
    cdef uint8_t c0 = p[0], c1
    if c0 < 0xC2:
        return 0
    if c0 < 0xE0:
        if p + 1 < e and (p[1] & 0xC0) == 0x80:
            return 2
        return 0
    if c0 < 0xF0:
        if p + 2 >= e:
            return 0
        c1 = p[1]
        if (c1 & 0xC0) != 0x80 or (p[2] & 0xC0) != 0x80:
            return 0
        if (c0 == 0xE0 and c1 < 0xA0) or (c0 == 0xED and c1 >= 0xA0):
            return 0
        return 3
    if c0 < 0xF5:
        if p + 3 >= e:
            return 0
        c1 = p[1]
        if (c1 & 0xC0) != 0x80 or (p[2] & 0xC0) != 0x80 or (p[3] & 0xC0) != 0x80:
            return 0
        if (c0 == 0xF0 and c1 < 0x90) or (c0 == 0xF4 and c1 >= 0x90):
            return 0
        return 4
    return 0

cdef const char* j_str(const char* p, const char* e, int* esc) noexcept nogil:
    """p is at the opening quote; returns the position after the closing quote, NULL if invalid."""
    cdef uint8_t ch
    cdef int n
    p += 1
    while p < e:
        ch = <uint8_t>p[0]
        if ch == c'"':
            return p + 1
        if ch == c'\\':
            esc[0] = 1
            if p + 1 >= e:
                return NULL
            ch = <uint8_t>p[1]
            if ch == c'u':
                if p + 6 > e or hex4(p + 2) < 0:
                    return NULL
                p += 6
            elif (ch == c'"' or ch == c'\\' or ch == c'/' or ch == c'b' or ch == c'f' or ch == c'n'
                  or ch == c'r' or ch == c't'):
                p += 2
            else:
                return NULL
        elif ch < 0x20:
            return NULL
        elif ch < 0x80:
            p += 1
        else:
            n = utf8_seq(<const uint8_t*>p, <const uint8_t*>e)
            if n == 0:
                return NULL
            p += n
    return NULL

cdef const char* j_num(const char* p, const char* e) noexcept nogil:
    if p < e and p[0] == c'-':
        p += 1
    if p >= e:
        return NULL
    if p[0] == c'0':
        p += 1
    elif p[0] >= c'1' and p[0] <= c'9':
        while p < e and p[0] >= c'0' and p[0] <= c'9':
            p += 1
    else:
        return NULL
    if p < e and p[0] == c'.':
        p += 1
        if p >= e or not (p[0] >= c'0' and p[0] <= c'9'):
            return NULL
        while p < e and p[0] >= c'0' and p[0] <= c'9':
            p += 1
    if p < e and (p[0] == c'e' or p[0] == c'E'):
        p += 1
        if p < e and (p[0] == c'+' or p[0] == c'-'):
            p += 1
        if p >= e or not (p[0] >= c'0' and p[0] <= c'9'):
            return NULL
        while p < e and p[0] >= c'0' and p[0] <= c'9':
            p += 1
    return p

cdef inline const char* j_lit(const char* p, const char* e, const char* lit, size_t n) noexcept nogil:
    if <size_t>(e - p) >= n and memcmp(p, lit, n) == 0:
        return p + n
    return NULL

cdef const char* j_value(const char* p, const char* e, int depth) noexcept nogil:
    """Skips one JSON value starting at p (no leading whitespace); NULL if invalid."""
    cdef int dummy = 0
    cdef char ch
    if p >= e or depth > 512:
        return NULL
    ch = p[0]
    if ch == c'"':
        return j_str(p, e, &dummy)
    if ch == c'{':
        p = j_ws(p + 1, e)
        if p < e and p[0] == c'}':
            return p + 1
        while True:
            if p >= e or p[0] != c'"':
                return NULL
            p = j_str(p, e, &dummy)
            if p == NULL:
                return NULL
            p = j_ws(p, e)
            if p >= e or p[0] != c':':
                return NULL
            p = j_value(j_ws(p + 1, e), e, depth + 1)
            if p == NULL:
                return NULL
            p = j_ws(p, e)
            if p >= e:
                return NULL
            if p[0] == c',':
                p = j_ws(p + 1, e)
            elif p[0] == c'}':
                return p + 1
            else:
                return NULL
    if ch == c'[':
        p = j_ws(p + 1, e)
        if p < e and p[0] == c']':
            return p + 1
        while True:
            p = j_value(p, e, depth + 1)
            if p == NULL:
                return NULL
            p = j_ws(p, e)
            if p >= e:
                return NULL
            if p[0] == c',':
                p = j_ws(p + 1, e)
            elif p[0] == c']':
                return p + 1
            else:
                return NULL
    if ch == c't':
        return j_lit(p, e, b"true", 4)
    if ch == c'f':
        return j_lit(p, e, b"false", 5)
    if ch == c'n':
        return j_lit(p, e, b"null", 4)
    return j_num(p, e)

cdef ssize_t j_unescape(const char* s, const char* e, char* out) noexcept nogil:
    """Decodes validated JSON string content s..e (without quotes) to UTF-8; -1 on a lone surrogate."""
    cdef char* o = out
    cdef const char* bs
    cdef int cp, lo
    cdef char ch
    while s < e:
        bs = <const char*>memchr(s, c'\\', e - s)
        if bs == NULL:
            memcpy(o, s, e - s)
            o += e - s
            break
        memcpy(o, s, bs - s)
        o += bs - s
        s = bs
        ch = s[1]
        if ch == c'u':
            cp = hex4(s + 2)
            s += 6
            if cp >= 0xD800 and cp <= 0xDBFF:
                if e - s >= 6 and s[0] == c'\\' and s[1] == c'u':
                    lo = hex4(s + 2)
                    if lo < 0xDC00 or lo > 0xDFFF:
                        return -1
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                    s += 6
                else:
                    return -1
            elif cp >= 0xDC00 and cp <= 0xDFFF:
                return -1
            if cp < 0x80:
                o[0] = <char>cp
                o += 1
            elif cp < 0x800:
                o[0] = <char>(0xC0 | (cp >> 6))
                o[1] = <char>(0x80 | (cp & 0x3F))
                o += 2
            elif cp < 0x10000:
                o[0] = <char>(0xE0 | (cp >> 12))
                o[1] = <char>(0x80 | ((cp >> 6) & 0x3F))
                o[2] = <char>(0x80 | (cp & 0x3F))
                o += 3
            else:
                o[0] = <char>(0xF0 | (cp >> 18))
                o[1] = <char>(0x80 | ((cp >> 12) & 0x3F))
                o[2] = <char>(0x80 | ((cp >> 6) & 0x3F))
                o[3] = <char>(0x80 | (cp & 0x3F))
                o += 4
        else:
            if ch == c'b':
                o[0] = 8
            elif ch == c'f':
                o[0] = 12
            elif ch == c'n':
                o[0] = 10
            elif ch == c'r':
                o[0] = 13
            elif ch == c't':
                o[0] = 9
            else:
                o[0] = ch           # " \ /
            o += 1
            s += 2
    return o - out

cdef inline bint key_eq(const char* ks, const char* ke, int kesc, const char* key, int klen) noexcept nogil:
    cdef char buf[64]
    cdef ssize_t n
    if not kesc:
        return ke - ks == klen and memcmp(ks, key, klen) == 0
    if ke - ks > 64:
        return False
    n = j_unescape(ks, ke, buf)
    return n == klen and memcmp(buf, key, klen) == 0

cdef int j_fields(const char* p, const char* e, JField* f, int nf) noexcept nogil:
    """Validates a whole JSON document and, if it is an object, records the members named in f
    (the last duplicate wins). Returns 1 for an object, 0 for any other valid document, -1 if invalid."""
    cdef const char* ks
    cdef const char* vs
    cdef const char* q
    cdef int kesc, vesc, t, i
    p = j_ws(p, e)
    if p >= e:
        return -1
    if p[0] != c'{':
        q = j_value(p, e, 0)
        if q == NULL or j_ws(q, e) != e:
            return -1
        return 0
    p = j_ws(p + 1, e)
    if p < e and p[0] == c'}':
        return 1 if j_ws(p + 1, e) == e else -1
    while True:
        if p >= e or p[0] != c'"':
            return -1
        kesc = 0
        ks = p + 1
        q = j_str(p, e, &kesc)
        if q == NULL:
            return -1
        p = j_ws(q, e)
        if p >= e or p[0] != c':':
            return -1
        vs = j_ws(p + 1, e)
        if vs >= e:
            return -1
        vesc = 0
        if vs[0] == c'"':
            t = JT_STR
            p = j_str(vs, e, &vesc)
        else:
            if vs[0] == c'{':
                t = JT_OBJ
            elif vs[0] == c'[':
                t = JT_ARR
            elif vs[0] == c't':
                t = JT_TRUE
            elif vs[0] == c'f':
                t = JT_FALSE
            elif vs[0] == c'n':
                t = JT_NULL
            else:
                t = JT_NUM
            p = j_value(vs, e, 1)
        if p == NULL:
            return -1
        for i in range(nf):
            if key_eq(ks, q - 1, kesc, f[i].key, f[i].klen):
                f[i].type = t
                f[i].vs = vs
                f[i].ve = p
                f[i].esc = vesc
        p = j_ws(p, e)
        if p >= e:
            return -1
        if p[0] == c',':
            p = j_ws(p + 1, e)
        elif p[0] == c'}':
            return 1 if j_ws(p + 1, e) == e else -1
        else:
            return -1

cdef inline void jf_init(JField* f, const char* key) noexcept nogil:
    f.key = key
    f.klen = <int>strlen(key)
    f.type = JT_NONE
    f.vs = NULL
    f.ve = NULL
    f.esc = 0

cdef inline ssize_t jf_string(JField* f, char* out, size_t cap) noexcept nogil:
    """Decoded value of a string field into out; -1 if it does not fit or has a lone surrogate."""
    cdef const char* s = f.vs + 1
    cdef const char* e = f.ve - 1
    if <size_t>(e - s) > cap:
        return -1
    if not f.esc:
        memcpy(out, s, e - s)
        return e - s
    return j_unescape(s, e, out)

cdef inline bint jf_number(JField* f, double* out) noexcept nogil:
    cdef char buf[64]
    cdef size_t n = f.ve - f.vs
    if n >= 64:
        out[0] = strtod(b"1e400", NULL) if f.vs[0] != c'-' else -strtod(b"1e400", NULL)
        return True
    memcpy(buf, f.vs, n)
    buf[n] = 0
    out[0] = strtod(buf, NULL)
    return True


# --- JWT --------------------------------------------------------------------------------------------

cdef enum:
    AUTH_OK = 0
    AUTH_INVALID = 1        # invalid or expired token
    AUTH_PAYLOAD = 2        # invalid token payload

cdef int verify_jwt(const char* tok, size_t n, int64_t* uid, size_t* ulen) noexcept nogil:
    cdef const char* d1
    cdef const char* d2
    cdef const char* end = tok + n
    cdef ssize_t m
    cdef JField hf[1]
    cdef JField pf[4]
    cdef uint8_t mac[32]
    cdef char enc[43]
    cdef char sub[32]
    cdef uint8_t diff = 0
    cdef double t
    cdef uint64_t v
    cdef int i
    if n > MAX_HEADER:
        return AUTH_INVALID
    d1 = <const char*>memchr(tok, c'.', n)
    if d1 == NULL:
        return AUTH_INVALID
    d2 = <const char*>memchr(d1 + 1, c'.', end - d1 - 1)
    if d2 == NULL or memchr(d2 + 1, c'.', end - d2 - 1) != NULL:
        return AUTH_INVALID
    # header: {"alg":"HS256",...}
    m = b64url_decode(tok, d1 - tok, g_jwt)
    if m < 0:
        return AUTH_INVALID
    jf_init(&hf[0], b"alg")
    if j_fields(<const char*>g_jwt, <const char*>g_jwt + m, hf, 1) != 1 or hf[0].type != JT_STR:
        return AUTH_INVALID
    m = jf_string(&hf[0], sub, 32)
    if m != 5 or memcmp(sub, b"HS256", 5) != 0:
        return AUTH_INVALID
    # signature over "<header>.<payload>", compared in constant time
    if end - d2 - 1 != 43:
        return AUTH_INVALID
    hmac_sha256(<const uint8_t*>tok, d2 - tok, mac)
    b64url_encode32(mac, enc)
    for i in range(43):
        diff |= <uint8_t>(enc[i] ^ d2[1 + i])
    if diff != 0:
        return AUTH_INVALID
    # payload
    m = b64url_decode(d1 + 1, d2 - d1 - 1, g_jwt)
    if m < 0:
        return AUTH_INVALID
    jf_init(&pf[0], b"sub")
    jf_init(&pf[1], b"username")
    jf_init(&pf[2], b"exp")
    jf_init(&pf[3], b"nbf")
    if j_fields(<const char*>g_jwt, <const char*>g_jwt + m, pf, 4) != 1:
        return AUTH_INVALID
    if pf[2].type != JT_NONE and pf[2].type != JT_NULL:
        if pf[2].type != JT_NUM:
            return AUTH_INVALID
        jf_number(&pf[2], &t)
        if <double>g_wall >= t:
            return AUTH_INVALID
    if pf[3].type != JT_NONE and pf[3].type != JT_NULL:
        if pf[3].type != JT_NUM:
            return AUTH_INVALID
        jf_number(&pf[3], &t)
        if <double>g_wall < t:
            return AUTH_INVALID
    # sub: a positive integer as a string; username: any string
    if pf[0].type != JT_STR or pf[1].type != JT_STR:
        return AUTH_PAYLOAD
    m = jf_string(&pf[0], sub, 32)
    if m <= 0 or m > 19:
        return AUTH_PAYLOAD
    v = 0
    for i in range(m):
        if sub[i] < c'0' or sub[i] > c'9':
            return AUTH_PAYLOAD
        v = v * 10 + <uint64_t>(sub[i] - 48)
    if v == 0 or v > <uint64_t>0x7FFFFFFFFFFFFFFF:
        return AUTH_PAYLOAD
    m = jf_string(&pf[1], g_uname, USERNAME_MAX)
    if m < 0:
        return AUTH_PAYLOAD
    uid[0] = <int64_t>v
    ulen[0] = <size_t>m
    return AUTH_OK


# --- responses --------------------------------------------------------------------------------------

cdef inline void out_reserve(size_t need) noexcept nogil:
    global g_out, g_out_cap
    cdef size_t cap
    if g_out_len + need > g_out_cap:
        cap = g_out_cap * 2
        while cap < g_out_len + need:
            cap *= 2
        g_out = <char*>xrealloc(g_out, cap)
        g_out_cap = cap

cdef void seg_add(Conn* c, size_t start, size_t end, int nreq) noexcept nogil:
    global g_nseg, g_seg_cap, g_seg_conn, g_seg_start, g_seg_end, g_seg_nreq, g_closeq
    cdef int i = g_nseg - 1
    if i >= 0 and g_seg_conn[i] == c and g_seg_end[i] == start:
        g_seg_end[i] = end
        g_seg_nreq[i] += nreq
        return
    if g_nseg == g_seg_cap:
        g_seg_cap *= 2
        g_seg_conn = <Conn**>xrealloc(g_seg_conn, g_seg_cap * sizeof(Conn*))
        g_seg_start = <size_t*>xrealloc(g_seg_start, g_seg_cap * sizeof(size_t))
        g_seg_end = <size_t*>xrealloc(g_seg_end, g_seg_cap * sizeof(size_t))
        g_seg_nreq = <int*>xrealloc(g_seg_nreq, g_seg_cap * sizeof(int))
        g_closeq = <Conn**>xrealloc(g_closeq, g_seg_cap * sizeof(Conn*))
    g_seg_conn[g_nseg] = c
    g_seg_start[g_nseg] = start
    g_seg_end[g_nseg] = end
    g_seg_nreq[g_nseg] = nreq
    g_nseg += 1
    c.flags |= CF_IN_ROUND

cdef inline char* put_status(char* d, int status) noexcept nogil:
    if status == 200:
        return cps(d, b"HTTP/1.1 200 OK\r\n")
    if status == 201:
        return cps(d, b"HTTP/1.1 201 Created\r\n")
    if status == 400:
        return cps(d, b"HTTP/1.1 400 Bad Request\r\n")
    if status == 401:
        return cps(d, b"HTTP/1.1 401 Unauthorized\r\n")
    if status == 404:
        return cps(d, b"HTTP/1.1 404 Not Found\r\n")
    if status == 413:
        return cps(d, b"HTTP/1.1 413 Content Too Large\r\n")
    if status == 503:
        return cps(d, b"HTTP/1.1 503 Service Unavailable\r\n")
    return cps(d, b"HTTP/1.1 500 Internal Server Error\r\n")

cdef inline char* put_head(char* d, Conn* c, int status, size_t blen) noexcept nogil:
    d = put_status(d, status)
    d = cps(d, b"Content-Type: application/json\r\nContent-Length: ")
    d = put_u64(d, blen)
    d = cps(d, b"\r\n")
    d = cpy(d, g_date, g_date_len)
    if c.flags & CF_CLOSE:
        d = cps(d, b"Connection: close\r\n")
    elif c.flags & CF_KEEPALIVE10:
        d = cps(d, b"Connection: keep-alive\r\n")
    return cps(d, b"\r\n")

cdef void respond(Conn* c, int status, const char* body, size_t blen) noexcept nogil:
    global g_out_len
    cdef char* d
    cdef size_t start
    out_reserve(256 + blen)
    start = g_out_len
    d = put_head(g_out + start, c, status, blen)
    d = cpy(d, body, blen)
    g_out_len = d - g_out
    seg_add(c, start, g_out_len, 1)

cdef void respond_bb(Conn* c, int status, char* p) noexcept nogil:
    respond(c, status, g_bb, p - g_bb)

cdef void err(Conn* c, int status, const char* msg) noexcept nogil:
    cdef char buf[128]
    cdef char* d = cps(buf, b'{"error":"')
    d = cps(d, msg)
    d = cps(d, b'"}')
    respond(c, status, buf, d - buf)

cdef inline void err500(Conn* c) noexcept nogil:
    err(c, 500, b"internal server error")


# --- database ---------------------------------------------------------------------------------------

cdef bint ensure_txn() noexcept nogil:
    """Opens the round's transaction on first use."""
    global g_in_txn
    cdef int rc
    if g_in_txn:
        return True
    if g_round_failed:
        return False
    rc = sqlite3_step(st_begin)
    sqlite3_reset(st_begin)
    if rc != SQLITE_DONE:
        return False
    g_in_txn = True
    return True

cdef void db_failed() noexcept nogil:
    """After a failed statement: if SQLite rolled the transaction back, the round's writes are lost."""
    global g_in_txn, g_round_failed
    if g_in_txn and sqlite3_get_autocommit(g_db):
        g_in_txn = False
        g_round_failed = True

cdef void end_txn() noexcept nogil:
    global g_in_txn, g_round_failed
    cdef int rc
    if not g_in_txn:
        return
    g_in_txn = False
    rc = sqlite3_step(st_commit)
    sqlite3_reset(st_commit)
    if rc != SQLITE_DONE:
        if not sqlite3_get_autocommit(g_db):
            sqlite3_step(st_rollback)
            sqlite3_reset(st_rollback)
        g_round_failed = True

cdef int wal_hook_cb(void* arg, sqlite3* db, const char* name, int nframes) noexcept nogil:
    """Runs after each commit: starts a checkpoint once the WAL is large.

    Checkpoints run in another process. Bulk passes copy the WAL back into the database while writes
    go on. Once only a short tail is left, writes are held back for one final pass, after which the
    whole WAL has been copied and the next write transaction restarts it from the beginning (SQLite
    only rewinds a fully checkpointed WAL, so without that pause it would grow without bound)."""
    global g_wal_frames, g_ck_state, g_ck_passes
    g_wal_frames = nframes
    if nframes >= CKPT_FRAMES and g_ck_state == CK_IDLE and not g_ckpt_disabled:
        if write(g_ckreq_fd, b"p", 1) == 1:
            g_ck_state = CK_BULK
            g_ck_passes = 0
    return SQLITE_OK

cdef char* put_post(char* p, sqlite3_stmt* st) noexcept nogil:
    cdef int64_t pid = sqlite3_column_int64(st, 0)
    cdef const uint8_t* body = sqlite3_column_text(st, 1)
    cdef size_t blen = sqlite3_column_bytes(st, 1)
    cdef const uint8_t* ca = sqlite3_column_text(st, 2)
    cdef size_t calen = sqlite3_column_bytes(st, 2)
    cdef const uint8_t* au = sqlite3_column_text(st, 3)
    cdef size_t aulen = sqlite3_column_bytes(st, 3)
    cdef int64_t likes = sqlite3_column_int64(st, 4)
    p = bb_ensure(p, 128 + 6 * (blen + calen + aulen))
    p = cps(p, b'{"id":')
    p = put_i64(p, pid)
    p = cps(p, b',"body":')
    p = put_json_str(p, body, blen)
    p = cps(p, b',"created_at":')
    p = put_json_str(p, ca, calen)
    p = cps(p, b',"author":')
    p = put_json_str(p, au, aulen)
    p = cps(p, b',"like_count":')
    p = put_i64(p, likes)
    p[0] = c'}'
    return p + 1


# --- handlers ---------------------------------------------------------------------------------------

cdef void h_health(Conn* c) noexcept nogil:
    cdef int rc = -1
    cdef char* p
    cdef const char* msg
    if ensure_txn():
        rc = sqlite3_step(st_one)
        sqlite3_reset(st_one)
    if rc == SQLITE_ROW:
        p = cps(g_bb, b'{"status":"ok","db":"ok","uptime_s":')
        p = put_i64(p, g_now - g_start)
        p = cps(p, b'}')
        respond_bb(c, 200, p)
        return
    db_failed()
    msg = sqlite3_errmsg(g_db)
    p = cps(g_bb, b'{"status":"degraded","db":"unreachable","error":')
    p = put_json_str(p, <const uint8_t*>msg, strlen(msg))
    p = cps(p, b'}')
    respond_bb(c, 503, p)

cdef void h_feed(Conn* c) noexcept nogil:
    cdef int rc
    cdef char* p
    cdef bint first = True
    if not ensure_txn():
        err500(c)
        return
    p = cps(g_bb, b'{"posts":[')
    while True:
        rc = sqlite3_step(st_feed)
        if rc != SQLITE_ROW:
            break
        if not first:
            p[0] = c','
            p += 1
        first = False
        p = put_post(p, st_feed)
    sqlite3_reset(st_feed)
    if rc != SQLITE_DONE:
        db_failed()
        err500(c)
        return
    p = cps(p, b']}')
    respond_bb(c, 200, p)

cdef void h_post(Conn* c, int64_t pid) noexcept nogil:
    cdef int rc
    cdef char* p
    if not ensure_txn():
        err500(c)
        return
    sqlite3_bind_int64(st_post, 1, pid)
    rc = sqlite3_step(st_post)
    if rc == SQLITE_ROW:
        p = cps(g_bb, b'{"post":')
        p = put_post(p, st_post)
        p = cps(p, b'}')
        sqlite3_reset(st_post)
        respond_bb(c, 200, p)
        return
    sqlite3_reset(st_post)
    if rc == SQLITE_DONE:
        err(c, 404, b"post not found")
    else:
        db_failed()
        err500(c)

cdef inline size_t ws_lead(const uint8_t* p, const uint8_t* e) noexcept nogil:
    """Length of the JavaScript String.prototype.trim() whitespace character at p, else 0."""
    cdef uint8_t ch = p[0]
    if ch == 32 or (ch >= 9 and ch <= 13):
        return 1
    if ch == 0xC2:
        return 2 if p + 1 < e and p[1] == 0xA0 else 0
    if p + 2 >= e:
        return 0
    return ws3(p)

cdef inline size_t ws3(const uint8_t* p) noexcept nogil:
    cdef uint8_t a = p[0], b = p[1], c = p[2]
    if a == 0xE2:
        if b == 0x80 and ((c >= 0x80 and c <= 0x8A) or c == 0xA8 or c == 0xA9 or c == 0xAF):
            return 3        # U+2000..U+200A, U+2028, U+2029, U+202F
        if b == 0x81 and c == 0x9F:
            return 3        # U+205F
        return 0
    if (a == 0xE1 and b == 0x9A and c == 0x80) or (a == 0xE3 and b == 0x80 and c == 0x80) \
            or (a == 0xEF and b == 0xBB and c == 0xBF):
        return 3            # U+1680, U+3000, U+FEFF
    return 0

cdef inline size_t ws_trail(const uint8_t* s, const uint8_t* e) noexcept nogil:
    cdef uint8_t ch = (e - 1)[0]
    if ch == 32 or (ch >= 9 and ch <= 13):
        return 1
    if e - s >= 2 and (e - 2)[0] == 0xC2 and ch == 0xA0:
        return 2
    if e - s >= 3:
        return ws3(e - 3)
    return 0

cdef void h_create(Conn* c, int64_t uid, size_t ulen, const char* body, size_t blen) noexcept nogil:
    cdef JField f[1]
    cdef int r, rc
    cdef ssize_t m
    cdef const uint8_t* s
    cdef const uint8_t* e
    cdef size_t k, n, chars
    cdef char* p
    jf_init(&f[0], b"body")
    r = j_fields(body, body + blen, f, 1)
    if r < 0:
        err(c, 400, b"malformed JSON body")
        return
    if r == 0 or f[0].type != JT_STR:
        err(c, 400, b"body is required")
        return
    m = jf_string(&f[0], g_tmp, MAX_BODY)
    if m < 0:
        err(c, 400, b"malformed JSON body")
        return
    s = <const uint8_t*>g_tmp
    e = s + m
    while s < e:
        k = ws_lead(s, e)
        if k == 0:
            break
        s += k
    while e > s:
        k = ws_trail(s, e)
        if k == 0:
            break
        e -= k
    n = e - s
    if n == 0:
        err(c, 400, b"body is required")
        return
    if n > 500:
        chars = 0
        for k in range(n):
            if (s[k] & 0xC0) != 0x80:
                chars += 1
        if chars > 500:
            err(c, 400, b"body must be at most 500 characters")
            return
    if not ensure_txn():
        err500(c)
        return
    sqlite3_bind_int64(st_ins_post, 1, uid)
    sqlite3_bind_text(st_ins_post, 2, <const char*>s, <int>n, SQLITE_STATIC)
    rc = sqlite3_step(st_ins_post)
    if rc != SQLITE_ROW:
        sqlite3_reset(st_ins_post)
        db_failed()
        err500(c)
        return
    p = bb_ensure(g_bb, 256 + 6 * (n + ulen))
    p = cps(p, b'{"post":{"id":')
    p = put_i64(p, sqlite3_column_int64(st_ins_post, 0))
    p = cps(p, b',"body":')
    p = put_json_str(p, s, n)
    p = cps(p, b',"created_at":')
    p = put_json_str(p, sqlite3_column_text(st_ins_post, 1), sqlite3_column_bytes(st_ins_post, 1))
    p = cps(p, b',"author":')
    p = put_json_str(p, <const uint8_t*>g_uname, ulen)
    p = cps(p, b',"like_count":0}}')
    rc = sqlite3_step(st_ins_post)
    sqlite3_reset(st_ins_post)
    if rc != SQLITE_DONE:
        db_failed()
        err500(c)
        return
    respond_bb(c, 201, p)

cdef void h_like(Conn* c, int64_t uid, int64_t pid) noexcept nogil:
    cdef int rc
    cdef char* p
    if not ensure_txn():
        err500(c)
        return
    sqlite3_bind_int64(st_ins_like, 1, uid)
    sqlite3_bind_int64(st_ins_like, 2, pid)
    rc = sqlite3_step(st_ins_like)
    sqlite3_reset(st_ins_like)
    if rc != SQLITE_DONE:
        db_failed()
        err500(c)
        return
    if sqlite3_changes(g_db) == 1:
        p = cps(g_bb, b'{"liked":true,"already_liked":false,"post_id":')
        p = put_i64(p, pid)
        p = cps(p, b'}')
        respond_bb(c, 201, p)
        return
    sqlite3_bind_int64(st_exists, 1, pid)
    rc = sqlite3_step(st_exists)
    sqlite3_reset(st_exists)
    if rc == SQLITE_ROW:
        p = cps(g_bb, b'{"liked":true,"already_liked":true,"post_id":')
        p = put_i64(p, pid)
        p = cps(p, b'}')
        respond_bb(c, 200, p)
    elif rc == SQLITE_DONE:
        err(c, 404, b"post not found")
    else:
        db_failed()
        err500(c)

cdef inline int64_t parse_id(const char* s, size_t n) noexcept nogil:
    """A positive decimal integer; -1 if invalid, -2 if too large to exist."""
    cdef uint64_t v = 0
    cdef size_t i
    cdef bint big = False
    if n == 0:
        return -1
    for i in range(n):
        if s[i] < c'0' or s[i] > c'9':
            return -1
        if v > <uint64_t>100000000000000000:
            big = True
        else:
            v = v * 10 + <uint64_t>(s[i] - 48)
    if big or v > <uint64_t>0x7FFFFFFFFFFFFFFF:
        return -2
    if v == 0:
        return -1
    return <int64_t>v

cdef bint authenticate(Conn* c, const char* auth, size_t alen, int64_t* uid, size_t* ulen) noexcept nogil:
    cdef int r
    if alen < 7 or memcmp(auth, b"Bearer ", 7) != 0:
        err(c, 401, b"missing bearer token")
        return False
    r = verify_jwt(auth + 7, alen - 7, uid, ulen)
    if r == AUTH_OK:
        return True
    if r == AUTH_PAYLOAD:
        err(c, 401, b"invalid token payload")
    else:
        err(c, 401, b"invalid or expired token")
    return False

cdef void route(Conn* c, bint is_get, bint is_post, const char* path, size_t plen,
                const char* auth, size_t alen, const char* body, size_t blen) noexcept nogil:
    cdef const char* q = <const char*>memchr(path, c'?', plen)
    cdef int64_t pid, uid = 0
    cdef size_t ulen = 0
    if q != NULL:
        plen = q - path
    if is_get:
        if plen == 5 and memcmp(path, b"/feed", 5) == 0:
            h_feed(c)
            return
        if plen > 7 and memcmp(path, b"/posts/", 7) == 0:
            if memchr(path + 7, c'/', plen - 7) != NULL:
                err(c, 404, b"not found")
                return
            pid = parse_id(path + 7, plen - 7)
            if pid == -1:
                err(c, 400, b"invalid post id")
            elif pid == -2:
                err(c, 404, b"post not found")
            else:
                h_post(c, pid)
            return
        if plen == 7 and memcmp(path, b"/health", 7) == 0:
            h_health(c)
            return
    elif is_post:
        if plen == 6 and memcmp(path, b"/posts", 6) == 0:
            if authenticate(c, auth, alen, &uid, &ulen):
                h_create(c, uid, ulen, body, blen)
            return
        if (plen > 12 and memcmp(path, b"/posts/", 7) == 0 and memcmp(path + plen - 5, b"/like", 5) == 0
                and memchr(path + 7, c'/', plen - 12) == NULL):
            if authenticate(c, auth, alen, &uid, &ulen):
                pid = parse_id(path + 7, plen - 12)
                if pid == -1:
                    err(c, 400, b"invalid post id")
                elif pid == -2:
                    err(c, 404, b"post not found")
                else:
                    h_like(c, uid, pid)
            return
    err(c, 404, b"not found")


# --- HTTP parsing -----------------------------------------------------------------------------------

cdef inline bint ci_eq(const char* a, const char* lower, size_t n) noexcept nogil:
    cdef size_t i
    for i in range(n):
        if (a[i] | 0x20) != lower[i]:
            return False
    return True

cdef bint ci_contains(const char* s, size_t n, const char* lower, size_t k) noexcept nogil:
    cdef size_t i
    if n < k:
        return False
    for i in range(n - k + 1):
        if ci_eq(s + i, lower, k):
            return True
    return False

cdef ssize_t dechunk(const char* p, const char* e, size_t* outlen) noexcept nogil:
    """Decodes a chunked body into g_chunk. Returns bytes consumed, 0 if incomplete, -1 if invalid,
    -2 if too large."""
    global g_chunk
    cdef const char* start = p
    cdef const char* nl
    cdef size_t size, olen = 0
    cdef int h
    if g_chunk == NULL:
        g_chunk = <char*>xrealloc(NULL, MAX_BODY)
    while True:
        nl = <const char*>memchr(p, c'\n', e - p)
        if nl == NULL:
            return 0 if e - p < 64 else -1
        size = 0
        h = hexv(p[0])
        if h < 0:
            return -1
        while p < nl:
            h = hexv(p[0])
            if h < 0:
                break
            size = size * 16 + h
            if size > MAX_BODY:
                return -2
            p += 1
        p = nl + 1
        if size == 0:
            while True:     # optional trailer fields, then an empty line
                nl = <const char*>memchr(p, c'\n', e - p)
                if nl == NULL:
                    return 0 if e - p < MAX_HEADER else -1
                if nl == p or (nl == p + 1 and p[0] == c'\r'):
                    outlen[0] = olen
                    return nl + 1 - start
                p = nl + 1
        if olen + size > MAX_BODY:
            return -2
        if <size_t>(e - p) < size + 2:
            return 0
        memcpy(g_chunk + olen, p, size)
        olen += size
        p += size
        if p[0] == c'\r' and p[1] == c'\n':
            p += 2
        elif p[0] == c'\n':
            p += 1
        else:
            return -1

cdef void defer_conn(Conn* c) noexcept nogil:
    global g_deferred, g_deferred_cap, g_ndeferred
    if g_ndeferred == g_deferred_cap:
        g_deferred_cap = g_deferred_cap * 2 if g_deferred_cap else 256
        g_deferred = <Conn**>xrealloc(g_deferred, g_deferred_cap * sizeof(Conn*))
    g_deferred[g_ndeferred] = c
    g_ndeferred += 1
    c.flags |= CF_DEFERRED

cdef void send_interim(Conn* c, const char* s) noexcept nogil:
    global g_out_len
    cdef size_t n = strlen(s), start
    out_reserve(n)
    start = g_out_len
    memcpy(g_out + start, s, n)
    g_out_len += n
    seg_add(c, start, g_out_len, 0)

cdef inline void bad_request(Conn* c, int status, const char* msg) noexcept nogil:
    c.flags |= CF_CLOSE
    err(c, status, msg)

cdef ssize_t handle_request(Conn* c, char* data, size_t len) noexcept nogil:
    """Parses and serves one request at data. Returns the bytes consumed, 0 if the request is incomplete
    (or deferred), -1 if the connection must stop reading."""
    cdef const char* end = data + len
    cdef const char* p = data
    cdef const char* nl
    cdef const char* le
    cdef const char* sp1
    cdef const char* sp2
    cdef const char* colon
    cdef const char* v
    cdef const char* ve
    cdef const char* path
    cdef const char* auth = NULL
    cdef const char* body = NULL
    cdef const char* hend
    cdef size_t plen, alen = 0, blen = 0, nlen, cl = 0, total
    cdef bint is_get, is_post, http10, has_cl = False, chunked = False, expect = False
    cdef bint conn_close = False, conn_ka = False
    cdef ssize_t r
    # tolerate empty lines before the request line (RFC 9112 2.2)
    while p < end and (p[0] == c'\r' or p[0] == c'\n'):
        p += 1
    if p == end:
        return len
    nl = <const char*>memchr(p, c'\n', end - p)
    if nl == NULL:
        if len > MAX_HEADER:
            bad_request(c, 400, b"bad request")
            return -1
        return 0
    le = nl - 1 if nl > p and (nl - 1)[0] == c'\r' else nl
    sp1 = <const char*>memchr(p, c' ', le - p)
    if sp1 == NULL:
        bad_request(c, 400, b"bad request")
        return -1
    sp2 = <const char*>memchr(sp1 + 1, c' ', le - sp1 - 1)
    if sp2 == NULL or le - sp2 - 1 != 8 or memcmp(sp2 + 1, b"HTTP/1.", 7) != 0:
        bad_request(c, 400, b"bad request")
        return -1
    http10 = sp2[8] == c'0'
    is_get = sp1 - p == 3 and memcmp(p, b"GET", 3) == 0
    is_post = sp1 - p == 4 and memcmp(p, b"POST", 4) == 0
    path = sp1 + 1
    plen = sp2 - path
    # header fields
    p = nl + 1
    while True:
        if p >= end:
            if len > MAX_HEADER:
                bad_request(c, 400, b"bad request")
                return -1
            return 0
        nl = <const char*>memchr(p, c'\n', end - p)
        if nl == NULL:
            if len > MAX_HEADER:
                bad_request(c, 400, b"bad request")
                return -1
            return 0
        le = nl - 1 if nl > p and (nl - 1)[0] == c'\r' else nl
        if le == p:
            break
        colon = <const char*>memchr(p, c':', le - p)
        if colon == NULL:
            bad_request(c, 400, b"bad request")
            return -1
        nlen = colon - p
        v = colon + 1
        while v < le and (v[0] == c' ' or v[0] == c'\t'):
            v += 1
        ve = le
        while ve > v and ((ve - 1)[0] == c' ' or (ve - 1)[0] == c'\t'):
            ve -= 1
        if nlen == 13 and ci_eq(p, b"authorization", 13):
            auth = v
            alen = ve - v
        elif nlen == 14 and ci_eq(p, b"content-length", 14):
            if ve == v:
                bad_request(c, 400, b"bad request")
                return -1
            cl = 0
            while v < ve:
                if v[0] < c'0' or v[0] > c'9' or cl > MAX_BODY:
                    bad_request(c, 400 if v[0] < c'0' or v[0] > c'9' else 413,
                                b"bad request" if v[0] < c'0' or v[0] > c'9' else b"request body too large")
                    return -1
                cl = cl * 10 + <size_t>(v[0] - 48)
                v += 1
            has_cl = True
        elif nlen == 10 and ci_eq(p, b"connection", 10):
            if ci_contains(v, ve - v, b"close", 5):
                conn_close = True
            if ci_contains(v, ve - v, b"keep-alive", 10):
                conn_ka = True
        elif nlen == 17 and ci_eq(p, b"transfer-encoding", 17):
            if ci_contains(v, ve - v, b"chunked", 7):
                chunked = True
        elif nlen == 6 and ci_eq(p, b"expect", 6):
            if ci_contains(v, ve - v, b"100-continue", 12):
                expect = True
        p = nl + 1
    hend = nl + 1
    if hend - data > MAX_HEADER:
        bad_request(c, 400, b"bad request")
        return -1
    # body
    if chunked:
        r = dechunk(hend, end, &blen)
        if r == 0:
            if expect and not (c.flags & CF_CONTINUE_SENT) and hend == end:
                send_interim(c, b"HTTP/1.1 100 Continue\r\n\r\n")
                c.flags |= CF_CONTINUE_SENT
            return 0
        if r < 0:
            bad_request(c, 413 if r == -2 else 400, b"request body too large" if r == -2 else b"bad request")
            return -1
        body = g_chunk
        total = (hend - data) + r
    elif has_cl:
        if cl > MAX_BODY:
            bad_request(c, 413, b"request body too large")
            return -1
        if <size_t>(end - hend) < cl:
            if expect and not (c.flags & CF_CONTINUE_SENT):
                send_interim(c, b"HTTP/1.1 100 Continue\r\n\r\n")
                c.flags |= CF_CONTINUE_SENT
            return 0
        body = hend
        blen = cl
        total = (hend - data) + cl
    else:
        total = hend - data
    # Writes wait during the final checkpoint pass, so that the WAL can be reset afterwards.
    if is_post and g_defer_writes:
        defer_conn(c)
        return 0
    c.flags &= ~(CF_CONTINUE_SENT | CF_KEEPALIVE10)
    if http10:
        if conn_ka and not conn_close:
            c.flags |= CF_KEEPALIVE10
        else:
            c.flags |= CF_CLOSE
    elif conn_close:
        c.flags |= CF_CLOSE
    route(c, is_get, is_post, path, plen, auth, alen, body, blen)
    if c.flags & CF_CLOSE:
        return -1
    return <ssize_t>total

cdef size_t process(Conn* c, char* data, size_t len, bint* more) noexcept nogil:
    """Serves up to MAX_PIPELINE complete requests in data; returns how many bytes were consumed.
    *more is set when it stopped at the cap with input left."""
    cdef size_t pos = 0
    cdef ssize_t r
    cdef int n = 0
    more[0] = False
    while pos < len:
        if n == MAX_PIPELINE:
            more[0] = True
            break
        r = handle_request(c, data + pos, len - pos)
        if r <= 0:
            if r < 0:
                return len      # closing: drop the rest
            break
        pos += r
        n += 1
    return pos


# --- connections ------------------------------------------------------------------------------------

cdef void set_events(Conn* c, uint32_t ev) noexcept nogil:
    cdef epoll_event e
    if c.ev == ev:
        return
    e.events = ev
    e.data.u64 = 0
    e.data.fd = c.fd
    if c.ev == 0:
        epoll_ctl(g_ep, EPOLL_CTL_ADD, c.fd, &e)
    elif ev == 0:
        epoll_ctl(g_ep, EPOLL_CTL_DEL, c.fd, &e)
    else:
        epoll_ctl(g_ep, EPOLL_CTL_MOD, c.fd, &e)
    c.ev = ev

cdef inline void lru_unlink(Conn* c) noexcept nogil:
    global g_lru_head, g_lru_tail
    if c.prev != NULL:
        c.prev.next = c.next
    else:
        g_lru_head = c.next
    if c.next != NULL:
        c.next.prev = c.prev
    else:
        g_lru_tail = c.prev
    c.prev = NULL
    c.next = NULL

cdef inline void lru_append(Conn* c) noexcept nogil:
    global g_lru_head, g_lru_tail
    c.prev = g_lru_tail
    c.next = NULL
    if g_lru_tail != NULL:
        g_lru_tail.next = c
    else:
        g_lru_head = c
    g_lru_tail = c

cdef inline void touch(Conn* c) noexcept nogil:
    c.last_active = g_now
    if c != g_lru_tail:
        lru_unlink(c)
        lru_append(c)

cdef void conn_free(Conn* c) noexcept nogil:
    global g_nconns
    lru_unlink(c)
    close(c.fd)
    g_conns[c.fd] = NULL
    free(c.rbuf)
    free(c.wbuf)
    free(c)
    g_nconns -= 1

cdef inline bint backlogged(Conn* c) noexcept nogil:
    return c.wlen - c.woff > BACKPRESSURE

cdef inline bint held(Conn* c) noexcept nogil:
    """Input is buffered but not served now: waiting for a checkpoint, for the next round, or for
    the client to read its responses."""
    return (c.flags & (CF_DEFERRED | CF_PENDING)) != 0 or backlogged(c)

cdef void update_events(Conn* c) noexcept nogil:
    """Derives the epoll interest from the connection's state."""
    cdef uint32_t ev = 0
    if c.wlen > c.woff and not (c.flags & CF_DEAD):
        ev = EPOLLOUT
    if not (c.flags & CF_CLOSE) and not backlogged(c) and c.rlen < RBUF_LIMIT:
        ev |= EPOLLIN
    set_events(c, ev)

cdef void add_pending(Conn* c) noexcept nogil:
    global g_pending, g_pending_spare, g_pending_cap, g_npending
    if c.flags & CF_PENDING:
        return
    if g_npending == g_pending_cap:
        g_pending_cap = g_pending_cap * 2 if g_pending_cap else 256
        g_pending = <Conn**>xrealloc(g_pending, g_pending_cap * sizeof(Conn*))
        g_pending_spare = <Conn**>xrealloc(g_pending_spare, g_pending_cap * sizeof(Conn*))
    g_pending[g_npending] = c
    g_npending += 1
    c.flags |= CF_PENDING

cdef void maybe_free(Conn* c) noexcept nogil:
    """Closes a connection marked CF_CLOSE once nothing refers to it any more."""
    if not (c.flags & CF_CLOSE) or (c.flags & (CF_IN_ROUND | CF_DEFERRED | CF_PENDING)):
        return
    if c.wlen > c.woff and not (c.flags & CF_DEAD):
        set_events(c, EPOLLOUT)
        return
    conn_free(c)

cdef void wbuf_append(Conn* c, const char* s, size_t n) noexcept nogil:
    cdef size_t live = c.wlen - c.woff, cap
    if c.woff > 0:
        memmove(c.wbuf, c.wbuf + c.woff, live)
        c.wlen = live
        c.woff = 0
    if c.wlen + n > c.wcap:
        cap = c.wcap * 2 if c.wcap else 4096
        while cap < c.wlen + n:
            cap *= 2
        c.wbuf = <char*>xrealloc(c.wbuf, cap)
        c.wcap = cap
    memcpy(c.wbuf + c.wlen, s, n)
    c.wlen += n

cdef void conn_send(Conn* c, const char* s, size_t n) noexcept nogil:
    cdef ssize_t r
    if c.flags & CF_DEAD:
        return
    if c.wlen > c.woff:
        wbuf_append(c, s, n)
    else:
        r = send(c.fd, s, n, MSG_NOSIGNAL)
        if r == <ssize_t>n:
            return
        if r < 0:
            if errno != EAGAIN and errno != EINTR:
                c.flags |= CF_DEAD | CF_CLOSE
                return
            r = 0
        wbuf_append(c, s + r, n - r)
    update_events(c)

cdef void on_write(Conn* c) noexcept nogil:
    cdef ssize_t r
    if c.wlen > c.woff:
        r = send(c.fd, c.wbuf + c.woff, c.wlen - c.woff, MSG_NOSIGNAL)
        if r < 0:
            if errno != EAGAIN and errno != EINTR:
                c.flags |= CF_DEAD | CF_CLOSE
                maybe_free(c)
            return
        c.woff += r
        touch(c)
    if c.woff == c.wlen:
        free(c.wbuf)
        c.wbuf = NULL
        c.wlen = 0
        c.woff = 0
        c.wcap = 0
        if c.flags & CF_CLOSE:
            set_events(c, 0)
            maybe_free(c)
            return
    if c.rlen > 0 and not held(c):
        add_pending(c)          # requests buffered while the client was not reading
    update_events(c)

cdef void keep_input(Conn* c, const char* data, size_t n) noexcept nogil:
    """Stores n bytes as the connection's unprocessed input (data may already be c.rbuf)."""
    if data == c.rbuf:
        return
    if c.rcap < n:
        c.rcap = n if n > 4096 else 4096
        free(c.rbuf)
        c.rbuf = <char*>xrealloc(NULL, c.rcap)
    memcpy(c.rbuf, data, n)
    c.rlen = n

cdef void consume(Conn* c, char* data, size_t len) noexcept nogil:
    """Processes data (the shared buffer or c.rbuf) and keeps whatever is left unprocessed."""
    cdef bint more
    cdef size_t used = process(c, data, len, &more)
    cdef size_t left = len - used
    if left == 0 or (c.flags & CF_CLOSE):
        free(c.rbuf)
        c.rbuf = NULL
        c.rlen = 0
        c.rcap = 0
    elif data == c.rbuf:
        if used:
            memmove(c.rbuf, c.rbuf + used, left)
        c.rlen = left
    else:
        keep_input(c, data + used, left)
    if more and not (c.flags & (CF_CLOSE | CF_DEFERRED)):
        add_pending(c)

cdef void on_eof(Conn* c) noexcept nogil:
    c.flags |= CF_CLOSE
    if c.flags & (CF_IN_ROUND | CF_DEFERRED | CF_PENDING) or c.wlen > c.woff:
        update_events(c)
        return
    conn_free(c)

cdef void on_read(Conn* c) noexcept nogil:
    cdef ssize_t n
    cdef size_t cap
    if c.flags & CF_CLOSE:
        update_events(c)
        return
    touch(c)
    if c.rlen == 0:
        n = recv(c.fd, g_rdbuf, RDBUF_SIZE, 0)
        if n > 0:
            if held(c):
                keep_input(c, g_rdbuf, n)
                update_events(c)
            else:
                consume(c, g_rdbuf, n)
            return
    else:
        if c.rcap - c.rlen < 16384 and c.rcap < RBUF_LIMIT:
            cap = c.rcap * 2
            if cap > RBUF_LIMIT:
                cap = RBUF_LIMIT
            c.rbuf = <char*>xrealloc(c.rbuf, cap)
            c.rcap = cap
        if c.rlen >= c.rcap:
            if held(c):
                update_events(c)    # full: stop reading until the buffered requests are served
            else:
                bad_request(c, 413, b"request too large")
                on_eof(c)
            return
        n = recv(c.fd, c.rbuf + c.rlen, c.rcap - c.rlen, 0)
        if n > 0:
            c.rlen += n
            if held(c):
                update_events(c)
            else:
                consume(c, c.rbuf, c.rlen)
            return
    if n == 0:
        on_eof(c)
    elif errno != EAGAIN and errno != EINTR:
        c.flags |= CF_DEAD
        on_eof(c)

cdef void run_deferred() noexcept nogil:
    """The checkpoint finished: serve the write requests that waited for it."""
    global g_ndeferred, g_deferred, g_deferred_cap
    cdef int i, n = g_ndeferred
    cdef Conn* c
    cdef Conn** lst = g_deferred
    g_deferred = NULL
    g_ndeferred = 0
    g_deferred_cap = 0
    for i in range(n):
        c = lst[i]
        c.flags &= ~CF_DEFERRED
        if not held(c) and not (c.flags & CF_DEAD):
            consume(c, c.rbuf, c.rlen)
        update_events(c)
        if c.flags & CF_CLOSE:
            maybe_free(c)
    free(lst)

cdef void run_pending() noexcept nogil:
    """Serves the next MAX_PIPELINE requests of connections that had more buffered."""
    global g_pending, g_pending_spare, g_npending
    cdef int i, n = g_npending
    cdef Conn* c
    cdef Conn** lst = g_pending
    g_pending = g_pending_spare     # connections that still have more go to the other list
    g_pending_spare = lst
    g_npending = 0
    for i in range(n):
        c = lst[i]
        c.flags &= ~CF_PENDING
        if not held(c) and not (c.flags & CF_DEAD):
            consume(c, c.rbuf, c.rlen)
        update_events(c)
        if c.flags & CF_CLOSE:
            maybe_free(c)

cdef bint evict_one() noexcept nogil:
    """Makes room for a new connection by closing the least recently active idle one."""
    cdef Conn* c = g_lru_head
    cdef int scanned = 0
    while c != NULL and scanned < 64:
        if (c.last_active < g_now and c.rlen == 0 and c.wlen == 0
                and not (c.flags & (CF_IN_ROUND | CF_DEFERRED | CF_PENDING))):
            conn_free(c)
            return True
        c = c.next
        scanned += 1
    return False

cdef void pause_listen() noexcept nogil:
    global g_listen_paused, g_listen_resume_ms
    cdef epoll_event e
    if not g_listen_paused:
        epoll_ctl(g_ep, EPOLL_CTL_DEL, g_lfd, &e)
        g_listen_paused = True
    g_listen_resume_ms = g_now_ms + 100

cdef void on_accept() noexcept nogil:
    global g_nconns
    cdef int fd, i, one = 1
    cdef Conn* c
    for i in range(512):
        if g_nconns >= g_max_conns and not evict_one():
            pause_listen()
            return
        fd = accept4(g_lfd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC)
        if fd < 0:
            if errno == EINTR or errno == ECONNABORTED:
                continue
            if errno == EMFILE or errno == ENFILE or errno == ENOBUFS or errno == ENOMEM:
                if not evict_one():
                    pause_listen()
                    return
                continue
            return
        if fd >= g_max_fds:
            close(fd)
            continue
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(int))
        c = <Conn*>calloc(1, sizeof(Conn))
        if c == NULL:
            close(fd)
            return
        c.fd = fd
        c.last_active = g_now
        g_conns[fd] = c
        g_nconns += 1
        lru_append(c)
        set_events(c, EPOLLIN)

cdef void on_ckpt_done() noexcept nogil:
    global g_ckpt_disabled, g_ck_state, g_ck_passes, g_defer_writes
    cdef int copied = 0
    cdef ssize_t n = read(g_ckdone_fd, &copied, sizeof(int))
    cdef epoll_event e
    if n < 0 and (errno == EAGAIN or errno == EINTR):
        return
    if n != sizeof(int):
        # The checkpointer is gone: let SQLite checkpoint on commit instead.
        epoll_ctl(g_ep, EPOLL_CTL_DEL, g_ckdone_fd, &e)
        g_ckpt_disabled = True
        sqlite3_wal_autocheckpoint(g_db, 1000)
        g_ck_state = CK_IDLE
        g_defer_writes = False
        run_deferred()
        return
    if g_ck_state == CK_BULK:
        g_ck_passes += 1
        if g_wal_frames - copied > CKPT_TAIL and g_ck_passes < CKPT_PASSES:
            if write(g_ckreq_fd, b"p", 1) != 1:
                g_ck_state = CK_IDLE
        else:
            g_ck_state = CK_FINAL_NEXT
            g_defer_writes = True
    elif g_ck_state == CK_FINAL:
        g_ck_state = CK_IDLE
        g_defer_writes = False
        run_deferred()

cdef void flush_round() noexcept nogil:
    """Ends the round: COMMIT, then write every response of the round."""
    global g_nseg, g_out_len, g_nclose, g_round_failed, g_ck_state, g_defer_writes, g_out, g_out_cap
    cdef int i, k
    cdef Conn* c
    cdef char tmp[512]
    cdef char* d
    end_txn()
    for i in range(g_nseg):
        c = g_seg_conn[i]
        c.flags &= ~CF_IN_ROUND
        if g_round_failed:
            # The commit failed: none of this round's responses may claim success.
            for k in range(g_seg_nreq[i]):
                d = put_head(tmp, c, 500, 33)
                d = cps(d, b'{"error":"internal server error"}')
                conn_send(c, tmp, d - tmp)
        else:
            conn_send(c, g_out + g_seg_start[i], g_seg_end[i] - g_seg_start[i])
        if (c.flags & CF_CLOSE) and not (c.flags & CF_CLOSE_QUEUED):
            c.flags |= CF_CLOSE_QUEUED
            g_closeq[g_nclose] = c
            g_nclose += 1
    g_round_failed = False
    for i in range(g_nclose):
        c = g_closeq[i]
        c.flags &= ~CF_CLOSE_QUEUED
        maybe_free(c)
    g_nseg = 0
    g_nclose = 0
    g_out_len = 0
    if g_out_cap > (8 << 20):      # give back the memory of an unusually large round
        g_out = <char*>xrealloc(g_out, 1 << 20)
        g_out_cap = 1 << 20
    if g_ck_state == CK_FINAL_NEXT:
        # This round's writes are committed and no more are taken: run the final pass.
        if write(g_ckreq_fd, b"f", 1) == 1:
            g_ck_state = CK_FINAL
        else:
            g_ck_state = CK_IDLE
            g_defer_writes = False
            run_deferred()

cdef void sweep_idle() noexcept nogil:
    cdef Conn* c = g_lru_head
    cdef Conn* nxt
    while c != NULL and c.last_active + IDLE_TIMEOUT <= g_now:
        nxt = c.next
        if not (c.flags & (CF_IN_ROUND | CF_DEFERRED | CF_PENDING)):
            conn_free(c)
        c = nxt

cdef void update_clock() noexcept nogil:
    global g_now, g_now_ms, g_wall, g_date_len
    cdef timespec ts
    cdef tm t
    clock_gettime(CLOCK_MONOTONIC_COARSE, &ts)
    g_now = ts.tv_sec
    g_now_ms = <int64_t>ts.tv_sec * 1000 + ts.tv_nsec // 1000000
    clock_gettime(CLOCK_REALTIME_COARSE, &ts)
    if ts.tv_sec != g_wall:
        g_wall = ts.tv_sec
        gmtime_r(&ts.tv_sec, &t)
        g_date_len = strftime(g_date, 64, b"Date: %a, %d %b %Y %H:%M:%S GMT\r\n", &t)

cdef int event_loop() noexcept nogil:
    global g_last_sweep, g_listen_paused
    cdef epoll_event events[MAX_EVENTS]
    cdef epoll_event e
    cdef int n, i, fd
    cdef uint32_t ev
    cdef Conn* c
    while True:
        if g_stop:
            return 0
        n = epoll_wait(g_ep, events, MAX_EVENTS, 0 if g_npending else (50 if g_listen_paused else 1000))
        if n < 0:
            if errno == EINTR:
                continue
            return -1
        update_clock()
        if g_npending:
            run_pending()
        for i in range(n):
            fd = events[i].data.fd
            ev = events[i].events
            if fd == g_lfd:
                on_accept()
                continue
            if fd == g_ckdone_fd:
                on_ckpt_done()
                continue
            c = g_conns[fd]
            if c == NULL:
                continue
            if ev & EPOLLOUT:
                on_write(c)
                c = g_conns[fd]
                if c == NULL:
                    continue
            if (ev & EPOLLIN) and (c.ev & EPOLLIN):
                on_read(c)
            elif (ev & (EPOLLERR | EPOLLHUP)) and not (ev & EPOLLOUT):
                c.flags |= CF_DEAD
                on_eof(c)
        flush_round()
        if g_listen_paused and g_now_ms >= g_listen_resume_ms:
            e.events = EPOLLIN
            e.data.u64 = 0
            e.data.fd = g_lfd
            epoll_ctl(g_ep, EPOLL_CTL_ADD, g_lfd, &e)
            g_listen_paused = False
        if g_now != g_last_sweep:
            g_last_sweep = g_now
            sweep_idle()


# --- setup (Python level) ---------------------------------------------------------------------------

cdef void init_tables():
    cdef int i
    cdef uint32_t k[64]
    k[:] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]
    for i in range(64):
        K256[i] = k[i]
    for i in range(256):
        B64D[i] = 255
        ESC[i] = 0
    for i in range(64):
        B64D[<uint8_t>B64E[i]] = i
    for i in range(32):
        ESC[i] = c'u'
    ESC[8] = c'b'
    ESC[9] = c't'
    ESC[10] = c'n'
    ESC[12] = c'f'
    ESC[13] = c'r'
    ESC[34] = c'"'
    ESC[92] = c'\\'

cdef sqlite3_stmt* prepare(const char* sql) except NULL:
    cdef sqlite3_stmt* st = NULL
    if sqlite3_prepare_v3(g_db, sql, -1, SQLITE_PREPARE_PERSISTENT, &st, NULL) != SQLITE_OK:
        raise RuntimeError("prepare failed: %s (%s)" % (sqlite3_errmsg(g_db).decode(), sql.decode()))
    return st

cdef void db_exec(const char* sql) except *:
    if sqlite3_exec(g_db, sql, NULL, NULL, NULL) != SQLITE_OK:
        raise RuntimeError("%s: %s" % (sql.decode(), sqlite3_errmsg(g_db).decode()))

cdef void on_stop_signal(int sig) noexcept nogil:
    global g_stop
    g_stop = 1

cdef void sqlite_single_thread():
    # No other thread in this process touches SQLite: drop all of its mutexes and memory accounting.
    sqlite3_config(SQLITE_CONFIG_SINGLETHREAD)
    sqlite3_config(SQLITE_CONFIG_MEMSTATUS, 0)
    sqlite3_initialize()


def sqlite_version():
    return sqlite3_libversion().decode()


def hmac_selftest(bytes key, bytes msg):
    """HMAC-SHA256 with this module's implementation (tests compare it with hashlib)."""
    cdef uint8_t out[32]
    init_tables()
    hmac_init(<const uint8_t*><const char*>key, len(key))
    hmac_sha256(<const uint8_t*><const char*>msg, len(msg), out)
    return bytes(out[:32])


def json_string_selftest(bytes s):
    """put_json_str() applied to s (tests compare it with the json module)."""
    cdef char* buf = <char*>malloc(6 * len(s) + 2)
    cdef char* end
    init_tables()
    end = put_json_str(buf, <const uint8_t*><const char*>s, len(s))
    try:
        return buf[:end - buf]
    finally:
        free(buf)


def checkpointer(bytes path, int req_fd, int done_fd):
    """Runs in its own process: a PASSIVE checkpoint each time the server asks for one, answered
    with the number of WAL frames now copied into the database file. The copying and the fsyncs
    happen here, so the event loop never waits on the disk (see wal_hook_cb)."""
    cdef sqlite3* db = NULL
    cdef char buf[64]
    cdef ssize_t n
    cdef int nlog = 0, nckpt = 0
    prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0)
    sqlite_single_thread()
    if sqlite3_open_v2(path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_NOMUTEX, NULL) != SQLITE_OK:
        raise RuntimeError("checkpointer: cannot open database")
    sqlite3_busy_timeout(db, 5000)
    sqlite3_exec(db, b"PRAGMA synchronous = NORMAL", NULL, NULL, NULL)
    with nogil:
        while True:
            n = read(req_fd, buf, 64)
            if n < 0 and errno == EINTR:
                continue
            if n <= 0:
                break
            nckpt = 0
            sqlite3_wal_checkpoint_v2(db, NULL, SQLITE_CHECKPOINT_PASSIVE, &nlog, &nckpt)
            if write(done_fd, &nckpt, sizeof(int)) != sizeof(int):
                break
    sqlite3_close(db)


def serve(int listen_fd, bytes path, bytes secret, int ckpt_req_fd, int ckpt_done_fd, int max_fds):
    """Runs the server on an already listening non-blocking socket until SIGTERM or SIGINT."""
    global g_db, g_lfd, g_ep, g_ckreq_fd, g_ckdone_fd, g_ckpt_disabled, g_max_fds, g_max_conns
    global g_conns, g_rdbuf, g_out, g_out_cap, g_bb, g_bb_cap, g_tmp, g_jwt, g_start
    global g_seg_cap, g_seg_conn, g_seg_start, g_seg_end, g_seg_nreq, g_closeq
    global st_begin, st_commit, st_rollback, st_one, st_feed, st_post, st_ins_post, st_ins_like, st_exists
    cdef epoll_event e
    cdef timespec ts
    cdef int rc
    init_tables()
    hmac_init(<const uint8_t*><const char*>secret, len(secret))

    sqlite_single_thread()
    if sqlite3_open_v2(path, &g_db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_NOMUTEX, NULL) != SQLITE_OK:
        raise RuntimeError("cannot open database %r" % path)
    sqlite3_busy_timeout(g_db, 5000)   # e.g. a crashed predecessor's checkpointer still exiting
    db_exec(b"PRAGMA journal_mode = WAL")
    db_exec(b"PRAGMA synchronous = NORMAL")          # rule 6: WAL + NORMAL, a commit is durable
    db_exec(b"PRAGMA mmap_size = 0")
    db_exec(b"PRAGMA cache_size = -262144")
    db_exec(b"PRAGMA temp_store = MEMORY")
    g_ckreq_fd = ckpt_req_fd
    g_ckdone_fd = ckpt_done_fd
    if ckpt_req_fd >= 0:
        sqlite3_wal_autocheckpoint(g_db, 0)          # the checkpointer process does it
        sqlite3_wal_hook(g_db, wal_hook_cb, NULL)
    else:
        g_ckpt_disabled = True
    st_begin = prepare(b"BEGIN")
    st_commit = prepare(b"COMMIT")
    st_rollback = prepare(b"ROLLBACK")
    st_one = prepare(b"SELECT 1")
    st_feed = prepare(SQL_FEED)
    st_post = prepare(SQL_POST)
    st_ins_post = prepare(SQL_INS_POST)
    st_ins_like = prepare(SQL_INS_LIKE)
    st_exists = prepare(SQL_EXISTS)

    g_max_fds = max_fds
    g_max_conns = max_fds - 64 if max_fds > 1024 else max_fds - 16
    g_conns = <Conn**>calloc(max_fds, sizeof(Conn*))
    g_rdbuf = <char*>malloc(RDBUF_SIZE)
    g_out_cap = 1 << 20
    g_out = <char*>malloc(g_out_cap)
    g_bb_cap = 1 << 16
    g_bb = <char*>malloc(g_bb_cap)
    g_tmp = <char*>malloc(MAX_BODY + 16)
    g_jwt = <uint8_t*>malloc(MAX_HEADER + 16)
    g_seg_cap = MAX_EVENTS * 2
    g_seg_conn = <Conn**>malloc(g_seg_cap * sizeof(Conn*))
    g_seg_start = <size_t*>malloc(g_seg_cap * sizeof(size_t))
    g_seg_end = <size_t*>malloc(g_seg_cap * sizeof(size_t))
    g_seg_nreq = <int*>malloc(g_seg_cap * sizeof(int))
    g_closeq = <Conn**>malloc(g_seg_cap * sizeof(Conn*))
    if (g_conns == NULL or g_rdbuf == NULL or g_out == NULL or g_bb == NULL or g_tmp == NULL
            or g_jwt == NULL or g_seg_conn == NULL or g_seg_start == NULL or g_seg_end == NULL
            or g_seg_nreq == NULL or g_closeq == NULL):
        raise MemoryError()

    g_lfd = listen_fd
    g_ep = epoll_create1(EPOLL_CLOEXEC)
    if g_ep < 0:
        raise OSError(errno, "epoll_create1")
    e.events = EPOLLIN
    e.data.u64 = 0
    e.data.fd = listen_fd
    epoll_ctl(g_ep, EPOLL_CTL_ADD, listen_fd, &e)
    if ckpt_done_fd >= 0:
        e.data.fd = ckpt_done_fd
        epoll_ctl(g_ep, EPOLL_CTL_ADD, ckpt_done_fd, &e)

    update_clock()
    clock_gettime(CLOCK_MONOTONIC_COARSE, &ts)
    g_start = ts.tv_sec
    g_last_sweep = g_now
    signal(SIGTERM, on_stop_signal)    # stop between rounds: every answered write is committed
    signal(SIGINT, on_stop_signal)
    with nogil:
        rc = event_loop()
    if rc != 0:
        raise OSError(errno, "event loop failed")
