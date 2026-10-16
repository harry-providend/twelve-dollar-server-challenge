"""Entry point: sets the process up in Python, then hands the socket to the Cython event loop.

Environment: SQLITE_PATH, JWT_SECRET, HOST, PORT (see SPEC.md).
"""
import os
import resource
import socket

import fastserver


def raise_fd_limit():
    """Serving directly means one keep-alive connection per user: use the whole descriptor allowance."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = (1 << 20) if hard == resource.RLIM_INFINITY else min(hard, 1 << 20)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        return target
    except (ValueError, OSError):
        return soft


def start_checkpointer(path):
    """Forks the process that checkpoints the WAL whenever the server asks (one byte per request)."""
    req_r, req_w = os.pipe()
    done_r, done_w = os.pipe()
    if os.fork() == 0:
        os.close(req_w)
        os.close(done_r)
        try:
            fastserver.checkpointer(path, req_r, done_w)
        finally:
            os._exit(0)
    os.close(req_r)
    os.close(done_w)
    os.set_blocking(done_r, False)
    return req_w, done_r


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "3000"))
    path = os.environ["SQLITE_PATH"].encode()
    secret = os.environ["JWT_SECRET"].encode()

    max_fds = raise_fd_limit()
    req_fd, done_fd = start_checkpointer(path)  # fork before SQLite is initialised in this process

    family = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][0]
    sock = socket.create_server((host, port), family=family, backlog=65535)
    sock.setblocking(False)
    print(f"listening on {host}:{port} (SQLite {fastserver.sqlite_version()}, fd limit {max_fds})", flush=True)
    fastserver.serve(sock.detach(), path, secret, req_fd, done_fd, max_fds)


if __name__ == "__main__":
    main()
