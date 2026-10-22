#!/usr/bin/env python3
"""Dependency-free HuggingFace snapshot downloader (urllib only).

Usage: python3 fetch_model.py <repo_id> <dest_dir> [--skip PATTERN]...
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

API = "https://huggingface.co/api/models/{}"
RESOLVE = "https://huggingface.co/{}/resolve/main/{}"


def get_json(url, tries=5):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "littlebit-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def download(url, dest, tries=5):
    tmp = dest + ".part"
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "littlebit-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                total = r.headers.get("Content-Length")
                total = int(total) if total else None
                done = 0
                last_print = 0.0
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last_print > 10:
                        last_print = now
                        pct = f"{100*done/total:.1f}%" if total else "?"
                        print(f"  {os.path.basename(dest)}: {done/1e6:.0f}MB {pct}", flush=True)
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa
            print(f"  retry {i+1} for {dest}: {e}", flush=True)
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"failed to download {url}")


def main():
    repo = sys.argv[1]
    dest = sys.argv[2]
    skip = [a for a in sys.argv[3:]]
    os.makedirs(dest, exist_ok=True)
    info = get_json(API.format(repo))
    files = [s["rfilename"] for s in info.get("siblings", [])]
    print(f"{repo}: {len(files)} files", flush=True)
    for f in files:
        if any(p in f for p in skip) or f.startswith("."):
            print(f"skip {f}", flush=True)
            continue
        target = os.path.join(dest, f)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            print(f"have {f}", flush=True)
            continue
        print(f"get  {f}", flush=True)
        download(RESOLVE.format(repo, f), target)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
