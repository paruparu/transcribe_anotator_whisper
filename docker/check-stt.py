#!/usr/bin/env python3
"""
文字起こしサーバー（stt_server.py）が動いているかを、音声を1本投げて確かめる。

    python3 docker/check-stt.py talk.m4a --url http://192.168.98.253:28640 --speakers 2

/health → /transcribe の順に呼び、本文・話者の数・かかった時間を出す。
`--expect` に語を並べると、それが本文に入っているかも確かめる（無ければ終了コード 1）。
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--url", default="http://127.0.0.1:28640")
    ap.add_argument("--speakers", type=int)
    ap.add_argument("--no-diarize", action="store_true")
    ap.add_argument("--expect", nargs="*", default=[], help="本文に入っているはずの語")
    ap.add_argument("--token", default=os.environ.get("STT_TOKEN", ""))
    args = ap.parse_args()

    base = args.url.rstrip("/")
    print("health:", urllib.request.urlopen(base + "/health", timeout=10).read().decode())

    q = {"name": os.path.basename(args.audio)}
    if args.speakers:
        q["speakers"] = args.speakers
    if args.no_diarize:
        q["diarize"] = 0
    body = open(args.audio, "rb").read()
    req = urllib.request.Request(base + "/transcribe?" + urllib.parse.urlencode(q), data=body,
                                 headers={"Content-Type": "application/octet-stream"})
    if args.token:
        req.add_header("Authorization", "Bearer " + args.token)
    t0 = time.time()
    res = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    wall = time.time() - t0

    print("\n" + res["text"] + "\n")
    print("音声 %.1f秒 / 往復 %.1f秒 / 内訳 %s / 話者 %s / 行 %d"
          % (res["duration"], wall, res["took"], res["speakers"], len(res["segments"])))
    missing = [w for w in args.expect if w not in res["text"]]
    if missing:
        print("× 本文に無い語: %s" % missing)
        sys.exit(1)
    if args.speakers and not args.no_diarize and len(res["speakers"]) != args.speakers:
        print("× 話者の数が %d ではありません" % args.speakers)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
