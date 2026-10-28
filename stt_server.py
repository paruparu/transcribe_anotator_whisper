#!/usr/bin/env python3
"""
文字起こし＋話者分離を HTTP で呼べるようにする（ほかのアプリから使う口）。

    python stt_server.py --host 0.0.0.0 --port 28640

annotator.py が画面から回すのと同じ処理（transcribe_cuda.py の設定で
faster-whisper → diarize_local.py の pyannote → 単語ごとに話者を決めて行で多数決）
を、音声を1本投げると JSON で返す形にしたものです。docker/ の Dockerfile で
suzume-pgx（DGX Spark）に載せて、patch-gm-kit などから呼ぶことを想定しています。

    curl -s --data-binary @talk.m4a \
      "http://<host>:28640/transcribe?name=talk.m4a&speakers=3" | jq .

| | |
|---|---|
| `GET /health` | 生きているか・モデルを読み込み済みか・処理中か |
| `POST /transcribe` | 本文に音声そのもの（m4a / mp3 / wav / webm …）。クエリで指定 |

`/transcribe` のクエリ:

- `name` … 元のファイル名（拡張子で形式を見分ける。既定 `audio.wav`）
- `speakers` … 話者の人数（分かっていれば指定したほうが精度が出る）
- `diarize` … `0` で話者分離をしない（既定 `1`）
- `language` … 既定 `ja`
- `prompt` … 初期プロンプト（既定は transcribe_cuda.py と同じ。空にすると外す）

返り値:

    {"duration": 秒, "language": "ja", "speakers": ["SPEAKER_00", …],
     "segments": [{"start", "end", "text", "speaker"}],
     "words": [{"start", "end", "word", "speaker"}],
     "text": "(SPEAKER_00) …\n(SPEAKER_01) …",
     "took": {"transcribe": 秒, "diarize": 秒}}

- **一度に1本ずつ**処理する（GPU を取り合わないため）。処理中に来たものは順に待つ
- モデルは最初の依頼で読み込み、`--idle-unload` 秒（既定 600）使われなければ手放す。
  pgx は LM Studio と同じメモリを分け合っているため
- `STT_TOKEN` を設定すると `Authorization: Bearer <STT_TOKEN>` の無い依頼を断る
"""

import argparse
import gc
import json
import os
import sys
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from diarize_local import assign_words, load_pipeline
from transcribe_cuda import DEFAULT_MODEL, DEFAULT_PROMPT

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".webm", ".opus")


class Engine:
    """モデルを持ち、1本ずつ処理する。使われなくなったら手放す。"""

    def __init__(self, model_name, device, idle_unload):
        self.model_name = model_name
        self.device = device
        self.idle_unload = idle_unload
        self.whisper = None
        self.diarizer = None
        self.lock = threading.Lock()
        self.busy = False
        self.last_used = time.time()
        threading.Thread(target=self._reaper, daemon=True).start()

    def _load(self, diarize):
        if self.whisper is None:
            from faster_whisper import WhisperModel
            print("モデルを読み込んでいます: %s（%s）" % (self.model_name, self.device), flush=True)
            self.whisper = WhisperModel(self.model_name, device=self.device,
                                        compute_type="float16" if self.device == "cuda" else "float32")
        if diarize and self.diarizer is None:
            print("話者分離モデルを読み込んでいます", flush=True)
            self.diarizer = load_pipeline(self.device)

    def _reaper(self):
        while True:
            time.sleep(30)
            if not self.idle_unload or self.busy:
                continue
            if self.whisper is None and self.diarizer is None:
                continue
            if time.time() - self.last_used < self.idle_unload:
                continue
            with self.lock:
                if time.time() - self.last_used < self.idle_unload:
                    continue
                self.whisper = None
                self.diarizer = None
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                print("しばらく使われないのでモデルを手放しました", flush=True)

    def status(self):
        return {"ok": True, "busy": self.busy, "model": self.model_name,
                "loaded": {"whisper": self.whisper is not None,
                           "diarizer": self.diarizer is not None}}

    def run(self, path, language, prompt, diarize, speakers):
        with self.lock:
            self.busy = True
            try:
                return self._run(path, language, prompt, diarize, speakers)
            finally:
                self.busy = False
                self.last_used = time.time()

    def _run(self, path, language, prompt, diarize, speakers):
        from faster_whisper import decode_audio

        self._load(diarize)
        t0 = time.time()
        # 設定は transcribe_cuda.py と同じ（理由はそちらの冒頭に書いてある）
        segments, info = self.whisper.transcribe(
            path,
            language=language,
            chunk_length=15,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            word_timestamps=True,
        )
        rows, words = [], []
        for s in segments:
            text = (s.text or "").strip()
            if not text:
                continue
            rows.append({"start": s.start, "end": s.end, "text": text})
            words += [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])]
        took = {"transcribe": round(time.time() - t0, 2)}

        speaker_list = []
        if diarize and rows:
            import torch
            t1 = time.time()
            # 音声は faster-whisper と同じ読み方で渡す（16kHz モノラル）。
            # pyannote 側の読み込み（torchcodec）を通さずに済む。
            audio = decode_audio(path, sampling_rate=16000)
            kw = {"num_speakers": speakers} if speakers else {}
            result = self.diarizer({"waveform": torch.from_numpy(audio)[None], "sample_rate": 16000}, **kw)
            annotation = getattr(result, "speaker_diarization", result)
            turns = [(seg.start, seg.end, label)
                     for seg, _, label in annotation.itertracks(yield_label=True)]
            speaker_list = sorted({label for *_, label in turns})
            if turns:
                words = assign_words(words, turns)
                label_rows(rows, words)
            took["diarize"] = round(time.time() - t1, 2)

        return {
            "duration": info.duration,
            "language": info.language,
            "speakers": speaker_list,
            "segments": rows,
            "words": words,
            "text": "\n".join(("(%s) " % r["speaker"] if r.get("speaker") else "") + r["text"]
                              for r in rows),
            "took": took,
        }


def label_rows(rows, words):
    """行ごとに、発話時間で重み付けした多数決で話者を決める（annotator と同じ考え方）。"""
    i = 0
    for r in rows:
        votes = {}
        while i < len(words) and words[i]["start"] < r["end"]:
            w = words[i]
            if w["end"] > r["start"] and w.get("speaker"):
                votes[w["speaker"]] = votes.get(w["speaker"], 0.0) + max(w["end"] - w["start"], 0.01)
            i += 1
        if votes:
            r["speaker"] = max(votes.items(), key=lambda kv: kv[1])[0]


class Handler(BaseHTTPRequestHandler):
    engine = None
    token = ""
    max_bytes = 0

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        if not self.token:
            return True
        return self.headers.get("Authorization", "") == "Bearer " + self.token

    def do_GET(self):
        if urllib.parse.urlparse(self.path).path == "/health":
            self._send(self.engine.status())
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/transcribe":
            self._send({"error": "not found"}, 404)
            return
        if not self._authorized():
            self._send({"error": "unauthorized"}, 401)
            return
        qs = {k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query, keep_blank_values=True).items()}
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._send({"error": "本文に音声を入れてください"}, 400)
            return
        if length > self.max_bytes:
            self._send({"error": "音声が大きすぎます（上限 %d MB）" % (self.max_bytes >> 20)}, 413)
            return
        ext = os.path.splitext(qs.get("name") or "audio.wav")[1].lower()
        if ext not in AUDIO_EXTENSIONS:
            self._send({"error": "対応していない形式です: %s" % ext}, 400)
            return
        try:
            speakers = int(qs["speakers"]) if qs.get("speakers") else None
        except ValueError:
            self._send({"error": "speakers は数で指定してください"}, 400)
            return
        diarize = qs.get("diarize", "1") not in ("0", "false", "no")
        language = qs.get("language") or "ja"
        prompt = qs["prompt"] if "prompt" in qs else DEFAULT_PROMPT

        # 音声は処理のあいだだけ一時ファイルに置き、終わったら消す（残さない）
        fd, path = tempfile.mkstemp(suffix=ext, prefix="stt-")
        try:
            with os.fdopen(fd, "wb") as f:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 1 << 20))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            result = self.engine.run(path, language, prompt, diarize, speakers)
        except Exception as e:
            print("失敗: %r" % e, file=sys.stderr, flush=True)
            self._send({"error": str(e)}, 500)
            return
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        self._send(result)


def main():
    ap = argparse.ArgumentParser(description="文字起こし＋話者分離の HTTP サーバー")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=28640)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--idle-unload", type=int, default=600,
                    help="この秒数使われなければモデルを手放す（0 で持ち続ける）")
    ap.add_argument("--max-upload-mb", type=int, default=1024)
    ap.add_argument("--preload", action="store_true", help="起動時にモデルを読み込んでおく")
    args = ap.parse_args()

    Handler.engine = Engine(args.model, args.device, args.idle_unload)
    Handler.token = os.environ.get("STT_TOKEN", "")
    Handler.max_bytes = args.max_upload_mb << 20
    if args.preload:
        Handler.engine._load(True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("文字起こしサーバー: http://%s:%d/（認証 %s）"
          % (args.host, args.port, "あり" if Handler.token else "なし"), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
