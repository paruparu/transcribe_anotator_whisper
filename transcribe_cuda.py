#!/usr/bin/env python3
"""
NVIDIA の GPU で文字起こしする（faster-whisper 版）。annotator.py --server が使う。

    python transcribe_cuda.py <音声かディレクトリ> [--model large-v3]

`transcribe_mlx.py` と同じ3つのファイルを書き出すので、annotator.py も
diarize_local.py もそのまま読めます。

    <名前>_timecoded.txt  [開始s -> 終了s] 本文
    <名前>_text.txt       本文だけ
    <名前>.words.json     単語の時刻（話者分離で使う。人が読むものではない）

設定は研究室の GPU 機（RTX 4090）で日本語のインタビューに合わせて実測で決めた
値です。

- `chunk_length=15`：30 だと話者が混ざり、8 だと固有名詞が落ちて「特に
  なかった」が「特によかった」に反転した
- `condition_on_previous_text=False`：幻聴の連鎖を防ぐ（transcribe.py と同じ）
- 句読点は Whisper が自分では付けないので、句読点入りの初期プロンプトで
  書き方を誘導する（`--prompt ""` で外せる）
- turbo と量子化版は使わない。日本語の固有名詞が落ちるため（transcribe_mlx.py と同じ理由）
"""

import argparse
import json
import os
import time

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4")
DEFAULT_MODEL = "large-v3"
DEFAULT_PROMPT = ("本日はお忙しいところ、ありがとうございます。"
                  "まず、これまでのご経歴について、お聞かせいただけますか。"
                  "はい、よろしくお願いします。")


def outputs(base):
    """この収録に対して書き出すファイル一式（transcribe_mlx.py と同じ）。"""
    return {
        "timecoded": base + "_timecoded.txt",
        "text": base + "_text.txt",
        "words": base + ".words.json",
    }


def transcribe_one(model, path, language, prompt, overwrite):
    base, _ = os.path.splitext(path)
    out = outputs(base)

    # 黙って上書きしない。手で直した本文が消えるのが一番痛い。
    exists = [p for p in out.values() if os.path.exists(p)]
    if exists and not overwrite:
        print("× %s : 既に結果があります。消してよければ --overwrite を付けてください"
              % os.path.basename(path))
        return False

    print("文字起こし中: %s" % os.path.basename(path), flush=True)
    t0 = time.time()
    segments, info = model.transcribe(
        path,
        language=language,
        chunk_length=15,
        condition_on_previous_text=False,
        initial_prompt=prompt or None,
        word_timestamps=True,
    )
    duration = info.duration or 0

    rows, words = [], []
    last = 0
    for s in segments:            # 生成しながら進むので、ここで進み具合を出す
        text = (s.text or "").strip()
        if not text:
            continue
        rows.append((s.start, s.end, text))
        words += [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])]
        if duration and s.end - last >= 60:
            last = s.end
            print("進み具合 %d%%（%d:%02d / %d:%02d）" % (100 * s.end / duration,
                                                 s.end // 60, s.end % 60,
                                                 duration // 60, duration % 60), flush=True)

    # 途中で落ちたときに半端なものが残らないよう、一時ファイルに書いてから置き換える
    def write(path_, content):
        tmp = path_ + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path_)

    write(out["timecoded"], "".join("[%.2fs -> %.2fs] %s\n" % r for r in rows))
    write(out["text"], "".join("%s\n" % r[2] for r in rows))
    write(out["words"], json.dumps({
        "segments": [{"start": a, "end": b, "text": t} for a, b, t in rows],
        "words": words}, ensure_ascii=False))

    took = time.time() - t0
    speed = (" / %.1f倍速" % (duration / took)) if duration and took else ""
    print("書き出し: %d行 / 単語 %d / %.0f秒%s" % (len(rows), len(words), took, speed), flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description="NVIDIA GPU で文字起こしする（faster-whisper）")
    ap.add_argument("target", help="音声ファイル、または音声の入ったディレクトリ")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="faster-whisper のモデル（既定: %s）" % DEFAULT_MODEL)
    ap.add_argument("--language", default="ja")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="句読点や固有名詞を誘導する初期プロンプト（\"\" で外す）")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--overwrite", action="store_true", help="既にある結果を上書きする")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    if os.path.isdir(args.target):
        targets = [os.path.join(args.target, fn) for fn in sorted(os.listdir(args.target))
                   if fn.lower().endswith(AUDIO_EXTENSIONS)]
        if not targets:
            print("音声が見つかりません: %s" % args.target)
            return
    else:
        targets = [args.target]

    print("モデルを読み込んでいます: %s（%s）" % (args.model, args.device), flush=True)
    model = WhisperModel(args.model, device=args.device,
                         compute_type="float16" if args.device == "cuda" else "float32")
    done = 0
    for path in targets:
        if transcribe_one(model, path, args.language, args.prompt, args.overwrite):
            done += 1
    print("完了: %d / %d 件" % (done, len(targets)), flush=True)
    if done < len(targets):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
