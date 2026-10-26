#!/usr/bin/env bash
# --server を、誰も使っていないときに新しい版へ入れ替える（nina で動かす）。
#
#   ops/update-when-idle.sh [待つ分数(既定15)]
#
# 入れ直すと、走っている処理と順番待ちが消える。そのため
#   - どの作業場所も「待つ分数」のあいだ使われていない
#   - 文字起こし・話者分離が走っていない
# の両方を満たすまで2分ごとに見直し、満たしたら git pull して入れ直す。
#
# 走っているかは pgrep で調べる。パターンの頭を [t] のように括弧で囲むのは、
# この見張り自身のコマンド行に当たらないようにするため（囲まないと、自分を
# 「処理中」と見て永久に待つ。2026-10-26 に一度そうなった）。

set -euo pipefail
IDLE_MIN="${1:-15}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${TRANSCRIBE_DATA:-$HOME/tool/transcribe-server-data}"
UNIT="${TRANSCRIBE_UNIT:-transcribe-server}"

busy() {
  local now f
  now=$(date +%s)
  for f in "$DATA"/*/.last-used; do
    [ -f "$f" ] || continue
    if [ $(( now - $(stat -c %Y "$f") )) -lt $(( IDLE_MIN * 60 )) ]; then
      echo "使用中: $(( (now - $(stat -c %Y "$f")) / 60 ))分前に使われた作業場所がある"
      return 0
    fi
  done
  if pgrep -f "[t]ranscribe_cuda.py|[d]iarize_local.py" >/dev/null; then
    echo "使用中: 文字起こし・話者分離が走っている"
    return 0
  fi
  return 1
}

while msg=$(busy); do
  echo "$(date +%H:%M) $msg。2分後に見直す"
  sleep 120
done

cd "$REPO"
git pull -q
systemctl --user restart "$UNIT"
sleep 8
echo "$(date +%H:%M) 反映して入れ直した: $(git log --oneline -1) / $(systemctl --user is-active "$UNIT")"
