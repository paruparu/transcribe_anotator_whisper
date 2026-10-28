#!/usr/bin/env bash
# suzume-pgx で文字起こしサーバー（stt_server.py）のコンテナを作り直して起動する。
#
#   bash docker/run-pgx.sh           # イメージを作って起動
#   bash docker/run-pgx.sh --no-build
#
# - ポート 28640。DNG-LAN から http://192.168.98.253:28640/ で届く
# - restart=unless-stopped。pgx を再起動しても上がる（止めるなら docker stop stt-server）
# - モデルは ~/.cache/stt-server/hf に残る。話者分離のモデル（ゲート付き）は、そこに
#   キャッシュがあればトークン無しで読める。無いときだけ ~/.config/whisperx/hf_token が要る
# - STT_TOKEN を ~/.config/stt-server/token に置くと、Bearer トークンの無い依頼を断る
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=stt-server:latest
NAME=stt-server
PORT=${PORT:-28640}

if [ "${1:-}" != "--no-build" ]; then
  docker build -f docker/Dockerfile -t "$IMAGE" .
fi

TOKEN_FILE="$HOME/.config/whisperx/hf_token"
if [ ! -s "$TOKEN_FILE" ] && [ ! -d "$HOME/.cache/stt-server/hf/hub/models--pyannote--speaker-diarization-community-1" ]; then
  echo "× 話者分離のモデルのキャッシュも $TOKEN_FILE もありません" >&2
  exit 1
fi
mkdir -p "$HOME/.cache/stt-server/hf"

ENV_ARGS=()
if [ -s "$TOKEN_FILE" ]; then
  ENV_ARGS+=(-v "$TOKEN_FILE:/root/.config/whisperx/hf_token:ro")
fi
if [ -s "$HOME/.config/stt-server/token" ]; then
  ENV_ARGS+=(-e "STT_TOKEN=$(tr -d '\n' < "$HOME/.config/stt-server/token")")
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped --gpus all \
  -p "$PORT:28640" \
  -v "$HOME/.cache/stt-server/hf:/root/.cache/huggingface" \
  "${ENV_ARGS[@]}" \
  "$IMAGE"
echo "起動しました: http://$(hostname -I | awk '{print $1}'):$PORT/health"
