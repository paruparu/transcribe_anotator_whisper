# 文字起こし＋話者分離の HTTP サーバー（stt_server.py）を NVIDIA の GPU 機で動かすイメージ。
# suzume-pgx（DGX Spark・aarch64・GB10）で作って動かすことを想定している。
#
#   docker build -f docker/Dockerfile -t stt-server .      # リポジトリの直下で
#
# 中身は nina の --server（~/tool/whisperX/.venv）と同じ版に揃えてある:
# faster-whisper 1.2.1 / CTranslate2 4.8.1 / pyannote.audio 4.0.7 / torch 2.8.0。
#
# - **CTranslate2 はソースからビルドする。**PyPI の aarch64 版は CPU 専用で、
#   GPU を使えないため。GB10 は sm_121（CUDA_SM=121）
# - torch は PyTorch 公式の aarch64 + CUDA 12.9 版。torchcodec は torch 2.8 に合う
#   aarch64 版が無いので入れない。stt_server.py は音声を波形にしてから pyannote に
#   渡すので、torchcodec が無くても話者分離できる

ARG CUDA_VERSION=12.9.1

# ---- CTranslate2 をビルドする -------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS ct2

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential python3 python3-dev python3-venv \
        libopenblas-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG CT2_VERSION=v4.8.1
# GB10 は sm_121。CMake 3.28 の FindCUDA は「12.1」のような2桁の番号を読めない
# （Unknown CUDA Architecture Name 12.1）ので、アーキテクチャを決める1行を
# nvcc の -gencode の直接指定に置き換える。PTX も入れておき、将来の GPU でも JIT で動くようにする
ARG CUDA_SM=121
# nvcc は1本で数GB使う。pgx は LM Studio とメモリを分け合っているので控えめに
ARG JOBS=6

RUN git clone --depth 1 --branch ${CT2_VERSION} --recurse-submodules \
        https://github.com/OpenNMT/CTranslate2.git /src/ct2
RUN sed -i "s|cuda_select_nvcc_arch_flags(ARCH_FLAGS \${CUDA_ARCH_LIST})|set(ARCH_FLAGS -gencode=arch=compute_${CUDA_SM},code=sm_${CUDA_SM} -gencode=arch=compute_${CUDA_SM},code=compute_${CUDA_SM})|" \
        /src/ct2/CMakeLists.txt \
    && grep -q "compute_${CUDA_SM}" /src/ct2/CMakeLists.txt
RUN cmake -S /src/ct2 -B /src/ct2/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/ct2 \
        -DWITH_CUDA=ON -DWITH_CUDNN=ON \
        -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DOPENMP_RUNTIME=COMP \
        -DCUDA_ARCH_LIST=Common \
    && ninja -C /src/ct2/build -j${JOBS} install
RUN python3 -m venv /opt/build \
    && /opt/build/bin/pip install -q -U pip setuptools wheel pybind11==2.11.1 \
    && cd /src/ct2/python \
    && CTRANSLATE2_ROOT=/opt/ct2 /opt/build/bin/pip wheel . --no-deps --no-build-isolation -w /wheels

# ---- 動かすイメージ --------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv ffmpeg libopenblas0 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ct2 /opt/ct2 /opt/ct2
COPY --from=ct2 /wheels /wheels
ENV LD_LIBRARY_PATH=/opt/ct2/lib:${LD_LIBRARY_PATH}

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}
# torch を先に固定して入れ、あとの依存で入れ替わらないよう constraints で縛る
RUN pip install -q -U pip \
    && pip install -q torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129 \
    && pip freeze | grep -E "^(torch|torchaudio)==" > /tmp/constraints.txt \
    && pip install -q -c /tmp/constraints.txt /wheels/ctranslate2-*.whl \
        faster-whisper==1.2.1 pyannote.audio==4.0.7 \
        av==17.1.0 huggingface_hub==1.24.0 numpy==2.2.6 onnxruntime==1.23.2
# ↑ 最後の4つも nina と同じ版に固定する。av を新しくすると（19.x）、faster-whisper 1.2.1 の
#   decode_audio が「open() got an unexpected keyword argument 'metadata_errors'」で落ちた
# pip が入れる torchcodec は torch 2.8 と合わない版（aarch64 は 0.11 から）なので外す
RUN pip uninstall -y -q torchcodec || true

WORKDIR /app
COPY transcribe_cuda.py diarize_local.py stt_server.py /app/

# モデルのキャッシュ（初回に落ちてくる。large-v3 で約3GB）は /root/.cache/huggingface に
# ボリュームを当てて残す。pyannote のゲート付きモデル用のトークンは
# /root/.config/whisperx/hf_token に読み取り専用で当てる（diarize_local.py が読む）
EXPOSE 28640
CMD ["python3", "stt_server.py", "--host", "0.0.0.0", "--port", "28640"]
