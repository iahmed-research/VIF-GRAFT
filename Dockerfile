FROM python:3.12.9-slim
WORKDIR /work/graft
COPY . .
RUN python -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 \
 && python -m pip install --no-cache-dir '.[test]'
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CMD ["graft", "doctor", "--device", "cuda"]
