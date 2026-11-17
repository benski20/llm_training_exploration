FROM python:3.13-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 OMP_WAIT_POLICY=PASSIVE
WORKDIR /home/user/app

COPY --chown=user constraints.txt /home/user/app/constraints.txt
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user pgw /home/user/app/pgw
RUN pip install --no-cache-dir -c constraints.txt "./pgw[gateway,service,factory,pack]"

USER user
# Model pack (int8 weights, fingerprint, tokenizer) is built into the image. One-time masks and all
# keys are generated when the container starts and never leave it.
RUN python pgw/scripts/client_demo/run.py --pack-only --pack-dir /home/user/pack --work /home/user/.pgw-build \
    && rm -rf /home/user/.cache/huggingface /home/user/.pgw-build

EXPOSE 7860
CMD ["python", "pgw/scripts/client_demo/run.py", "--host", "0.0.0.0", "--port", "7860", \
     "--pack-dir", "/home/user/pack", "--work", "/home/user/demo", \
     "--slots", "300", "--refill-low", "250", "--refill-target", "600"]
