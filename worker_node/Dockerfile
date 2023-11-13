FROM python:3.11-slim
LABEL name=termokarst_worker

WORKDIR /action/workspace/worker_node
COPY ./ /action/workspace/worker_node

RUN python3 -m pip install --no-cache-dir -r /action/workspace/worker_node/requirements.txt \
    && apt-get -y update \
    && apt-get -y install --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

CMD ["/action/workspace/worker_node/server.py"]
ENTRYPOINT ["python3", "-u"]