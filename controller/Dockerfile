FROM python:3.11-slim
LABEL name=termokarst_controller

WORKDIR /action/workspace
COPY ./ /action/workspace/controller

RUN python3 -m pip install --no-cache-dir -r /action/workspace/controller/requirements.txt \
    && apt-get -y update \
    && apt-get -y install --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

CMD ["/action/workspace/controller/server.py"]
ENTRYPOINT ["python3", "-u"]