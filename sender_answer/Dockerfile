FROM python:3.11-slim
LABEL name=web_endpoints

WORKDIR /action/workspace/web
COPY ./ /action/workspace/web

RUN python3 -m pip install --no-cache-dir -r /action/workspace/web/requirements.txt \
    && apt-get -y update \
    && apt-get -y install --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

CMD ["/action/workspace/web/server.py"]
ENTRYPOINT ["python3", "-u"]