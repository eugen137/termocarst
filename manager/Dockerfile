FROM python:3.11-slim
LABEL name=termokarst_manager

WORKDIR /action/workspace
COPY ./requirements.txt ./manager_server.py *.ini /action/workspace/
COPY ./ /action/workspace/manager

RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    && apt-get -y update \
    && apt-get -y install --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

CMD ["/action/workspace/manager_server.py"]
ENTRYPOINT ["python3", "-u"]