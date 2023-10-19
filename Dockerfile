FROM python:3.11-slim
LABEL name=termokarst_square_forecasting

WORKDIR /action/workspace
COPY requirements.txt *.py *.ini /action/workspace/
COPY ./src /action/workspace/src

RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    && apt-get -y update \
    && apt-get -y install --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

CMD ["/action/workspace/server.py"]
ENTRYPOINT ["python3", "-u"]