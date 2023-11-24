import json
import logging
from enum import Enum
import numpy as np
from aiokafka import AIOKafkaProducer

from config import config


class TypesOfParameters(Enum):
    TEMPERATURE = 1
    PRECIPITATIONS = 2
    ERRORS = 3


def normalize(arr):
    return (arr - np.min(arr)) / \
        (np.max(arr) - np.min(arr))


def generator(edges, prv, lh_r):
    max_value_prv = max(prv(edges[0], lh_r), prv(edges[1], lh_r))
    while True:
        x1 = np.random.rand()
        x2 = np.random.rand()
        x1_ = edges[0] + x1 * (edges[1] - edges[0])
        if max_value_prv * x2 <= prv(x1_, lh_r):
            break
    return x1_


def generator_param(edges, prv, rnd):
    max_value_prv = max(prv(edges[0]), prv(edges[1]))
    while True:
        x1 = edges[0] + rnd.uniform(0, 1) * (edges[1] - edges[0])
        x2 = rnd.uniform(0, 1)
        if x2 <= prv(x1) / max_value_prv:
            break
    return x1


def is_none(a):
    if a is None:
        return True
    if np.isnan(a):
        return True
    return False


def make_data_matrix(square: dict, temperature: dict, precipitation: dict):
    years = list(temperature.keys())
    years.sort()
    data_ = np.ones((len(years), 3))
    for i in range(0, len(years)):
        data_[i, 0] = square[years[i]] if years[i] in square.keys() else None
        data_[i, 1] = temperature[years[i]]
        data_[i, 2] = precipitation[years[i]]
    return data_


async def send_worker_list(mess, key):
    topic = "Workers"
    await send_message(topic, mess, key)


async def send_answer(mess, key):
    topic = "Workers"
    await send_message(topic, mess, key)


async def send_message(topic, mess, key):
    topic = topic
    producer = AIOKafkaProducer(
        bootstrap_servers=config['KAFKA']['bootstrap_server'])
    await producer.start()
    val = bytes(str(mess).replace("'", '"'), 'UTF-8')
    key = bytes(json.dumps(key.replace("'", '"'), indent=4), 'UTF-8')
    try:
        logging.info("Конвертация сообщения в байт-строку {}".format(val))
        logging.info("Отправка сообщения в топик {}".format(topic))
        await producer.send_and_wait(topic=topic, value=val, key=key)
        logging.info("Ответ отправлен")
    except TypeError:
        logging.info("Неудача с отправлением сообщения message {} key {} в топик {}".format(val, key, topic))
    finally:
        await producer.stop()
