import enum
import logging

from aiokafka import AIOKafkaProducer

from manager.config import config


class State(enum.Enum):
    created = 0
    running = 1
    finished = 2
    wait = 3
    failed = 4


class ServerState(enum.Enum):
    busy = 0
    free = 1
    failed = 2


async def send_to_current_task(mess, key):
    topic = "CurrentTasks"
    producer = AIOKafkaProducer(
        bootstrap_servers=config['KAFKA']['bootstrap_server'])
    await producer.start()
    try:
        val = str.encode(str(mess))
        key = str.encode(key)
        logging.info("Конвертация сообщения в байт-строку {}".format(val))
        logging.info("Отправка сообщения в топик {}".format(topic))
        await producer.send_and_wait(topic=topic, value=val, key=key)
        logging.info("Ответ отправлен")
    except TypeError:
        logging.info("Неудача с отправлением сообщения message {} key {} в топик {}".format(mess, key, topic))
    finally:
        await producer.stop()


async def send_answer(mess, key, topic):
    topic_answer = 'RecoveryAnswer' if topic == 'RecoveryRequest' else 'ForecastAnswer'
    producer = AIOKafkaProducer(
        bootstrap_servers=config['KAFKA']['bootstrap_server'])
    await producer.start()
    try:
        val = str.encode(str(mess))
        logging.info("Конвертация сообщения в байт-строку {}".format(val))
        logging.info("Отправка сообщения в топик {}".format(topic_answer))
        await producer.send_and_wait(topic=topic_answer, value=val, key=key)
        logging.info("Ответ отправлен")
    except TypeError:
        logging.info("Неудача с отправлением сообщения {}".format(mess))
    finally:
        await producer.stop()