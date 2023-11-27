import enum
import json
import logging
import numpy as np
from aiokafka import AIOKafkaProducer
from configuration import static_config


async def send_task(mess, key, topic_task):
    producer = AIOKafkaProducer(
        bootstrap_servers=static_config['KAFKA']['bootstrap_server'])
    await producer.start()
    try:
        val = bytes(json.dumps(mess, indent=4), 'UTF-8')
        key = bytes(key, 'UTF-8')
        logging.info("Конвертация сообщения в байт-строку {}".format(val))
        logging.info("Отправка сообщения в топик {}".format(topic_task))
        await producer.send_and_wait(topic=topic_task, value=val, key=key)
        logging.info("Задача отправлена")
    except TypeError:
        logging.info("Неудача с отправлением сообщения {}".format(mess))
    finally:
        await producer.stop()


message_schema = {
        "type": "object",
        "properties": {
            "data": {
                "description": "Данные для задачи в формате {[имя_параметра:{[год:значение]}]}",
                "type": "object",
                "patternProperties": {
                    "^.*$": {
                        "type": "object",
                        "patternProperties": {
                            "^[0-9]*$": {"type": "number"}
                        },
                        "additionalProperties": False
                    }
                }
            },
            "period_type": {
                "description": "Временной промежуток прогнозирования (short, middle, long)",
                "enum": ["short", "middle", "long"]},
            "count_of_trajectories": {
                "description": "Количество сэмплированных траекторий",
                "type": "number"},
            "id": {
                "description": "Уникальный идентификатор задачи",
                "type": "string"},
            "main_param_name": {
                "description": "Имя целевого параметра",
                "type": "string"},
        },
        "required": ["id", "data"]
    }
