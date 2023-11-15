import asyncio
import json
import logging

from aiokafka import AIOKafkaConsumer
from config import config
from utils import send_answer
from worker import Worker


async def worker_server():
    worker_consumer = AIOKafkaConsumer('CurrentTasks',
                                       bootstrap_servers=[config['KAFKA']['bootstrap_server']])
    await worker_consumer.start()
    worker = Worker()
    await worker.send_first_message()
    async for msg in worker_consumer:
        logging.info("Мой ИД {}".format(worker.id))
        logging.info("Получено новое сообщение, топик {}".format(msg.topic))
        message = json.loads(msg.value)
        if msg.key.decode('utf-8').replace('"', '') != worker.id:
            logging.info("Задача пришла не этому воркеру "
                         "(текущий id {}, id сообщения {})".format(worker.id,
                                                                   msg.key.decode('utf-8').replace('"', '')))
            continue
        else:
            logging.info("Задача назначена этому воркеру "
                         "(текущий id {}, я id сообщения {})".format(worker.id,
                                                                     msg.key.decode('utf-8').replace('"', '')))
        worker.import_from_message(message)
        ans = worker.calc()
        id_ = worker.id
        worker = Worker()
        worker.id = id_
        await send_answer(ans, id_)


asyncio.run(worker_server())
