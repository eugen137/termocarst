import asyncio
import json
import logging

from aiokafka import AIOKafkaConsumer
from worker.config import config
from worker.utils import send_answer
from worker.worker import Worker


async def worker_server():
    worker_consumer = AIOKafkaConsumer('CurrentTasks',
                                       bootstrap_servers=[config['KAFKA']['bootstrap_server']])
    await worker_consumer.start()
    worker = Worker()
    await worker.send_first_message()
    async for msg in worker_consumer:
        logging.info("Получено новое сообщение, топик {}".format(msg.topic))
        message = json.loads(msg.value)
        if msg.key.decode('utf-8').replace('"', '') != worker.id:
            logging.info("Задача пришла не этому воркеру "
                         "(текущий id {}, я id сообщения {})".format(worker.id,
                                                                     msg.key.decode('utf-8').replace('"', '')))
            continue
        worker.import_from_message(message)
        ans = worker.calc()
        await send_answer(ans, worker.id)


asyncio.run(worker_server())
