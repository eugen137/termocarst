import asyncio
import json
import logging
import os

from aiokafka import AIOKafkaConsumer
from config import config
from utils import send_answer
from worker import Worker


async def worker_server(worker_id):
    worker_consumer = AIOKafkaConsumer('CurrentTasks', group_id="Worker_nodes",
                                       bootstrap_servers=[config['KAFKA']['bootstrap_server']],
                                       enable_auto_commit=True,
                                       )
    await worker_consumer.start()
    worker = Worker(worker_id)
    # await worker.send_first_message()
    async for msg in worker_consumer:
        logging.info("Мой ИД {}".format(worker.id))
        logging.info("Получено новое сообщение, топик {}".format(msg.topic))
        message = json.loads(msg.value)
        logging.info("Задача назначена этому воркеру "
                     "(текущий id {}, id в сообщения {})".format(worker.id,
                                                                 msg.key.decode('utf-8').replace('"', '')))
        worker.import_from_message(message)
        ans = worker.calc()
        worker_id = worker.id
        worker = Worker()
        worker.id = worker_id
        await send_answer(ans, worker_id)

if __name__ == "__main__":
    id_ = os.environ['HOSTNAME']
    asyncio.run(worker_server(id_))
