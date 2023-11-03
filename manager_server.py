import asyncio
import json
import logging

from aiokafka import AIOKafkaConsumer
from manager.config import config
from manager.tasks_manager import TaskManager
from manager.utils import send_to_current_task


async def master_manager():
    # TODO сделать удаление message["parent_ids"].pop(0) при нахождении
    master_manager_consumer = AIOKafkaConsumer(
        'RecoveryRequest', 'ForecastRequest',
        bootstrap_servers=[config['KAFKA']['bootstrap_server']]
    )
    worker_consumer = AIOKafkaConsumer("Workers", bootstrap_servers=[config['KAFKA']['bootstrap_server']])
    # await worker_consumer.start()

    await master_manager_consumer.start()
    task_managers = {}
    server_managers = {}
    async for msg in master_manager_consumer:
        logging.info("Получено новое сообщение, топик {}".format(msg.topic))
        message = json.loads(msg.value)
        type_task = "recovery"
        if msg.topic == 'RecoveryRequest':
            type_task = "recovery"
        elif msg.topic == 'ForecastRequest':
            type_task = "forecast"
        task_manager = TaskManager()
        task_manager.import_from_message(message, type_task=type_task)
        task_managers[task_manager.id] = task_manager
        task_managers[task_manager.id].make_task()
        mess = task_managers[task_manager.id].make_message()
        for m in mess:
            # async for worker_msg in worker_consumer:
            #     print(worker_msg.value)
            await send_to_current_task(m, "data")


asyncio.run(master_manager())
