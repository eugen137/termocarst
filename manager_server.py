import asyncio
import json
import logging

from aiokafka import AIOKafkaConsumer
from manager.config import config
from manager.tasks_manager import TaskManager
from manager.utils import send_to_current_task, WorkerState


async def master_manager():
    # TODO сделать удаление message["parent_ids"].pop(0) при нахождении
    master_manager_consumer = AIOKafkaConsumer(
        'RecoveryRequest', 'ForecastRequest', 'Workers',
        bootstrap_servers=[config['KAFKA']['bootstrap_server']]
    )

    await master_manager_consumer.start()
    task_managers = {}
    worker_managers = {}
    messages_for_workers = []

    async for msg in master_manager_consumer:
        logging.info("Получено новое сообщение, топик {}".format(msg.topic))
        message = json.loads(msg.value)
        if msg.topic in ('RecoveryRequest', 'ForecastRequest'):
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
            messages_for_workers.extend(mess)

        if msg.topic == 'Workers':
            # cперва нужно определить, это воркер свободен
            worker_id = msg.key.decode('utf-8')
            print(worker_id)
            if worker_id in worker_managers.keys():
                # нужно проверить были ли задачи у воркера
                if worker_managers[worker_id].state == WorkerState.busy:
                    # значит воркер был занят
                    if message["state"] != WorkerState.free.name:
                        logging.error("Ошибка получения статуса воркера {} s")
            if len(messages_for_workers) != 0:
                print(messages_for_workers.pop(0))



asyncio.run(master_manager())
