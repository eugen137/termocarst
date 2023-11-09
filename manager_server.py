import asyncio
import json
import logging

from aiokafka import AIOKafkaConsumer
from manager.configuration import static_config
from manager.queue import Queue


async def master_manager():
    master_manager_consumer = AIOKafkaConsumer(
        'RecoveryRequest', 'ForecastRequest', 'Workers',
        bootstrap_servers=[static_config['KAFKA']['bootstrap_server']]
    )
    await master_manager_consumer.start()
    queue = Queue()

    async for msg in master_manager_consumer:
        logging.info("Получено новое сообщение, топик {}".format(msg.topic))
        message = json.loads(msg.value)
        if msg.topic in ('RecoveryRequest', 'ForecastRequest'):
            type_task = "recovery"
            if msg.topic == 'RecoveryRequest':
                type_task = "recovery"
            elif msg.topic == 'ForecastRequest':
                type_task = "forecast"

            await queue.add_main_task_message(message, type_task)

        if msg.topic == 'Workers':
            # сперва нужно определить, это воркер свободен
            worker_id = msg.key.decode('utf-8')
            await queue.add_worker_message(worker_id, message)


asyncio.run(master_manager())
