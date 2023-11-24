import asyncio
import json
import logging
from json import JSONDecodeError

import aiohttp
from aiokafka import AIOKafkaConsumer
from configuration import static_config


async def read_kafka():
    master_manager_consumer = AIOKafkaConsumer(
        'RecoveryAnswer', 'ForecastAnswer', group_id="Answer_Sender",
        bootstrap_servers=[static_config['KAFKA']['bootstrap_server']]
    )
    await master_manager_consumer.start()
    async for msg in master_manager_consumer:
        logging.info("Получено сообщение ответа для задачи в топике {}".format(msg.topic))
        async with aiohttp.ClientSession() as session:
            try:
                mess = json.loads(msg.value)
                async with session.patch(static_config['UI_SERVER']['url_answer'], data=mess,
                                         auth=aiohttp.BasicAuth("administrator", "admin")) as response:
                    status = response.status
                    logging.info("Отправлено сообщение с кодом {}".format(status))
            except JSONDecodeError:
                logging.error("Не удалось отправить сообщение")


if __name__ == '__main__':
    asyncio.run(read_kafka())
