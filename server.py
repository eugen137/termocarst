import logging
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import asyncio
from config import config
from src.recovery import Recovery


async def send_answer(mess, key, topic, headers):
    topic_answer = 'RecoveryAnswer' if topic == 'RecoveryRequest' else 'ForecastAnswer'
    producer = AIOKafkaProducer(
        bootstrap_servers=config['KAFKA']['bootstrap_server'])
    await producer.start()
    try:
        await producer.send_and_wait(topic_answer, value=mess, key=key, headers=list(headers))
    finally:
        await producer.stop()


async def consume():
    consumer = AIOKafkaConsumer(
        'RecoveryRequest', 'ForecastRequest',
        bootstrap_servers=[config['KAFKA']['bootstrap_server']])
    await consumer.start()
    async for msg in consumer:
        logging.info("Получено новое сообщение, топик {}".format(msg.topic))
        if msg.topic == 'RecoveryRequest':
            recovery = Recovery("polynomial")
            if recovery.import_from_message(msg.value):
                logging.info("Импортированны данные из сообщения")
                sq = recovery.get_recovered_square()
            else:
                logging.info("Данные из сообщения не удалось импортировать")
        await send_answer(mess=msg.value, key=msg.key, topic=msg.topic, headers=msg.headers)

    try:
        pass
    finally:
        await consumer.stop()


asyncio.run(consume())
