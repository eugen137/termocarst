import logging

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import asyncio
from src.config import config
from src.forecast import Forecasting
from src.recover import Recovering


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


async def forecast(msg):
    forecasting = Forecasting(forecast_type="randomize_modeling", precipitation=None, temperature=None,
                              square=None,
                              period_type=None, task_id=None)
    if forecasting.import_from_message(msg.value):
        logging.info("Импортированы данные из сообщения")
        forecast_square = forecasting.forecast()
    else:
        logging.error("Данные из сообщения не удалось импортировать")
        forecast_square = None
    logging.info("Начало отправки ответа")
    await send_answer(mess=forecast_square, key=msg.key, topic=msg.topic)


async def consume():
    consumer = AIOKafkaConsumer(
        'RecoveryRequest', 'ForecastRequest',
        bootstrap_servers=[config['KAFKA']['bootstrap_server']])
    await consumer.start()
    async for msg in consumer:

        logging.info("Получено новое сообщение, топик {}".format(msg.topic))
        if msg.topic == 'RecoveryRequest':
            recovery = Recovering("polynomial")
            if recovery.import_from_message(msg.value):
                logging.info("Импортированы данные из сообщения")
                recovered_square = recovery.get_recovered_square()
            else:
                logging.error("Данные из сообщения не удалось импортировать")
                recovered_square = None
            logging.info("Начало отправки ответа")
            await send_answer(mess=recovered_square, key=msg.key, topic=msg.topic)
        if msg.topic == 'ForecastRequest':
            await forecast(msg)
    try:
        pass
    finally:

        await consumer.stop()


asyncio.run(consume())
