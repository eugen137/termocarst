import jsonschema
from aiohttp import web
from aiohttp.web_request import Request
from jsonschema import validate
from utils import send_task, message_schema


async def web_server(request: Request):

    if request.match_info.route.method != "GET":
        return
    data = await request.json()

    try:
        validate(instance=data, schema=message_schema)
    except jsonschema.exceptions.ValidationError as err:
        return web.HTTPBadRequest(text="Неверный формат, ошибка{}".format(err))

    if request.path_qs == "/forecast":
        topic = "ForecastRequest"
        answer = "Задача прогнозирования принята"
    else:
        topic = "RecoveryRequest"
        answer = "Задача восстановления принята"
    await send_task(data, "data", topic)
    return web.Response(text=answer)


if __name__ == '__main__':
    app = web.Application()
    app.add_routes([web.get('/forecast', web_server),
                    web.get('/recovery', web_server)])
    web.run_app(app)

