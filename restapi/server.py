from aiohttp import web
from aiohttp.web_request import Request


async def handle(request: Request):

    if request.match_info.route.method != "GET":
        return
    data = await request.json()
    print(data)
    text = "Hello!!!"
    return web.Response(text=text)

app = web.Application()
app.add_routes([web.get('/', handle)])

if __name__ == '__main__':
    web.run_app(app)
