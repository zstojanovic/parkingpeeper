import asyncio
import aiosqlite
import time
from aiohttp import web
import cv2
import yolo
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

async def initDb():
    db = await aiosqlite.connect("parking.sqlite")
    count = await db.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='parkingstate';")
    row = await count.fetchone()
    if row[0] == 0:
        log("Creating DB...")
        # TODO close this cursor?
        await db.execute("CREATE TABLE parkingstate (id INTEGER PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, imagefile TEXT NOT NULL, carcount INTEGER NOT NULL);")
    await count.close()
    return db

def captureFrame():
    os.system("libcamera-still -n --immediate -o capture.jpg 2> /dev/null")
    #os.system("libcamera-still -n --immediate --awb=daylight -o capture.jpg 2> /dev/null")
    return cv2.rotate(cv2.imread("capture.jpg"), cv2.ROTATE_90_CLOCKWISE)

async def backgroundTask(app):
    try:
        log("Background task started")
        while True:
            frame = await asyncio.get_event_loop().run_in_executor(app["executor"], captureFrame)
            filename = time.strftime("capture-%Y%m%d-%H%M%S.jpg", time.gmtime()) # TODO use same timestamp for filename and for DB record

            start = time.time()
            carCount = await asyncio.get_event_loop().run_in_executor(app["executor"], app["detector"].detect, frame, filename)
            end = time.time()
            log("YOLO took {:.6f} seconds".format(end - start))


            await app["db"].execute('INSERT INTO parkingstate (imagefile, carcount) values (?, ?);', (filename, carCount))
            await app["db"].commit()
            #await asyncio.sleep(10)
    except asyncio.CancelledError:
        pass
    finally:
        await app["db"].close()
        log("Ending background task")

async def startBackgroundTask(app):
    app['backgroundTask'] = asyncio.create_task(backgroundTask(app))

async def cleanupBackgroundTask(app):
    app['backgroundTask'].cancel()
    await app['backgroundTask']

async def index(request):
    log(request.remote + " requested parking.html")
    return web.FileResponse("parking.html")

async def getCurrentData(request):
    cursor = await request.app["db"].execute("SELECT datetime(timestamp, 'localtime'), imagefile, carcount FROM parkingstate ORDER BY timestamp DESC LIMIT 1;")
    row = await cursor.fetchone()
    carCount = row[2]
    availableSpots = request.app["total"] - carCount
    status = ""
    message = ""
    if (availableSpots <= 0):
        status = "red"
        message = "I don't see an empty spot"
    elif (availableSpots <= request.app["total"] / 20):
        status = "yellow"
        message = "I'm not sure there's an empty spot"
    else:
        status = "green"
        message = "There's an empty spot!"

    data = { "timestamp": row[0], "imageFile": row[1], "carCount": carCount, "total": request.app["total"], "status": status, "message": message }

    return web.json_response(data)

def log(message):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(t + " | " + message)

async def createApp(args):
    app = web.Application()
    app.router.add_route('GET', "/", index)
    app.router.add_route('GET', "/current", getCurrentData)
    app.router.add_static('/images/', path='images/', name='images')
    app.on_startup.append(startBackgroundTask)
    app.on_cleanup.append(cleanupBackgroundTask)

    app["db"] = await initDb()
    app["detector"] = yolo.CarDetector()
    app["total"] = args["total"]
    app["executor"] = ThreadPoolExecutor(2)

    return app

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--total", required=True, type=int, help="total number of parking spots in camera view")
parser.add_argument("-p", "--port", required=True, type=int, help="server port")
args = vars(parser.parse_args())

web.run_app(createApp(args), port=args["port"], access_log=None)
