import asyncio
import os
import sys
import time

sys.path.append('../')

from bfxapi import Client

now = int(round(time.time() * 1000))

API_KEY = os.getenv("BFX_KEY")
API_SECRET = os.getenv("BFX_SECRET")

bfx = Client(
    API_KEY=API_KEY,
    API_SECRET=API_SECRET,
    logLevel='DEBUG',
    rest_host='https://test.bitfinex.com/v2'
)


async def log_historical_candles(past_day):
    then = now - (1000 * 60 * 60 * 24 * past_day)  # 10 days ago
    candles = await bfx.rest.get_candles('tBTCUSD', 0, then)
    print("Candles {}:".format(past_day))
    [print(c) for c in candles]


async def log_historical_trades(past_day):
    print(API_KEY)
    then = now - (1000 * 60 * 60 * 24 * past_day)  # 10 days ago
    trades = await bfx.rest.get_trades('tBTCUSD', 0, then)
    print("Trades {}:".format(past_day))
    [print(t) for t in trades]


@bfx.ws.on('error')
def log_error(err):
    print("Error: {}".format(err))


@bfx.ws.on('new_candle')
def log_candle(candle):
    print("New candle: {}".format(candle))


@bfx.ws.on('new_trade')
def log_trade(trade):
    print("New trade: {}".format(trade))


async def get_live_data():
    await bfx.ws.subscribe('candles', 'tBTCUSD', timeframe='1m')
    await bfx.ws.subscribe('trades', 'tBTCUSD')
    await bfx.ws.subscribe('ticker', 'tBTCUSD')


def live_data():
    bfx.ws.on('connected', get_live_data)
    bfx.ws.run()


async def historical_data(past_day):
    await log_historical_candles(past_day)
    await log_historical_trades(past_day)


def collect_historical_data(past_day):
    t = asyncio.ensure_future(historical_data(past_day))
    asyncio.get_event_loop().run_until_complete(t)
