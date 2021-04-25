import aiohttp
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher.filters import content_types
from aiogram.utils import executor
import os
import json
import importlib
# import utils


login, password = 'student', 'TH8FwlMMwWvbJF8FYcq0'
bot_token = '1096257761:AAEjHE8icCqMb_3Ruja5GBDYyf3y-SNUfdM'

proxy_auth = aiohttp.BasicAuth(login=login, password=password)
bot = Bot(token=bot_token)
dp = Dispatcher(bot)

start_message = "Привет!\nЯ бот для анализа треков самолетов.\n"
help_message = "Отправь файл такого же вида, как эталон, чтобы получить два файла - с качественными треками и с некачественными\n"
default_message = 'Что-то пошло не так. Проверь формат файла на соответствие эталону.'


async def load_file(file_name):
    importlib.reload(utils)

    time, _, _ = utils.predict(file_name)

    return time


async def send_msg(message, text):
    await dp.bot.send_message(message.chat.id, text)


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await send_msg(message, start_message + help_message)


@dp.message_handler(commands=['help'])
async def send_help(message: types.Message):
    await send_msg(message, help_message)


@dp.message_handler(content_types.DOCUMENT)
async def send_film_info(message: types.File):
    chat_id = message.chat.id
    ## retrieve file_id from document sent by user
    fileID = message.document.file.id
    dp.bot.loadDocument(chat_id = chat_id, document = fileID)
    name = message.document.file.name
    await load_file(name)
    dp.bot.loadDocument(chat_id = chat_id, document = 'GoodTracks' + name)
    dp.bot.loadDocument(chat_id = chat_id, document = 'BadTracks' + name)
    dp.bot.loadDocument(chat_id = chat_id, document = 'hist')



if __name__ == '__main__':
    executor.start_polling(dp)
