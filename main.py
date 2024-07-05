import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import BotCommand, BotCommandScopeDefault, InputFile
from dotenv import load_dotenv, find_dotenv, dotenv_values
import os
import openai
import aiohttp
import asyncio
from PIL import Image
import matplotlib.pyplot as plt
import io
import re
import tempfile

env_file = '.env'

if not os.path.exists(env_file):
    raise ValueError(f"{env_file} file not found in the directory {os.getcwd()}")

config = dotenv_values(env_file)

API_TOKEN = config.get('TELEGRAM_BOT_API_TOKEN')
OPENAI_API_KEY = config.get('OPENAI_API_KEY')
logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

openai.api_key = OPENAI_API_KEY

chat_history = {}

async def set_commands(bot: Bot):
    commands = [
        BotCommand(command='/start', description='Запустить бота'),
        BotCommand(command='help', description='Возможности бота'),
        BotCommand(command='delete_memory', description='Стереть контекст')
    ]
    await bot.set_my_commands(commands, scope=BotCommandScopeDefault())

@dp.message(Command(commands=['delete_memory']))
async def delete_memory(message: types.Message):
    user_id = message.from_user.id
    if user_id in chat_history:
        del chat_history[user_id]
        await message.reply('История сообщений была успешно удалена.')
    else:
        await message.reply("У вас нет сохраненной истории сообщений.")

@dp.message(Command(commands=['start']))
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    chat_history[user_id] = []
    await message.reply("Привет! Я твой ассистент по математическому анализу. Напиши /help, чтобы узнать, что я умею.")

@dp.message(Command(commands=['help']))
async def send_help(message: types.Message):
    help_text = (
        "Я могу помочь с математическим анализом:\n"
        "- Проверять домашние задания\n"
        "- Объяснять материал понятным языком\n"
        "- Отвечать на вопросы по предмету\n"
        "- Давать подсказки по домашнему заданию\n"
        "Просто задай мне вопрос, и я постараюсь помочь!"
    )
    await message.reply(help_text)

@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    user_input = message.text

    if user_id not in chat_history:
        chat_history[user_id] = []
    chat_history[user_id].append({"role": "user", "content": user_input})

    response = await get_openai_response(user_id)
    chat_history[user_id].append({"role": "assistant", "content": response})

    max_message_length = 4096
    for i in range(0, len(response), max_message_length):
        await message.reply(response[i:i + max_message_length])

async def get_openai_response(user_input: int) -> str:
    messages = chat_history[user_input]

    system_prompt = {
        "role": "system",
        "content": "Ты - ассистент преподавателя по математическому анализу. Помогай с проверкой домашних заданий, объясняй материал, отвечай на вопросы по предмету и давай подсказки по домашнему заданию, но никогда не давай готовых решений задач. Все формулы ты пишешь обычным текстом, а вычисления делаешь строго в python. Код ты не выводишь, Вычисления прописываешь текстом. Также ты можешь проверять домашние задания и ставить оценку от 0 до 10, где 0 это пустое или полностью неправильно домашнее задание, а 10 это идеальное домашнее задание. При проверке домашнего задания ты сначала говоришь оценку, а потом говоришь в каких номерах есть ошибки и потом более детально их разбираешь. В чате не поддреживается латех так что пиши формулы без всяких \\ чтобы было понятно что написано обычным текстом. Нельзя использовать LaTex в ответе, только по просьбе ученика!!!"
    }

    messages.insert(0, system_prompt)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {OPENAI_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'gpt-4o',
                'messages': messages,
                'max_tokens': 1500,
                'temperature': 0.7,
            }
        ) as resp:
            data = await resp.json()

            if 'choices' not in data:
                error_message = data.get('error', {}).get('message', 'Unknown error')
                logging.error(f"OpenAI API error: {error_message}")
                return f"Произошла ошибка при запросе к OpenAI: {error_message}"

            return data['choices'][0]['message']['content'].strip()

async def main():
    await set_commands(bot)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
