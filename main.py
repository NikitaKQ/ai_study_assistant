import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from dotenv import load_dotenv, find_dotenv, dotenv_values
import os
import openai
import aiohttp
import asyncio

env_file = '.env'

if not os.path.exists(env_file):
    raise ValueError(f"{env_file} file not found in the directory {os.getcwd()}")

config = dotenv_values(env_file)

API_TOKEN = config.get('TELEGRAM_BOT_API_TOKEN')
OPENAI_API_KEY = config.get('OPENAI_API_KEY')
# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

chat_history = {}

# Start command handler
@dp.message(Command(commands=['start']))
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    chat_history[user_id] = []
    await message.reply("Привет! Я твой ассистент по математическому анализу. Напиши /help, чтобы узнать, что я умею.")

# Help command handler
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

# Handle text messages
@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    user_input = message.text

    if user_id not in chat_history:
        chat_history[user_id] = []
    chat_history[user_id].append({"role": "user", "content": user_input})

    # Add assistant response to chat history
    response = await get_openai_response(user_id)
    chat_history[user_id].append({"role": "assistant", "content": response})

    max_message_length = 4096
    for i in range(0, len(response), max_message_length):
        await message.reply(response[i:i + max_message_length])

async def get_openai_response(user_input: int) -> str:
    messages = chat_history[user_input]

    async with aiohttp.ClientSession() as session:
        async with session.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {OPENAI_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'gpt-3.5-turbo',
                'messages': messages,
                'max_tokens': 500,
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
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
