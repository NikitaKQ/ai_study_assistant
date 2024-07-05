import gradio as gr
import openai
import logging
from controllers.chatbot_controller import ChatBotController
from models.llms import OpenAIModel
import os
from dotenv import load_dotenv, find_dotenv, dotenv_values

# Установите ваш API-ключ OpenAI здесь или загрузите его из .env файла
env_file = '.env'
if not os.path.exists(env_file):
    raise ValueError(f"{env_file} file not found in the directory {os.getcwd()}")

config = dotenv_values(env_file)

openai.api_key = config.get('OPENAI_API_KEY')

# Инициализируйте ChatBotController с использованием модели OpenAI GPT-3.5 Turbo
chatbot = ChatBotController(OpenAIModel("gpt-3.5-turbo", openai.api_key))

undo_button = gr.Button("↩️ Отменить")
clear_button = gr.Button("🗑️ Очистить")

def filter_doc_types(doc_type):
    url_field = False
    file_field = False

    if doc_type == "PDF":
        file_field = True
    else:
        url_field = True
    return gr.Textbox("URL", visible=url_field), gr.File(visible=file_field)

def reset_chat():
    return "", []  # Очистка истории чата

try:
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                prompt_label = gr.Label("Выберите дисциплину")
                prompt_dropdown = gr.Dropdown(
                    label="Дисциплины",
                    choices=["Мат анализ", "Экономика", "Бизнесс и Менеджмент"],
                    value="Мат анализ"
                )
                select_prompt_button = gr.Button("Выбрать дисциплину")
                doc_label = gr.Label("Выберите документ")
                doc_type = gr.Radio(label="Тип документа", value="PDF", choices=["PDF", "WEB", "YouTube"])
                url = gr.Textbox(label="Источник документа", placeholder="URL", visible=False)
                file = gr.File()
                doc_type.change(filter_doc_types, inputs=[doc_type], outputs=[url, file])
                process_button = gr.Button("Process")
                process_button.click(chatbot.set_retrieval, inputs=[doc_type, url, file], outputs=[doc_label])

            with gr.Column(scale=3):
                chat_interface = gr.ChatInterface(
                    chatbot.predict,
                    retry_btn="🔄 Повторить попытку",
                    undo_btn=undo_button,
                    clear_btn=clear_button
                )
                select_prompt_button.click(fn=reset_chat, inputs=[], outputs=[chat_interface])

    if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)
        logging.info("Launching Gradio interface...")
        demo.launch(share=True)
        logging.info("Gradio interface launched successfully!")
except Exception as e:
    logging.error(f"An error occurred: {e}")





