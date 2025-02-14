from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, BrowserConfig, Browser, Controller
import asyncio
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel

import os
import json

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

class List(BaseModel):
    extension: str
    content: dict
    title: str

controller = Controller(output_model=List)
model="gemini"

async def main():
    config = BrowserConfig(
        headless=False,
        disable_security=True
    )
    browser = Browser(config=config)

    llm = ChatOpenAI(model="gpt-4o")  # Valor por defecto

    if model == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)

    agent = Agent(
        task="quiero que ingreses a este sitio https://main.un.org/securitycouncil/en/sanctions/1591/materials y me devuelvas el contenido del XML que esta en 'List in alphabetical order' y devuelve me el contenido en XML",
        llm=llm,
        browser=browser,
        save_conversation_path="logs/tasks",
        controller=controller
        
    )
    history = await agent.run()

    result = history.final_result()
    if result:
        parsed: List = List.model_validate_json(result)

        # Crear carpeta si no existe
        os.makedirs('lists', exist_ok=True)

        # Generar nombre de archivo con fecha
        filename = f"lists/item_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Escribir el contenido en el archivo
        with open(filename, 'w') as f:
            json.dump(parsed.model_dump(), f)  # Guardar el contenido en formato JSON

        print(f"Contenido guardado en {filename}")
    else:
        print("no result")

    await browser.close()
    
if __name__ == '__main__':
    asyncio.run(main())