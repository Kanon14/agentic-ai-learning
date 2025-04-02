import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

system_prompt = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate([
    ("system", system_prompt),
    ("user", "{text}"),
])

chain = prompt_template | llm | parser

app = FastAPI(
    title="LangServe Learning: Simple Translation Service",
    version="1.0",
    description="A simple API server using LangChain's Runnable interface",
)

add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)