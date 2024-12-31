from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.storage.agent.postgres import PgAgentStorage
from phi.vectordb.pgvector import PgVector, SearchType
from phi.playground import Playground, serve_playground_app

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 1. Traditional RAG
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Initialize the knowledge base with the PDF file
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(table_name="recipes", 
                       db_url=db_url, 
                       search_type=SearchType.hybrid,
                       embedder=OpenAIEmbedder(model="text-embedding-3-small"),
                       ),
    )

knowledge_base.load(upsert=True)

# Traditional RAG
# agent = Agent(
#     model=OpenAIChat(id="gpt-4o-mini"),
#     knowledge_base=knowledge_base,
#     add_context=True, 
#     search_knowledge=False,
#     markdown=True
# )

# Agentic RAG
# agent = Agent(
#     model=OpenAIChat(id="gpt-4o-mini"),
#     knowledge_base=knowledge_base,
#     add_context=True, 
#     search_knowledge=True,
#     show_tool_calls=True,
#     markdown=True
# )

# agent.print_response(
#     "Hi, I want to make a three course meal. Can you recommend some recipes. "
#     "I'd like to start with a soup, then I am thinking a thai curry for the main course "
#     "and finish with a dessert", 
#     stream=True)

# UI
rag_agent = Agent(
    name="RAG Agent",
    agent_id="rag-agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    knowledge_base=knowledge_base,
    search_knowledge=True,
    read_chat_history=True,
    storage=PgAgentStorage(table_name="rag_agent_sessions", db_url=db_url), 
    instructions=[
        "Always search your knowledge base first and use it if available.",
        "Share the page number or source URL of the information you used in your response.",
        "If health benefits are mentioned, include them in the response.",
        "Important: Use tables where possible.",
    ], 
    markdown=True
)

app = Playground(agents=[rag_agent]).get_app()
if __name__ == "__main__":
    knowledge_base.load(upsert=True)
    serve_playground_app(app)