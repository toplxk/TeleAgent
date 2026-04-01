import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import tongyi

from Embedding import Embedding
from tools.NetWorkSearch import net_work_search
from tools.tools import get_weather, multipy


load_dotenv()
QIANWEN_API_KEY = os.getenv("QIANWEN_API_KEY")
PROMPT_PATH = Path(__file__).with_name("prompt.txt")

app = FastAPI()
prompt = PromptTemplate.from_template(PROMPT_PATH.read_text(encoding="utf-8"))

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="input",
    return_messages=False,
)

TOOLS = [multipy, get_weather, net_work_search]


@app.get("/")
def read_root():
    return {"hello": "world"}


@app.post("/chat")
def chat(input: str):
    # 加载向量数据库
    db = Embedding().db
    # 向量搜索
    docs = db.similarity_search(input, k=5)
    print("匹配结果")
    for doc in docs:
        print("内容：", doc.page_content)
        print("匹配度：", doc.metadata["score"])
        input += f"\n\n以下是来自文档中的匹配内容：\n\n{doc.page_content}"

    llm = tongyi.Tongyi(api_key=QIANWEN_API_KEY)
    agent = create_react_agent(
        llm=llm,
        tools=TOOLS,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=True,
    )
    return agent_executor.invoke({"input": input})


@app.post("/add_urls")
def add_urls():
    return {"response": "URLs added"}


@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDFs added"}


@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)

    filename = Path(file.filename or "upload.txt").name
    save_path = uploads_dir / filename

    content = await file.read()
    save_path.write_bytes(content)
    await file.close()

    embedding = Embedding()
    embedding.embedding(save_path)

    return {"filename": filename, "save_path": str(save_path)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
