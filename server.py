import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

from AutoAgent import ReactAgent
from Embedding import Embedding


load_dotenv()
PROMPT_PATH = Path(__file__).with_name("prompt.txt")

app = FastAPI()


@app.get("/")
def read_root():
    return {"hello": "world"}


@app.post("/chat")
def chat(input: str):
    # 加载向量数据库
    db = Embedding().db
    # 向量搜索
    results = db.similarity_search_with_relevance_scores(input, k=5)

    rag_input = input + "\n\n以下是来自文档中的匹配内容："
    for doc, score in results:
        print("内容：", doc.page_content)
        print("匹配度：", score)
        rag_input += f"\n\n{doc.page_content}"

    agent = ReactAgent()
    return agent.think(rag_input)


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

    try:
        Embedding().embedding(save_path)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"filename": filename, "save_path": str(save_path)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
