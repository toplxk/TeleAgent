import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


class Embedding(object):

    _instance = None
    # 单例模式
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            return cls._instance

    def __init__(self):
        self.embedding_model = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=DASHSCOPE_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=10,
            separators=["\n\n","\n","。","!"]
        )
        # 创建向量数据库
        self.db = Chroma()
        # self.db = Chroma.from_documents(splits, embedding=self.embedding_model, persist_directory="./chroma_db")


    def embedding(self, file_path):
        # 获取文件类型
        file_type = file_path.suffix.lower()
        loader = None
        match file_type:
            case ".pdf":
                loader = PyPDFLoader(file_path)
            case ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            case ".docx" | ".doc":
                try:
                    import docx2txt  # noqa: F401
                except ImportError as exc:
                    raise RuntimeError(
                        "Word file support requires the 'docx2txt' package. Install it with: pip install docx2txt"
                    ) from exc
                loader = Docx2txtLoader(file_path)
            case ".csv":
                loader = CSVLoader(file_path, encoding="utf-8")
            case _:
                raise ValueError(f"Unsupported file type: {file_type}")
        # 加载文件
        documents = loader.load()
        # 分割文本
        splits = self.text_splitter.split_documents(documents)
        # 创建并保存到本地的Chroma数据库
        self.db.from_documents(splits, embedding=self.embedding_model, persist_directory="./chroma_db")
