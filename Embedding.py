import os
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


class Embedding:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.embedding_model = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=DASHSCOPE_API_KEY,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=10,
            separators=["\n\n", "\n", "。", "!", "?"],
        )
        self.db = Chroma(persist_directory="./chroma_db", embedding_function=self.embedding_model)
        self._initialized = True

    def embedding(self, file_path):
        # 文件路径
        file_path = Path(file_path)

        # 文件类型
        file_type = file_path.suffix.lower()

        match file_type:
            case ".pdf":
                loader = PyPDFLoader(str(file_path))
            case ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            case ".docx":
                try:
                    import docx2txt  # noqa: F401
                except ImportError as exc:
                    raise RuntimeError(
                        "Word file support requires the 'docx2txt' package. Install it with: pip install docx2txt"
                    ) from exc
                loader = Docx2txtLoader(str(file_path))
            case ".doc":
                raise ValueError("Legacy .doc files are not supported. Please convert the file to .docx first.")
            case ".csv":
                loader = CSVLoader(str(file_path), encoding="utf-8")
            case _:
                raise ValueError(f"Unsupported file type: {file_type}")

        documents = loader.load()
        splits = self.text_splitter.split_documents(documents)
        self.db.add_documents(splits)ss
        self.db.persist()
