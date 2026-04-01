import re
import os
from pathlib import Path
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import tongyi
from tools.NetWorkSearch import net_work_search
from tools.tools import get_weather, multipy
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


class IntentRouter:
    """意图路由决策器"""

    def __init__(self, classify_agent):
        self.rules = {
            'RAG': {  # RAG判定
                'keywords': ['小米', '小米创业', '小米公司', '小米手机'],
            }
        }
        self.classify_agent = classify_agent


    def detect(self, user_input : str):
        """意图路由决策器"""
        # 1.移除标点符号(保留中文、英文和数字)
        text_clean = re.sub(r'[^\w\u4e00-\u9fa5]','',user_input)
        # 2. 技术类关键词优先检查
        if any(kw in text_clean for kw in self.rules['RAG']['keywords']):
            return 'RAG'
        else :
            return "LLM"

PROMPT_PATH = Path(__file__).with_name("prompt.txt")

class ReactAgent:
    """agent"""

    def __init__(self):
        self.llm = tongyi.Tongyi(api_key=os.getenv("QIANWEN_API_KEY"))
        self.tools = [multipy, get_weather, net_work_search]
        self.prompt = PromptTemplate.from_template(PROMPT_PATH.read_text(encoding="utf-8"))
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            return_messages=False,
        )
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
        )
        self.agentExecutor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
        )


    def think(self, user_input: str):
        return self.agentExecutor.invoke({"input": user_input})
