

class IntentRouter:
    """意图路由决策器"""

    def __init__(self, classify_agent):
        self.rules = {
            'RAG': {  # RAG判定
                'keywords': ['参数', '规格', '型号', '连接', '对比'],
            }
        }
        self.classify_agent = classify_agent