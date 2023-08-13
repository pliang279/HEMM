class OKVQAPrompt:
    def __init__(self):
        self.prompt = """You are given an image and a question. Answer the question in a single word.
                    Question:{}
                    """

    def format_prompt(self, text):
        return self.prompt.format(text)
