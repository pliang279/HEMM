class SlakePrompt:
    def __init__(self):
        self.prompt = """ Answer the question in a single word, Question: {} """

    def format_prompt(self, text=None):
        return self.prompt.format(text)
    