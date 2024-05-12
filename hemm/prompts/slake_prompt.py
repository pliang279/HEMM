class SlakePrompt:
    def __init__(self):
        self.prompt = """Given a Radiology Image. Answer the question in a single word, Question: {} Short Answer: """

    def format_prompt(self, text=None):
        return self.prompt.format(text)
    