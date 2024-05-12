class visualgenomeprompt:
    def __init__(self):
        self.prompt = """Given an image and a question. Answer the question in a single word. Question: {} Short Answer: """

    def format_prompt(self, text):
        return self.prompt.format(text)
    