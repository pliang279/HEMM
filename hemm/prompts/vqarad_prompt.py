class VQARADPrompt:
    def __init__(self):
        self.prompt = """Given a Radiology image and a Question. Answer the question in a single word.\nQuestion: {}\nAnswer: """

    def format_prompt(self, text):
        return self.prompt.format(text)
        