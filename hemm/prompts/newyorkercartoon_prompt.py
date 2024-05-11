class NewYorkerCartoonPrompt:
    def __init__(self):
        self.prompt = """Given a cartoon image and a caption.\nCaption: {}. Question: Is the Caption Funny? Short Answer: """

    def format_prompt(self, text):
        return self.prompt.format(text)
