class HatefulMemesPrompt:
    def __init__(self):
        self.prompt = """Given a meme with {} written on it. Is it hateful? Answer: """

    def format_prompt(self, text):
        prompt_text = self.prompt.format(text)
        return prompt_text
    