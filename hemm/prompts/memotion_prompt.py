class MemotionPrompt:
    def __init__(self):
        self.prompt = "Question: Given the Meme and the following caption, Caption:{}. How funny is the meme? Choose from the following comma separated options: funny, very funny, not funny, hilarious."

    def format_prompt(self, caption):
        return self.prompt.format(caption)
    