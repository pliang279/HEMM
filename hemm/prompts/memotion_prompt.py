class MemotionPrompt:
    def __init__(self):
        self.prompt = "Question: Given the Meme and the following caption, is the meme funny, very funny, not funny, hilarious, Caption:{}. Answer with the options given above."

    def format_prompt(self, caption):
        return self.prompt.format(caption)
    