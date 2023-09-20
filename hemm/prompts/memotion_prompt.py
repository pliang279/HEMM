class MemotionPrompt:
    def __init__(self):
        self.prompt = "Question: Given the Meme and the following caption, is the meme 0) funny 1) very funny 2) not funny 3) hilarious, Caption:{}. Answer with option choice."

    def format_prompt(self, caption):
        return self.prompt.format(caption)
    