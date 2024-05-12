class MemotionPrompt:
    def __init__(self):
        self.prompt = "Given the Meme and the following caption\nCaption: {}.\nQuestion: How funny is the meme? Choose from the following comma separated options: funny, very funny, not funny, hilarious. Answer: "

    def format_prompt(self, caption):
        return self.prompt.format(caption)
    