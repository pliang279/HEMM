class MemeCapsPrompt:
    def __init__(self):
        self.prompt = "This is a meme with the title {}. The image description is {}. What is the meme poster trying to convey? Answer:"

    def format_prompt(self, title, image_description):
        return self.prompt.format(title, image_description)
