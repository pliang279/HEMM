class WinogroundPrompt:
    def __init__(self):
       self.prompt="""Given an image and a text. Answer yes if the text matches the image and no if the text does not match the image.\nText: {}\nAnswer: """

    def format_prompt(self, text):
        return self.prompt.format(text)
    