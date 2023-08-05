class WinogroundPrompt:
    def __init__(self):
       self.prompt="""
        You are given a an image and a text. Answer yes if the text matches the image and no if the text does not match the image. 
        Text: {}
        Answer:
        """

    def format_prompt(self, text):
        self.prompt = self.prompt.format(text)
        return self.prompt