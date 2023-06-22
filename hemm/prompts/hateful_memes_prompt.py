class HatefulMemesPrompt:
    def __init__(self):
        self.prompt = """You are given an image. In the image, the
           text phrase which you will be given and the image are innocuous when considered by themselves.
           The semantic content of the meme becomes mean only when the text phrase and image are considered together.
           Text phrase: {}
           You have to judge if the combination of image and text is hateful or not.
           Always begin your answer with either 'yes' or 'no' with 'yes' indicating that the meme is hateful and 'no' if it is not hateful.
           Answer:
           """

    def format_prompt(self, text):
        self.prompt = self.prompt.format(text)
        return self.prompt
