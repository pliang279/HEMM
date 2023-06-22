class NoCapsPrompt:
    def __init__(self):
        self.prompt = """You are given an image. This image might contain a lot of objects.
            You have to generate a caption for the image but the caption should just be a single sentence.
            Please do not generate more than one sentences.
            Caption:
            """

    def format_prompt(self):
        return self.prompt
