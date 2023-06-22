class NewYorkerCartoonPrompt:
    def __init__(self):
        self.prompt = f"""You are given an cartoon image.
            You have to generate a funny caption for the image. Try to make it as funny as possible.
            Funny Caption:
            """

    def format_prompt(self):
        return self.prompt
