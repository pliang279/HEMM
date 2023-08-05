class NewYorkerCartoonPrompt:
    def __init__(self):
        self.prompt = """You are given an cartoon image and a caption. start the answer with yes if the caption is funny or No if the caption is not funny.
                    Caption:{}
                    """

    def format_prompt(self, text):
        return self.prompt.format(text)
