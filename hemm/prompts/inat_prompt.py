class INATPrompt:
    def __init__(self):
        self.prompt = """The scientific species name of the species present in the image is: """

    def format_prompt(self, text=None):
        return self.prompt.format()
