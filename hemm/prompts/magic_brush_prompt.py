class MagicBrushPrompt:
    def __init__(self):
       self.prompt="""
        Edit the given image based on the provided instruction. 
        Instruction: {}
        """

    def format_prompt(self, text):
        self.prompt = self.prompt.format(text)
        return self.prompt
