class LNCOCOPrompt:
    def __init__(self):
       self.prompt="""Generate an Image based on the provided caption.\nCaption: {}"""

    def format_prompt(self, text):
        prompt_text = self.prompt.format(text)
        return prompt_text
