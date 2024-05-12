class MagicBrushPrompt:
    def __init__(self):
       self.prompt="""Edit the given image based on the provided instruction.\nInstruction: {}"""

    def format_prompt(self, text):
        prompt_text = self.prompt.format(text)
        return prompt_text
