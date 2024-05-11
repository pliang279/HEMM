class IRFLPrompt:
    def __init__(self):
       self.prompt=""" You are given a simile and a picture along with the simile. You have to say if the simile matches the given picture. Answer the following question in a single word with a yes or no.\nSimile: {}. Answer:"""

    def format_prompt(self, text):
        prompt_text = self.prompt.format(text)
        return prompt_text
