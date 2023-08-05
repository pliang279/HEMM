class IRFLPrompt:
    def __init__(self):
       self.prompt="""
        You are given a simile and a picture along with the similie. You have to say if the similie matches the given picture. Answer the following question in a single word with a yes or no.
        Similie: {}
        Answer:
        """

    def format_prompt(self, text):
        self.prompt = self.prompt.format(text)
        return self.prompt
