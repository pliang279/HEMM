class PathVQAPrompt:
    def __init__(self):
        self.prompt = """Given a Radiology Image and a question. Answer the question in a single word.\nQuestion: {} Short Answer: """
        
    def format_prompt(self, text):
        return self.prompt.format(text)
