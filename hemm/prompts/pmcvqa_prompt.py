class PMCVQAPrompt:
    def __init__(self):
        self.prompt = """You are given a radiology image and a question. Answer the question in a single word.
                    Question:{} 
                    Choices 
                    A:{},
                    B:{},
                    C:{},
                    D:{} 
                    """

    def format_prompt(self, question, choice_a, choice_b, choice_c, choice_d):
        return self.prompt.format(question, choice_a, choice_b, choice_c, choice_d)