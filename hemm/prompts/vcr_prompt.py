class VCRPrompt:
    def __init__(self):
        self.prompt = "Given an Image and a Question: {}. Choose the correct answet from the choices: {}, {}, {}, {}, Answer: "

    def format_prompt(self, question, answer_choices, rationale_choices=None, answer_label=None):
        return self.prompt.format(question, answer_choices[0], answer_choices[1], answer_choices[2], answer_choices[3])