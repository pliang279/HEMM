class VCRPrompt:
    def __init__(self):
        self.prompt = "Question: {} Choose from the below choices: 0) {} 1) {} 2) {} 3) {}"
        self.rationale_prompt = "Answer: {}. Question: {} is correct because? Choose from the below choices: 0) {} 1) {} 2) {} 3) {}"

    def format_prompt(self, question, answer_choices, rationale_choices=None, answer_label=None):
        self.prompt.format(question, answer_choices[0], answer_choices[1], answer_choices[2], answer_choices[3])
		
        if answer_label is not None:
            self.rationale_prompt.format(answer_label, answer_label, rationale_choices[0], rationale_choices[1], rationale_choices[2], rationale_choices[3])
            self.rationale_prompt = self.prompt +  self.rationale_prompt
            return self.prompt, self.rationale_prompt
        else:
            return self.prompt, None