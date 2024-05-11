class ScienceQAPrompt:
    def __init__(self):
        self.prompt = "You are given an image, question and few choices. A context is also provided to help you understand the image. In order to answer the question, you have been given lecture notes. You can use these lecture notes, image, and the context in order to answer the question. There are some choices given to you which are comma separated. You have to select which choice best answers the question. Generate choice as it is from the given choices.\nLecture: {}\nQuestion: {}\nContext: {}\nChoices: {}\nAnswer:"

    def format_prompt(self, lecture, question, context, choices):
        choices = ", ".join(choices).strip()
        prompt_text = self.prompt.format(lecture, question, context, choices)
        return prompt_text
    