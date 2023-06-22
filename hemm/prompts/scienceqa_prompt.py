class ScienceQAPrompt:
    def __init__(self):
        self.prompt = """You are given a question and few choices. There is context provided with the image which will help you to understand the image.
            In order to answer the question, you have been given lecture notes. You can use these lecture notes, image, context in order to answer the question. There are some choices given to you which are comma separated.
            You have to select which choice best answers the question. You also have to provide an explanation as to why the choice is the best answer for the question. Generate choice as it is from the given choices.
            lecture: {}
            question: {}
            context: {}
            choices: {}
            You have to generate answers in the format ->
            Choice:
            Explanation:
            """

    def format_prompt(self, lecture, question, context, choices):
        self.prompt = self.prompt.format(lecture, question, context, choices)
        return self.prompt