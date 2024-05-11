class FaceEmotionPrompt:
    def __init__(self):
        self.prompt = "Given an Image of a face, determinte the face expression, choose from the following choices: angry, disgust, fear, happy, neutral, sad, surprise. Answer: "

    def format_prompt(self):
        return self.prompt
    