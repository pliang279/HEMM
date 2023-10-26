class FaceEmotionPrompt:
    def __init__(self):
        self.prompt = "Given the photo of a face, determinte the face expression, choose from the following choices: angry, disgust, fear, happy, neutral, sad, surprise. Answer in a single word."

    def format_prompt(self):
        return self.prompt
    