class FaceEmotionPrompt:
    def __init__(self):
        self.prompt = "Question: Given the face photo is the expression angry, disgust, fear, happy, neutral, sad, surprise. Answer with option choice."

    def format_prompt(self):
        return self.prompt