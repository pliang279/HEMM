class FaceEmotionPrompt:
    def __init__(self):
        self.prompt = "Question: Given the face photo is the expression 0) angry, 1) disgust, 2) fear, 3) happy, 4) neutral, 5) sad, 6) surprise. Answer with option choice."

    def format_prompt(self):
        return self.prompt