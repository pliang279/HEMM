class FerPrompt:
    def __init__(self):
        self.prompt = """Given an image of a facial expression, the task is to categorize the image based on the emotion shown in the facial expression into one of seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. The category for the give image is: """

    def format_prompt(self, text=None):
        return self.prompt
    