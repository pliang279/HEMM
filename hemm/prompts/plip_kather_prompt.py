class PlipKatherPrompt:
    def __init__(self):
        self.prompt = """Answer the Question from the below Options.\nQuestion: Given Pathalogy image is a hematoxylin and eosin image of:\nOptions: cancer-associated stroma, adipose tissue, debris, lymphocytes, mucus, background, normal colon mucosa, colorectal adenocarcinoma epithelium, smooth muscle\nAnswer:"""

    def format_prompt(self, text=None):
        return self.prompt
    