class PlipKatherPrompt:
    def __init__(self):
        self.prompt = """ Choose from the below choices, Given image is a hematoxylin and eosin image of: cancer-associated stroma, adipose tissue, debris, lymphocytes, mucus, background, normal colon mucosa, colorectal adenocarcinoma epithelium, smooth muscle """

    def format_prompt(self, text=None):
        return self.prompt
    