# -*- coding: utf-8 -*-
"""cocoqa_prompt.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z4xpHfN57tQQ6PvKnSEkPKRkuoTRBnmX
"""

class cocoprompt:
    def __init__(self):
        self.prompt = """You are given an image and a question. Answer the question in a single word only.
                    Question:{}
                    """

    def format_prompt(self, text):
        return self.prompt.format(text)