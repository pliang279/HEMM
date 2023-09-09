class EnricoPrompt:
    def __init__(self):
        self.prompt = """Given a screenshot of the user interface of a mobile application.
                            Choose the most appropriate design topic from the following comma separated choices:
                            bare, dialer, camera, chat, editor, form, gallery, list, login, maps, mediaplayer, menu, modal, news, other, profile, search, settings, terms, tutorial 
           """

    def format_prompt(self, text=None):
        return self.prompt
    