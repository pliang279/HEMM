class MMIMDBPrompt:
    def __init__(self):
        self.prompt = """Given the movie poster and the corresponding plot of the movie, choose the appropriate genres from the following comma separated genres: drama, comedy, romance, thriller, crime, action, adventure, horror, documentry, mystery, sci-fi, fantasy, family, biography, war, history, music, animation, musical, western, sport, short, film-noir.\nPlot: {}\nNote that a movie can belong to more than one genres, provide all the suitable genres seperated by commas. Answer: """

    def format_prompt(self, text=None):
        return self.prompt.format(text)
    