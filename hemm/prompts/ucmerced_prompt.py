class UCMercedPrompt:
    def __init__(self):
        self.prompt = """Image is given to you. Classify if the image belongs to one of the following classes: 
         mediumresidential, buildings, tenniscourt, denseresidential, baseballdiamond, intersection, harbor, parkinglot, river, overpass, mobilehomepark, runway, forest, beach, freeway, 
         airplane, storagetanks, chaparral, golfcourse, sparseresidential, agricultural. """

    def format_prompt(self):
        return self.prompt