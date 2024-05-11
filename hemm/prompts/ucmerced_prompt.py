class UCMercedPrompt:
    def __init__(self):
        self.prompt = """Classify the given land image into the following classes: mediumresidential, buildings, tenniscourt, denseresidential, baseballdiamond, intersection, harbor, parkinglot, river, overpass, mobilehomepark, runway, forest, beach, freeway, airplane, storagetanks, chaparral, golfcourse, sparseresidential, agricultural.\nChoose a class from the above classes: """

    def format_prompt(self):
        return self.prompt