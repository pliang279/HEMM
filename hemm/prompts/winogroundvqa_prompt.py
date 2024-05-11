class WinogroundVQAprompt():
	def __init__(self):
		self.prompt="""Given this image along with a question about the image,
	please answer the question with only the word 'yes' or 'no'.
	Question: {}

	Answer:"""
	def format_prompt(self, text):
		return self.prompt.format(text)

