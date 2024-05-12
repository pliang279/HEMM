class DecimerPrompt:
	def __init__(self):
		self.prompt = """Simplified molecular-input line-entry system (SMILES) notation of the given molecule: """

	def format_prompt(self, text=None):
		return self.prompt
	