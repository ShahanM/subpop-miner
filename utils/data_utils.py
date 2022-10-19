

class DataContext():
	def __init__(self, **kwargs) -> None:
		self.kwargs = kwargs

		for k, v in kwargs.items():
			self.__setattr__(k, v)

	def add_data(self, data):
		for k, v in data.items():
			self.__setattr__(k, v)

	def add_key_value_pair(self, key, value):
		self.__setattr__(key, value)



class Config:
	"""Config class for storing user defined parameters"""
	def __init__(self, **kwargs) -> None:
		self.kwargs = kwargs

		for k, v in kwargs.items():
			self.__setattr__(k, v)
