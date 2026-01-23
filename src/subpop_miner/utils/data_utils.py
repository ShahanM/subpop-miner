from itertools import chain, combinations

import pandas as pd
from tqdm import tqdm


class DataContext:
	def __init__(self, **kwargs) -> None:
		self.kwargs = kwargs

		for k, v in kwargs.items():
			self.__setattr__(k, v)

	def add_data(self, data):
		for k, v in data.items():
			self.__setattr__(k, v)

	def add_key_value_pair(self, key, value):
		self.__setattr__(key, value)

	def get_data(self):
		return self.__dict__

	def get_value(self, key):
		return self.__getattribute__(key)


class Config:
	"""Config class for storing user defined parameters"""

	def __init__(self, **kwargs) -> None:
		self.kwargs = kwargs

		for k, v in kwargs.items():
			self.__setattr__(k, v)


def load_data(filepath, relev_cat, chunksize=10000, totalrows=None, rowstoload=None):
	if totalrows:
		data = []
		if rowstoload:
			totalrows = min(rowstoload, totalrows)
			chunksize = min(chunksize, rowstoload)

		with tqdm(total=totalrows) as pbar:
			for chunk in tqdm(pd.read_csv(filepath, chunksize=chunksize, low_memory=False)):
				data.append(chunk)
				pbar.update(chunksize)

		all_data = pd.concat(data)
	else:
		all_data = pd.read_csv(filepath, nrows=rowstoload)

	all_data = all_data[relev_cat]

	return all_data


def powerset(iterable):
	"""
	Generate the powerset of a set
	"""
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
