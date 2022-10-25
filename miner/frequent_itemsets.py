from apriori import generate_frequent_itemsets  # type: ignore
import pandas as pd


class FrequentItemsets():
	"""Finds frequent itemsets in a dataset.

	This class finds frequent itemsets in a dataset using the Apriori algorithm.

	Attributes:

	min_support: The minimum support for an itemset to be considered frequent.

	"""
	def __init__(self, min_support: float = 0.5, max_len: float = 3) -> None:
		self.min_support = min_support
		self.max_len = max_len
		self.itemsets: dict = {}
		self.id2item: dict = {}

	def find_frequent_itemsets(self, data: pd.DataFrame, \
		relevant_columns: list) -> list:
		"""Finds frequent itemsets in a dataset.

		Args:
			data (pd.DataFrame): The dataset to find frequent itemsets in.

		Returns:
			list: A list of frequent itemsets.

		"""
		datatable = self.__convert_dataframe_to_datatable(data, \
			relevant_columns)
		itemsets, id2item = generate_frequent_itemsets(datatable, \
			min_support=self.min_support, max_length=self.max_len)
		self.itemsets = itemsets
		self.id2item = id2item
		
		return itemsets


	def __convert_dataframe_to_datatable(self, data: pd.DataFrame, \
		relevant_columns: list) -> list:
		"""Converts a pandas dataframe to a datatable.

		Args:
			data (pd.DataFrame): The dataframe to convert.

		Returns:
			list: A datatable.
		"""
		datatable = []
		for row in data[relevant_columns].to_numpy():
			str_repr = ['='.join([npair[0], str(npair[1])]) for npair in zip(relevant_columns, row)]
			datatable.append(set(str_repr))

		return datatable
