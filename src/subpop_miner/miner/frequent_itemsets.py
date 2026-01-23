import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder


class FrequentItemsets:
	"""Finds frequent itemsets in a dataset.

	This class finds frequent itemsets in a dataset using the FP-Growth algorithm
	(via mlxtend), which is a faster alternative to Apriori.

	Attributes:
		min_support: The minimum support for an itemset to be considered frequent.
		max_len: The maximum length of itemsets to discover.
	"""

	def __init__(self, min_support: float = 0.5, max_len: int = 3) -> None:
		self.min_support = min_support
		self.max_len = max_len
		self.itemsets: dict = {}
		self.id2item: dict = {}

	def find_frequent_itemsets(self, data: pd.DataFrame, relevant_columns: list) -> dict:
		"""Finds frequent itemsets in a dataset.

		Args:
			data (pd.DataFrame): The dataset to find frequent itemsets in.
			relevant_columns (list): List of columns to include in the analysis.

		Returns:
			dict: A dictionary of frequent itemsets grouped by length.
				Structure: {length: {frozenset(ids): count}}
		"""
		# Convert DataFrame to List of Sets (Legacy Format)
		# We reuse your existing logic here as it prepares the strings nicely (e.g., "Col=Val")
		dataset = self.__convert_dataframe_to_datatable(data, relevant_columns)

		# Encode Transactions for mlxtend (One-Hot Encoding)
		te = TransactionEncoder()
		te_ary = te.fit(dataset).transform(dataset)
		df_bool = pd.DataFrame(te_ary, columns=te.columns_)

		# Run FP-Growth (Faster than Apriori)
		# Note: mlxtend returns support as a float fraction (0.0 - 1.0)
		frequent_df = fpgrowth(df_bool, min_support=self.min_support, use_colnames=True, max_len=self.max_len)

		# Reconstruct Legacy Data Structures (itemsets dict and id2item)
		# The old library likely mapped string items to integer IDs. We recreate that here.

		# Create mappings: ID <-> Item String
		# We sort columns to ensure deterministic ID assignment
		all_items = sorted(te.columns_)
		self.id2item = {i: item for i, item in enumerate(all_items)}
		item2id = {item: i for i, item in enumerate(all_items)}

		# Initialize the itemsets dictionary structure: {length: {frozenset(ids): count}}
		self.itemsets = {}
		total_transactions = len(data)

		for row in frequent_df.itertuples(index=False):
			# row.itemsets is a frozenset of strings
			# row.support is the support fraction

			itemset_strings = row.itemsets
			k = len(itemset_strings)

			# Convert strings to IDs
			ids = frozenset(item2id[s] for s in itemset_strings)

			# Convert support fraction back to absolute count (to match old library behavior)
			count = int(round(row.support * total_transactions))

			if k not in self.itemsets:
				self.itemsets[k] = {}

			self.itemsets[k][ids] = count

		return self.itemsets

	def __convert_dataframe_to_datatable(self, data: pd.DataFrame, relevant_columns: list) -> list:
		"""Converts a pandas dataframe to a list of sets of strings."""
		datatable = []
		for row in data[relevant_columns].to_numpy():
			transaction = []
			for col, val in zip(relevant_columns, row):
				# Skip NaNs (Treat as missing info, don't create 'Col=nan' item)
				if pd.isna(val):
					continue

				# Normalize Floats (Treat 1.0 as 1)
				# This prevents splitting support between "1" and "1.0"
				if isinstance(val, float) and val.is_integer():
					val = int(val)

				transaction.append('='.join([str(col), str(val)]))
			datatable.append(transaction)

		return datatable
