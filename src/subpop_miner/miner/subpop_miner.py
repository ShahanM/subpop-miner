import numpy as np

from subpop_miner.miner.frequent_itemsets import FrequentItemsets
from subpop_miner.miner.rule import ContinuousVariable, DiscreteVariableDescriptor, RuleMeta


class SubpopMiner:
	config = {
		# Rule mining parameters
		'minsup': 0.01,
		'maxlen': 3,
		'delta': 0,  # Fitness threshold
		'delta2': float('inf'),  # Pruning threshold
		# Genetic algorithm parameters
		# Defaults taken from (Salleb-Aouissi et al., 2007)
		'pop_size': 250,
		'num_gen': 100,
		'crossover_prob': 0.5,
		'mutation_prob': 0.4,
	}

	def __init__(self, config, data, col_types, target):
		self.data = data
		self.col_types = col_types
		self.rule_meta = self._build_rulemeta(target)
		self.freq_itemsets = self._mine_frequent_itemsets()

	def _build_rulemeta(self, target):
		# FIXME: Use Pandas DataFrame's dtypes attribute to infer column types
		_discvars = {}
		_contvars = {}

		for col, coltype in self.col_types:
			if coltype == 'cat':
				_discvars[col] = DiscreteVariableDescriptor(name=col, values=sorted(self.data[col].unique()))
			elif coltype == 'cont':
				lbound = self.data[col].min()
				ubound = self.data[col].max()
				corref = np.corrcoef(self.data[col], self.data[target])[0, 1]
				_contvars[col] = ContinuousVariable(name=col, lbound=lbound, ubound=ubound, correlation=corref)
		_rulemeta = RuleMeta(discrete_vars=_discvars, continuous_vars=_contvars)

		return _rulemeta

	def _mine_frequent_itemsets(self):
		freq_itemsets = FrequentItemsets(min_support=self.config['minsup'], max_len=self.config['maxlen'])
		freq_itemsets.find_frequent_itemsets(self.data, list(self.rule_meta.discrete_vars.keys()))

		return freq_itemsets
