from pydantic import BaseModel
from typing import List, Dict, Tuple, Union, Optional
import pandas as pd


class ContinuousVariable(BaseModel):
	name: str
	lbound: float
	ubound: float
	correlation: float
	freeze: bool = False

	def __gt__(self, other: 'ContinuousVariable') -> bool:
		return self.name == other.name and ((other.correlation > 0 \
			and self.ubound > other.ubound) or (other.correlation < 0 \
			and self.lbound < other.lbound))
	
	def __lt__(self, other: 'ContinuousVariable') -> bool:
		return self.name == other.name and ((other.correlation > 0 \
			and self.ubound < other.ubound) or (other.correlation < 0 \
			and self.lbound < other.lbound))

	def __ge__(self, other: 'ContinuousVariable') -> bool:
		return self.name == other.name and ((other.correlation > 0 \
			and self.ubound >= other.ubound) or (other.correlation < 0 \
			and self.lbound <= other.lbound))
	
	def __le__(self, other: 'ContinuousVariable') -> bool:
		return self.name == other.name and ((other.correlation > 0 \
			and self.ubound <= other.ubound) or (other.correlation < 0 \
			and self.lbound >= other.lbound))

	def __eq__(self, other: 'ContinuousVariable') -> bool:
		return self.name == other.name and self.lbound == other.lbound \
			and self.ubound == other.ubound

	def __hash__(self) -> int:
		return hash((self.name, self.lbound, self.ubound, self.correlation))


class DiscreteVariableDescriptor(BaseModel):
	name: str
	values: list


class DiscreteVariable(BaseModel):
	name: str
	value: Union[str, int]

	def __hash__(self) -> int:
		return hash((self.name, self.value))


class RuleMeta(BaseModel):
	continuous_vars: Dict[str, ContinuousVariable] = {}
	discrete_vars: Dict[str, DiscreteVariableDescriptor] = {}


class Rule():
	def __init__(self, itemset: List[str], \
		target: Optional[str]=None, \
		continuous_vars: Optional[Dict[str, ContinuousVariable]]=None,\
		discrete_vars: Optional[Dict[str, DiscreteVariable]]=None) -> None:
		"""Initializes a Rule object.

		Args:
			itemset (list) [Optional]: The itemset to build the rule from.
			target (str) [Optional]: The target variable.
			continuous_vars (dict) [Optional]: The continuous variables.
			discrete_vars (dict) [Optional]: The discrete variables.
		"""

		rule_dict: Dict[str, Union[str, int, float]] = {}
		# rule_str: str =  ''
		
		# FIXME itemset is not a set but a list - ideally it should be a set
		self.itemset = itemset
		# if len(self.itemset) > 0:
			# rule_dict, rule_str = self.build_rule_from_itemset()

		self.rule_dict = rule_dict
		# self.rule_str = rule_str

		self.target = target


		if continuous_vars:
			self.continuous_vars = continuous_vars
		else:
			self.continuous_vars = {}
		if discrete_vars:
			self.discrete_vars = discrete_vars
		else:
			self.discrete_vars = {}

		self.rule_str = self.__build_rule_string()
		
		# extra features for convenience but aren't always used
		self.numrows: int = 0
		self.support: float = 0.0
		self.q1: Union[int, float] = 0.0
		self.q3: Union[int, float] = 0.0
		self.target_threshold: Union[int, float] = 0.0

	def __repr__(self) -> str:
		return {
			'itemset': self.itemset,
			'target': self.target,
			'rule_dict': self.rule_dict,
			'rule_str': self.rule_str,
			'continuous_vars': self.continuous_vars,
			'discrete_vars': self.discrete_vars
		}.__repr__()

	def __hash__(self) -> int:
		return hash(self.rule_str)
	
	def __eq__(self, other) -> bool:
		return self.rule_str == other.rule_str

	def __ne__(self, other) -> bool:
		return not self.__eq__(other)

	def add_continuous_variable(self, contvar: ContinuousVariable) -> None:
		"""Adds a continuous variable to the rule.

		Args:
			contvar (ContinuousVariable): The continuous variable to add.
		"""
		self.continuous_vars[contvar.name] = contvar.copy()
		newitemset = self.itemset.copy()
		varupdt = 0
		for i, item in enumerate(self.itemset):
			if contvar.name in item:
				if '>=' in item:
					newitemset[i] = contvar.name + '>=' + str(contvar.lbound)
					varupdt += 1
				elif '<=' in item:
					newitemset[i] = contvar.name + '<=' + str(contvar.ubound)
					varupdt += 1
		else:
			if varupdt == 0:
				newitemset.append(contvar.name + '>=' + str(contvar.lbound))
				newitemset.append(contvar.name + '<=' + str(contvar.ubound))
		self.itemset = newitemset

		# rule_dict, rule_str = self.build_rule_from_itemset()
		# self.rule_dict = rule_dict
		self.rule_str = self.__build_rule_string()

	def remove_continuous_variable(self, contvar: ContinuousVariable) -> None:
		"""Removes a continuous variable from the rule.

		Args:
			contvar (ContinuousVariable): The continuous variable to remove.
		"""
		if contvar.name in self.continuous_vars:
			newitemset = self.itemset.copy()
			for i, item in enumerate(self.itemset):
				if contvar.name in item:
					# print('==>', contvar, item)
					newitemset.remove(item)
			self.itemset = newitemset
			del self.continuous_vars[contvar.name]

			# rule_dict, rule_str = self.build_rule_from_itemset()
			# self.rule_dict = rule_dict
			self.rule_str = self.__build_rule_string()

	def update_continuous_variable(self, contvar: ContinuousVariable) -> None:
		"""Updates a continuous variable in the rule.

		Args:
			contvar (ContinuousVariable): The continuous variable to update.
		"""
		self.continuous_vars[contvar.name] = contvar.copy()
		newitemset = self.itemset.copy()
		for i, item in enumerate(self.itemset):
			if contvar.name in item:
				if '>=' in item:
					newitemset[i] = contvar.name + '>=' + str(contvar.lbound)
				elif '<=' in item:
					newitemset[i] = contvar.name + '<=' + str(contvar.ubound)
		self.itemset = newitemset

		# rule_dict, rule_str = self.build_rule_from_itemset()
		# self.rule_dict = rule_dict
		self.rule_str = self.__build_rule_string()

	def add_discrete_variable(self, discvar: DiscreteVariable) -> None:
		"""Adds a discrete variable to the rule.

		Args:
			discvar (DiscreteVariable): The discrete variable to add.
		"""
		self.discrete_vars[discvar.name] = discvar.copy()
		newitemset = self.itemset.copy()

		dvarstr = discvar.name + '=' + str(discvar.value)
		if dvarstr not in newitemset:
			newitemset.append(dvarstr)

		self.itemset = newitemset

		# rule_dict, rule_str = self.build_rule_from_itemset()
		# self.rule_dict = rule_dict
		self.rule_str = self.__build_rule_string()

	def remove_discrete_variable(self, discvar: DiscreteVariable) -> None:
		"""Removes a discrete variable from the rule.

		Args:
			discvar (DiscreteVariable): The discrete variable to remove.
		"""
		if discvar.name in self.discrete_vars:
			del self.discrete_vars[discvar.name]
			newitemset = self.itemset.copy()
			dvarstr = discvar.name + '=' + str(discvar.value)
			if dvarstr in newitemset:
				newitemset.remove(dvarstr)
			self.itemset = newitemset

			# rule_dict, rule_str = self.build_rule_from_itemset()
			# self.rule_dict = rule_dict
			self.rule_str = self.__build_rule_string()

	def __build_rule_string(self) -> str:
		"""Builds the rule string from the itemset.

		Returns:
			str: The rule string.
		"""
		# dvaritems = []
		# for dvar in self.discrete_vars.values():
			# dvaritems.append(dvar.name + '==' + str(dvar.value))
		dvaritems = [dvar.name + '==' + str(dvar.value) \
			for dvar in self.discrete_vars.values()]
		# dvaritems.sort()

		cvaritems = []
		for cvar in self.continuous_vars.values():
			cvaritems.append(cvar.name + '>=' + str(cvar.lbound))
			cvaritems.append(cvar.name + '<=' + str(cvar.ubound))
		# cvaritems.sort()

		items = sorted(dvaritems) + sorted(cvaritems, reverse=True)

		return ' & '.join(items)

	# def build_rule_from_itemset(self, itemset: Optional[List[str]] = None) \
	# 	-> Tuple[dict, str]:
	# 	"""Builds a rule from an itemset.

	# 	Args:
	# 		itemset (list) [Optional]: The itemset to build the rule from.

	# 	Returns:
	# 		dict: The rule dictionary.
	# 		str: The rule string.
	# 	"""
	# 	if itemset and len(itemset) > 0:
	# 		self.itemset = itemset
	# 	elif self.itemset and len(self.itemset) > 0:
	# 		itemset = self.itemset
	# 	else:
	# 		return {}, ''

	# 	query_ = []
	# 	rule_dict = {}
	# 	assert isinstance(itemset, list)
	# 	for itm in iter(itemset):
	# 		if '>=' in itm:
	# 			quant_ = itm.split('>=')
	# 			quant_cat = quant_[0].strip()
	# 			quant_lbound = quant_[1].strip()
	# 			rule_dict[quant_cat] = {'lbound': float(quant_lbound)}
	# 			query_.append(itm)
	# 		elif '<=' in itm:
	# 			quant_ = itm.split('<=')
	# 			quant_cat = quant_[0].strip()
	# 			quant_ubound = quant_[1].strip()
	# 			rule_dict[quant_cat] = {'ubound': float(quant_ubound)}
	# 			query_.append(itm)
	# 		else:
	# 			splitsville = itm.split('=')
	# 			qual_cat = splitsville[0].strip()
	# 			qual_val = splitsville[1].strip()
	# 			rule_dict[qual_cat] = int(qual_val)
	# 			query_.append(qual_cat + '==' + qual_val)

	# 	rule_str = ' & '.join(sorted(query_, reverse=True))
	# 	self.rule_dict = rule_dict
		
	# 	self.rule_str = rule_str

	# 	return rule_dict, rule_str

	def evaluate(self, data: pd.DataFrame, \
		eval_params: Dict[str, Union[int, float]], \
		datalen: Optional[int] = None,
		target: Optional[str] = None, weights: Optional[pd.Series]=None) -> bool:
		"""Evaluates the rule against a data context.

		Args:
			eval_params ({str: Union[int, float]}): The data context to evaluate
			the rule against.
			Expected parameter keys:
				- 'minsup': The support threshold.
				- 'minq1': The lower quartile threshold.
				- 'minq3': The upper quartile threshold.
				- 'delta': The threshold for determining how much smaller the
				the subpopulation, represented by the rule, needs to be to be
				considered an outlier.
				- 'minthreshold': (minq3 - 3 * (minq3 - minq1)) - delta. 
				This parameter is used when provided instead of minq1, minq3, 
				and delta. At least this or 'minq3', 'minq1', and 'delta' must 
				be provided.
			
			datalen (int) [Optional]: Number of rows in the dataset. Providing
			this value will avoid having to compute it for each rule evaluation.
			
			target (str) [Optional]: The target variable. Must be provided if
			it was not provided when the rule was created.

		Returns:
			bool: True if the rule is satisfied, False otherwise.
		"""
		# 1. Evaluate rule
		# 2. Return True if rule is satisfies threshold, False otherwise

		if target and len(target) > 0:
			self.target = target
		elif self.target and len(self.target) > 0:
			target = self.target
		else:
			raise ValueError('No target specified.')
		
		if not datalen:
			datalen = len(data)
		assert target and len(target) > 0
		q1: Union[int, float] = 0.0
		q3: Union[int, float] = 0.0
		subpopfreq: int = 0
		if weights is not None:
			q1, q3, subpopfreq = self.get_subpop_iqr(data, target, weights)
		else:
			q1, q3, subpopfreq = self.get_subpop_iqr(data, target)
		self.q1 = q1
		self.q3 = q3
		self.numrows = subpopfreq
		support = subpopfreq/datalen
		self.support = support
		subpopiqr = q3 - q1
		subthreshold = q3 + 3*subpopiqr
		self.target_threshold = subthreshold

		if 'minthreshold' in eval_params:
			minthreshold = eval_params['minthreshold']
		else:
			popiqr = eval_params['minq3'] - eval_params['minq1']
			minthreshold = eval_params['minq3'] + 3*popiqr - eval_params['delta']

		if support >= eval_params['minsup']:
			if subthreshold <= minthreshold:
				return True
			else:
				return False
		else:
			return False

	def get_subpop_iqr(self, data: pd.DataFrame, target: str, \
		weights: Optional[pd.Series] = None,
		query: Optional[str] = None) -> Tuple[float, float, int]:
		"""Gets the interquartile range of the subpopulation represented by the
		rule.

		Args:
			data (pd.DataFrame): The data context to evaluate the rule against.
			target (str): The target variable.
			query (str) [Optional]: The query to evaluate.

		Returns:
			float: The lower quartile.
			float: The upper quartile.
			int: The number of rows in the subpopulation.
		"""
		# 1. Get subpopulation
		# 2. Get IQR
		# 3. Return IQR, subpopulation size
		if query is None:
			query = self.rule_str
		elif self.rule_str is None:
			raise ValueError('No query string found.')
		assert query is not None
		try:
			subpop = data.query(query)
		except Exception as e:
			print(e)
			print('Error evaluating query: {}'.format(query))
			return 0, 0, 0
		
		if weights is not None:
			return weighted_percentile(subpop[target], weights, 0.25), \
				weighted_percentile(subpop[target], weights, 0.75), len(subpop)
		return subpop[target].quantile(0.25), \
			subpop[target].quantile(0.75), len(subpop)

	def __copy__(self):
		rule = Rule(self.itemset, self.target)

		for disc in self.discrete_vars.values():
			rule.add_discrete_variable(disc)

		for cont in self.continuous_vars.values():
			rule.add_continuous_variable(cont)

		rule.numrows = self.numrows
		rule.support = self.support
		rule.q1 = self.q1
		rule.q3 = self.q3
		rule.target_threshold = self.target_threshold

		return rule
	
	def copy(self):
		return self.__copy__()

	def issubrule(self, other):
		if not isinstance(other, Rule):
			return False
		
		selfdset = set(self.discrete_vars.values())
		otherdset = set(other.discrete_vars.values())
		selfcset = set(self.continuous_vars.values())
		othercset = set(other.continuous_vars.values())

		if selfdset.issubset(otherdset) and len(selfcset.union(othercset)) == 0:
			return True


		if len(self.discrete_vars) > len(other.discrete_vars):
			return otherdset.issubset(selfdset)
		elif len(self.discrete_vars) == len(other.discrete_vars):
			if selfdset == otherdset:
				if selfcset == othercset:
					return True
				else:
					uniondiff = selfcset.union(othercset)\
						.difference(selfcset.intersection(othercset))
					if len(uniondiff) == 2:
						cone = uniondiff.pop()
						ctwo = uniondiff.pop()
						if cone.name == ctwo.name:
							return self.continuous_vars[cone.name] <= \
								self.continuous_vars[ctwo.name]
						else:
							return False
		# elif len(self.continuous_vars) == len(self.continuous_vars):
			# if selfcset == othercset:
				
			# if set(self.continuous_vars).difference(set(other.continuous_vars)) == 1:

		# else:
			# return False
		
		return False
