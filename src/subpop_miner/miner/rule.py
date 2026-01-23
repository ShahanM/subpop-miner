import numpy as np
import pandas as pd
from pydantic import BaseModel


def weighted_percentile(data: pd.DataFrame, weights: pd.Series, quantile):
	np_vector = data.to_numpy()
	ix = np.argsort(np_vector)
	data = np_vector[ix]
	weights = weights[ix]
	cdf = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
	return np.interp(quantile, cdf, np_vector)


class ContinuousVariable(BaseModel):
	name: str
	lbound: float
	ubound: float
	correlation: float
	freeze: bool = False

	def __gt__(self, other: 'ContinuousVariable') -> bool:
		return self.name == other.name and (
			(other.correlation > 0 and self.ubound > other.ubound)
			or (other.correlation < 0 and self.lbound < other.lbound)
		)

	def __lt__(self, other: 'ContinuousVariable') -> bool:
		return self.name == other.name and (
			(other.correlation > 0 and self.ubound < other.ubound)
			or (other.correlation < 0 and self.lbound < other.lbound)
		)

	def __ge__(self, other: 'ContinuousVariable') -> bool:
		return self.name == other.name and (
			(other.correlation > 0 and self.ubound >= other.ubound)
			or (other.correlation < 0 and self.lbound <= other.lbound)
		)

	def __le__(self, other: 'ContinuousVariable') -> bool:
		return self.name == other.name and (
			(other.correlation > 0 and self.ubound <= other.ubound)
			or (other.correlation < 0 and self.lbound >= other.lbound)
		)

	def __eq__(self, other: 'ContinuousVariable') -> bool:
		return self.name == other.name and self.lbound == other.lbound and self.ubound == other.ubound

	def __hash__(self) -> int:
		return hash((self.name, self.lbound, self.ubound, self.correlation))


class DiscreteVariableDescriptor(BaseModel):
	name: str
	values: list


class DiscreteVariable(BaseModel):
	name: str
	value: str | int

	def __hash__(self) -> int:
		return hash((self.name, self.value))


class RuleMeta(BaseModel):
	continuous_vars: dict[str, ContinuousVariable] = {}
	discrete_vars: dict[str, DiscreteVariableDescriptor] = {}


class Rule:
	def __init__(
		self,
		itemset: list[str],
		target: str | None = None,
		continuous_vars: dict[str, ContinuousVariable] | None = None,
		discrete_vars: dict[str, DiscreteVariable] | None = None,
	) -> None:
		"""Initializes a Rule object.

		Args:
			itemset: The itemset to build the rule from.
			target: The target variable.
			continuous_vars: The continuous variables.
			discrete_vars: The discrete variables.
		"""

		rule_dict: dict[str, str | int | float] = {}
		self.itemset = itemset
		self.rule_dict = rule_dict
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
		self.q1: int | float = 0.0
		self.q3: int | float = 0.0
		self.target_threshold: int | float = 0.0

	def __repr__(self) -> str:
		return {
			'itemset': self.itemset,
			'target': self.target,
			'rule_dict': self.rule_dict,
			'rule_str': self.rule_str,
			'continuous_vars': self.continuous_vars,
			'discrete_vars': self.discrete_vars,
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
			contvar: The continuous variable to add.
		"""
		self.continuous_vars[contvar.name] = contvar.model_copy()
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

		self.rule_str = self.__build_rule_string()

	def remove_continuous_variable(self, contvar: ContinuousVariable) -> None:
		"""Removes a continuous variable from the rule.

		Args:
			contvar: The continuous variable to remove.
		"""
		if contvar.name in self.continuous_vars:
			newitemset = self.itemset.copy()
			for item in self.itemset:
				if contvar.name in item:
					newitemset.remove(item)
			self.itemset = newitemset
			del self.continuous_vars[contvar.name]

			self.rule_str = self.__build_rule_string()

	def update_continuous_variable(self, contvar: ContinuousVariable) -> None:
		"""Updates a continuous variable in the rule.

		Args:
			contvar: The continuous variable to update.
		"""
		self.continuous_vars[contvar.name] = contvar.model_copy()
		newitemset = self.itemset.copy()
		for i, item in enumerate(self.itemset):
			if contvar.name in item:
				if '>=' in item:
					newitemset[i] = contvar.name + '>=' + str(contvar.lbound)
				elif '<=' in item:
					newitemset[i] = contvar.name + '<=' + str(contvar.ubound)
		self.itemset = newitemset

		self.rule_str = self.__build_rule_string()

	def add_discrete_variable(self, discvar: DiscreteVariable) -> None:
		"""Adds a discrete variable to the rule.

		Args:
			discvar: The discrete variable to add.
		"""
		self.discrete_vars[discvar.name] = discvar.model_copy()
		newitemset = self.itemset.copy()

		dvarstr = discvar.name + '=' + str(discvar.value)
		if dvarstr not in newitemset:
			newitemset.append(dvarstr)

		self.itemset = newitemset

		self.rule_str = self.__build_rule_string()

	def remove_discrete_variable(self, discvar: DiscreteVariable) -> None:
		"""Removes a discrete variable from the rule.

		Args:
			discvar (DiscreteVariable): The discrete variable to remove.
		"""
		if discvar.name in self.discrete_vars:
			del self.discrete_vars[discvar.name]
			newitemset = self.itemset.model_copy()
			dvarstr = discvar.name + '=' + str(discvar.value)
			if dvarstr in newitemset:
				newitemset.remove(dvarstr)
			self.itemset = newitemset

			self.rule_str = self.__build_rule_string()

	def __build_rule_string(self) -> str:
		"""Builds the rule string from the itemset.

		Returns:
			str: The rule string.
		"""
		dvaritems = [dvar.name + '==' + str(dvar.value) for dvar in self.discrete_vars.values()]

		cvaritems = []
		for cvar in self.continuous_vars.values():
			cvaritems.append(cvar.name + '>=' + str(cvar.lbound))
			cvaritems.append(cvar.name + '<=' + str(cvar.ubound))

		items = sorted(dvaritems) + sorted(cvaritems, reverse=True)

		return ' & '.join(items)

	def evaluate(
		self,
		data: pd.DataFrame,
		eval_params: dict[str, int | float],
		datalen: int | None = None,
		target: str | None = None,
		weights: pd.Series | None = None,
	) -> bool:
		"""Evaluates the rule against a data context.

		Args:
			eval_params: The data context to evaluate
			the rule against.
			Expected parameter keys:
				- 'minsup': The support threshold.
				- 'minq1': The lower quartile threshold.
				- 'minq3': The upper quartile threshold.
				- 'delta1': The threshold for determining how much smaller the
				the subpopulation, represented by the rule, needs to be to be
				considered an outlier.
				- 'minthreshold': (minq3 - 3 * (minq3 - minq1)) - delta1.
				This parameter is used when provided instead of minq1, minq3,
				and delta1. At least this or 'minq3', 'minq1', and 'delta1' must
				be provided.

			datalen: Number of rows in the dataset. Providing
			this value will avoid having to compute it for each rule evaluation.

			target: The target variable. Must be provided if
			it was not provided when the rule was created.

		Returns:
			bool: True if the rule is satisfied, False otherwise.
		"""
		if target and len(target) > 0:
			self.target = target
		elif self.target and len(self.target) > 0:
			target = self.target
		else:
			raise ValueError('No target specified.')

		if not datalen:
			datalen = len(data)
		assert target and len(target) > 0
		q1: int | float = 0.0
		q3: int | float = 0.0
		subpopfreq: int = 0
		if weights is not None:
			q1, q3, subpopfreq = self.get_subpop_iqr(data, target, weights)
		else:
			q1, q3, subpopfreq = self.get_subpop_iqr(data, target)
		self.q1 = q1
		self.q3 = q3
		self.numrows = subpopfreq
		support = subpopfreq / datalen
		self.support = support
		subpopiqr = q3 - q1
		subthreshold = q3 + 3 * subpopiqr
		self.target_threshold = subthreshold

		if 'minthreshold' in eval_params:
			minthreshold = eval_params['minthreshold']
		else:
			popiqr = eval_params['minq3'] - eval_params['minq1']
			minthreshold = eval_params['minq3'] + 3 * popiqr - eval_params['delta1']

		if support >= eval_params['minsup']:
			if subthreshold <= minthreshold:
				return True
			else:
				return False
		else:
			return False

	def get_subpop_iqr(
		self, data: pd.DataFrame, target: str, weights: pd.Series | None = None, query: str | None = None
	) -> tuple[float, float, int]:
		"""Gets the interquartile range of the subpopulation represented by the
		rule.

		Args:
			data: The data context to evaluate the rule against.
			target: The target variable.
			query: The query to evaluate.

		Returns:
			float: The lower quartile.
			float: The upper quartile.
			int: The number of rows in the subpopulation.
		"""
		if query is None:
			query = self.rule_str
		elif self.rule_str is None:
			raise ValueError('No query string found.')
		assert query is not None
		try:
			subpop = data.query(query)
		except Exception as e:
			print(e)
			print(f'Error evaluating query: {query}')
			return 0, 0, 0

		if weights is not None:
			return (
				weighted_percentile(subpop[target], weights, 0.25),
				weighted_percentile(subpop[target], weights, 0.75),
				len(subpop),
			)
		return subpop[target].quantile(0.25), subpop[target].quantile(0.75), len(subpop)

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

		# Discrete variables must be a subset
		if not selfdset.issubset(otherdset):
			return False

		# Continuous variables must be a subset (or logically covered)
		# If sets are identical, we are good
		if selfcset == othercset:
			return True

		# Check if every continuous variable in self exists in other and is 'wider/equal'
		for cvar_name, cvar_self in self.continuous_vars.items():
			if cvar_name not in other.continuous_vars:
				# If self has a constraint that other doesn't, self is NOT a subrule
				return False

			cvar_other = other.continuous_vars[cvar_name]
			# Check if self is 'wider' (more general) than other
			if not (cvar_self.lbound <= cvar_other.lbound and cvar_self.ubound >= cvar_other.ubound):
				return False

		return True
