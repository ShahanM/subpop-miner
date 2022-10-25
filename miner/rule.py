from pydantic import BaseModel
from typing import List, Dict, Tuple, Union, Optional
import pandas as pd


class ContinuousVariable(BaseModel):
	name: str
	lbound: float
	ubound: float
	correlation: float

class DiscreteVariableDescriptor(BaseModel):
	name: str
	values: list

class DiscreteVariable(BaseModel):
	name: str
	value: Union[str, int]


class RuleMeta(BaseModel):
	continuous_vars: Dict[str, ContinuousVariable] = {}
	discrete_vars: Dict[str, DiscreteVariableDescriptor] = {}

# {'rule_dict': rule[0], 'rule_str': rule[1], 'support': supp, 'confidence': conf, 'threshold': subthreshold}
# {'rule_dict': rule[0], 'rule_str': rule[1], 'support': supp, 'confidence': conf, 'q1': q1, 'q3': q3, 'threshold': subthreshold, 'true_frequency': freq}
class Rule():
	def __init__(self, *args, **kwargs) -> None:
		self.kwargs = kwargs
		for k, v in kwargs.items():
			self.__setattr__(k, v)

		if 'rule_dict' in kwargs:
			self.rule_dict = kwargs['rule_dict']
		else:
			self.rule_dict = {}
		
		if 'query' in kwargs:
			self.rule_str = kwargs['query']
		else:
			self.rule_str = ''

		if 'itemset' in kwargs:
			self.itemset = kwargs['itemset']
			self.rule_dict, self.rule_str = self.build_rule_from_itemset()
		else:
			self.itemset = []

		if 'target' in kwargs:
			self.target = kwargs['target']
		else:
			self.target = None

		self.continuous_vars: Dict[str, ContinuousVariable] = {}
		self.discrete_vars: Dict[str, DiscreteVariable] = {} 

	def add_continuous_variable(self, contvar: ContinuousVariable) -> None:
		"""Adds a continuous variable to the rule.

		Args:
			contvar (ContinuousVariable): The continuous variable to add.
		"""
		self.continuous_vars.update({contvar.name: contvar})
		self.rule_dict[contvar.name] = {'lbound': contvar.lbound, \
			'ubound': contvar.ubound, 'correlation': contvar.correlation}


	def build_rule_from_itemset(self, itemset: Optional[List[str]] = None) \
		-> Tuple[dict, str]:
		"""Builds a rule from an itemset.

		Args:
			itmset (list) [Optional]: The itemset to build the rule from.

		Returns:
			dict: The rule dictionary.
			str: The rule string.
		"""
		if itemset and len(itemset) > 0:
			self.itemset = itemset
		elif self.itemset and len(self.itemset) > 0:
			itemset = self.itemset
		else:
			return {}, ''
		query_ = []
		rule_dict = {}
		assert isinstance(itemset, list)
		for itm in iter(itemset):
			if '>=' in itm:
				quant_ = itm.split('>=')
				quant_cat = quant_[0].strip()
				quant_lbound = quant_[1].strip()
				rule_dict[quant_cat] = {'lbound': float(quant_lbound)}
				query_.append(itm)
			elif '<=' in itm:
				quant_ = itm.split('<=')
				quant_cat = quant_[0].strip()
				quant_ubound = quant_[1].strip()
				rule_dict[quant_cat] = {'ubound': float(quant_ubound)}
				query_.append(itm)
			else:
				splitsville = itm.split('=')
				qual_cat = splitsville[0].strip()
				qual_val = splitsville[1].strip()
				rule_dict[qual_cat] = int(qual_val)
				query_.append(qual_cat + '==' + qual_val)

		rule_str = ' & '.join(query_)
		self.rule_dict = rule_dict
		self.rule_str = rule_str

		return rule_dict, rule_str

	def evaluate(self, data: pd.DataFrame, \
		eval_params: Dict[str, Union[int, float]], \
		datalen: Optional[int] = None,
		target: Optional[str] = None) -> bool:
		"""Evaluates the rule against a data context.

		Args:
			eval_params ({str: Union[int, float]}): The data context to evaluate
			the rule against.
			Expeted parameter keys:
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
		# TODO - implement rule evaluation
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
		# ruleset = set(map(str.strip, rule[1].split('&')))
		assert target and len(target) > 0
		q1, q3, subpopfreq = self.get_subpop_iqr(data, target)
		support = subpopfreq/datalen
		subpopiqr = q3 - q1
		subthreshold = q3 + 3*subpopiqr

		if 'minthreshold' in eval_params:
			minthreshold = eval_params['minthreshold']
		else:
			popiqr = eval_params['minq3'] - eval_params['minq1']
			minthreshold = eval_params['minq3'] + 3*popiqr - eval_params['delta']
		# conf = freq/subpopfreq
		# rulemeta = {'rule_dict': rule[0], 'rule_str': rule[1], 'support': supp, 'confidence': conf, \
			# 'q1': q1, 'q3': q3, 'threshold': subthreshold, 'true_frequency': freq}
		if support >= eval_params['minsup'] and subthreshold >= minthreshold:
			return True
			# passcandidates.append(rule[1])
			# if subthreshold <= minthreshold:
				# passlist.append(rulemeta)
				# if not skiprule(ruleset, finalcandidates):
					# finalcandidates.append(ruleset)
					# finallist.append(rulemeta)
				# else:
					# pass_prune_candidates.append(rulemeta)
			# else:
				# faillist.append(rulemeta)
				# if not skiprule(ruleset, failprunecandidates):
					# failcandidates.append(ruleset)
					# gencandidates.append(rulemeta)
				# else:
					# fail_prune_candidates.append(rulemeta)
		else:
			return False

	def get_subpop_iqr(self, data: pd.DataFrame, target: str, \
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
		# TODO - implement get_subpop_IQR
		# 1. Get subpopulation
		# 2. Get IQR
		# 3. Return IQR, subpopulation size
		if query is None:
			query = self.rule_str
		elif self.rule_str is None:
			raise ValueError('No query string found.')
		assert query is not None
		subpop = data.query(query)
		
		return subpop[target].quantile(0.25), \
			subpop[target].quantile(0.75), len(subpop)
