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
		rule_str: str =  ''
		# itemset: List[str]
		# target: Optional[str]
		# continuous_vars: Dict[str, ContinuousVariable]
		# discrete_vars: Dict[str, DiscreteVariable]
		
		# rule_dict = {}
		# rule_str = ''

		# super().__init__(itemset=itemset, target=target, rule_dict=rule_dict, 
		# 	rule_str=rule_str, continuous_vars=continuous_vars, \
		# 	discrete_vars=discrete_vars)
		
		self.itemset = itemset
		if len(self.itemset) > 0:
			rule_dict, rule_str = self.build_rule_from_itemset()
		# object.__setattr__(self, 'itemset', itemset)
		# object.__setattr__(self, 'rule_dict', rule_dict)
		self.rule_dict = rule_dict
		self.rule_str = rule_str
		# object.__setattr__(self, 'rule_str', rule_str)

		self.target = target
		# object.__setattr__(self, 'target', target)

		if continuous_vars:
			self.continuous_vars = continuous_vars
		else:
			self.continuous_vars = {}
		# object.__setattr__(self, 'continuous_vars', continuous_vars)
		if discrete_vars:
			self.discrete_vars = discrete_vars
		else:
			self.discrete_vars = {}
		# object.__setattr__(self, 'discrete_vars', discrete_vars)

	def __repr__(self) -> str:
		return {
			'itemset': self.itemset,
			'target': self.target,
			'rule_dict': self.rule_dict,
			'rule_str': self.rule_str,
			'continuous_vars': self.continuous_vars,
			'discrete_vars': self.discrete_vars
		}.__repr__()

	def add_continuous_variable(self, contvar: ContinuousVariable) -> None:
		"""Adds a continuous variable to the rule.

		Args:
			contvar (ContinuousVariable): The continuous variable to add.
		"""
		# self.continuous_vars.update({contvar.name: contvar})
		# print(self.continuous_vars)
		self.continuous_vars[contvar.name] = contvar
		newitemset = self.itemset.copy()
		# if len(self.continuous_vars) > 0:
		# 	self.itemset.append(contvar.name + '>=' + str(contvar.lbound))
		# 	self.itemset.append(contvar.name + '<=' + str(contvar.ubound))
		
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
				
		# object.__setattr__(self, 'itemset', newitemset)
		self.itemset = newitemset

		rule_dict, rule_str = self.build_rule_from_itemset()
		self.rule_dict = rule_dict
		self.rule_str = rule_str
		# object.__setattr__(self, 'rule_dict', rule_dict)
		# object.__setattr__(self, 'rule_str', rule_str)

		# print(self.rule_str)
		# cvar_dict = {'lbound': contvar.lbound, \
			# 'ubound': contvar.ubound}

		# object.__setattr__(self, 'rule_dict', {**self.rule_dict, **{contvar.name: cvar_dict}})
		# self.rule_dict[contvar.name]

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
		# object.__setattr__(self, 'rule_dict', rule_dict)
		self.rule_str = rule_str
		# object.__setattr__(self, 'rule_str', rule_str)

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

	def __copy__(self):
		rule = Rule(self.itemset, self.target)
		rule.discrete_vars = self.discrete_vars
		for cont in self.continuous_vars.values():
			rule.add_continuous_variable(cont)

		return rule
	
	def copy(self):
		return self.__copy__()
