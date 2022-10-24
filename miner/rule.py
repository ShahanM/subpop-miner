from pydantic import BaseModel
from typing import List, Dict
from utils.data_utils import DataContext


class ContinuousVariable(BaseModel):
	name: str
	lbound: float
	ubound: float
	correlation: float

class DiscreteVariable(BaseModel):
	name: str
	values: list

class RuleMeta(BaseModel):
	continuous_vars: Dict[str, ContinuousVariable]
	discrete_vars: Dict[str, DiscreteVariable]

# {'rule_dict': rule[0], 'rule_str': rule[1], 'support': supp, 'confidence': conf, 'threshold': subthreshold}
# {'rule_dict': rule[0], 'rule_str': rule[1], 'support': supp, 'confidence': conf, 'q1': q1, 'q3': q3, 'threshold': subthreshold, 'true_frequency': freq}
class Rule(RuleMeta):
	def __init__(self, **kwargs) -> None:
		super(Rule, self).__init__(**kwargs)
		self.kwargs = kwargs

		for k, v in kwargs.items():
			self.__setattr__(k, v)

		if 'rule_dict' in kwargs:
			self.rule_dict = self.kwargs['rule_dict']
		else:
			self.rule_dict = {}
		
		if 'itemset' in kwargs:
			self.itemset = self.kwargs['itemset']
			self.rule_dict, self.rule_str = self.build_rule_from_itemset(self.itemset)

		if 'df_query' in kwargs:
			self.rule_str = self.kwargs['query']

	def add_continuous_variable(self, contvar: ContinuousVariable) -> None:
		"""Adds a continuous variable to the rule.

		Args:
			contvar (ContinuousVariable): The continuous variable to add.
		"""
		self.continuous_vars.update({contvar.name: contvar})
		self.rule_dict[contvar.name] = {'lbound': contvar.lbound, \
			'ubound': contvar.ubound}


	def build_rule_from_itemset(self, itmset: list):
		query_ = []
		rule_dict = {}
		for itm in iter(itmset):
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

		return rule_dict, ' & '.join(query_)

	def evaluate(self, data_context: DataContext) -> bool:
		"""Evaluates the rule against a data context.

		Args:
			data_context (DataContext): The data context to evaluate the rule against.

		Returns:
			bool: True if the rule is satisfied, False otherwise.
		"""
		# TODO - implement rule evaluation
		# 1. Evaluate rule
		# 2. Return True if rule is satisfies threshold, False otherwise

		
		
		return True