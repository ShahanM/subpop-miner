import math
import random

import numpy as np
import pandas as pd
from pydantic import BaseModel
from subpop_miner.utils.memmap_manager import MemmapManager
from .rule import ContinuousVariable, Rule, RuleMeta


class Individual(BaseModel):
	rule: Rule
	fitness: float

	class Config:
		arbitrary_types_allowed = True


class OptimizerBase:
	def __init__(self, **kwargs) -> None:
		self.kwargs = kwargs

		for k, v in kwargs.items():
			self.__setattr__(k, v)

	def optimize(self):
		raise NotImplementedError


class GeneticOptimizer(OptimizerBase):
	fitness_lookup = {}

	def __init__(
		self,
		population_size: int,
		generations: int,
		mutation_rate: float,
		crossover_rate: float,
		rulemeta: RuleMeta,
		data_metadata: dict,
		**kwargs,
	) -> None:
		super()

		self.population_size = population_size
		self.generations = generations
		self.mutation_rate = mutation_rate
		self.crossover_rate = crossover_rate
		self.rulemeta = rulemeta

		self.data_manager = MemmapManager(metadata=data_metadata)
		self.datalen = data_metadata['len']

		self.aggressive_mutation = kwargs.get('aggressive_mutation', False)
		self.force_int_bounds = kwargs.get('force_int_bounds', False)

		self.min_interval = None

		# Optimization context placeholders
		self.np_target = None
		self.np_cont_vars = {}
		self.discrete_mask = None

	def hydrate(self):
		"""Calls the data manager to load memmaps."""
		self.data_manager = MemmapManager(metadata=self.data_manager.metadata)

	def optimize(self, rule: Rule, eval_params: dict[str, int | float]) -> Rule:
		"""Optimizes the rule.

		Args:
			rule (Rule): The rule to optimize.

		Returns:
			Rule: The optimized rule.
		"""
		if not isinstance(rule, Rule):
			raise ValueError('Rule must be of type Rule')

		# 0. Prepare Vectorized Context (Speedup Step)
		self._prepare_optimization_context(rule)
		try:
			# 1. Generate initial population
			population = self._generate_population(rule)
			# 2. Evaluate population
			population = self._evaluate(population, eval_params)
			for _ in range(self.generations):
				# 3. Select parents
				evo_candidates = self._select(population)
				# 4. Crossover parents
				new_generation = self._crossover(evo_candidates)
				new_generation = self._evaluate(new_generation, eval_params)
				# 5. Mutate offspring
				new_generation = self._mutate(new_generation)
				# 6. Evaluate offspring
				new_generation = self._evaluate(new_generation, eval_params)
				# 7. Select survivors
				new_generation = self._select(new_generation)
				# 8. Replace population with offspring
				population = self._replace(population, new_generation)
				population = self._evaluate(population, eval_params)
				# 9. Repeat steps 2-7 until termination criteria is met

			# 10. Return best individual
			return population[0].rule
		finally:
			self._clear_optimization_context()

	def _prepare_optimization_context(self, rule: Rule):
		"""Pre-loads data into NumPy arrays for fast vectorized evaluation."""
		if not rule.target:
			raise ValueError('Rule target must be set for optimization')

		self.np_target = self.data_manager.get_column(rule.target)

		# Load Continuous Variables (All potential ones)
		self.np_cont_vars = {}
		for name in self.rulemeta.continuous_vars:
			self.np_cont_vars[name] = self.data_manager.get_column(name)

		# re-compute Discrete Mask
		# The discrete variables do NOT change during the GA (only continuous bounds change)
		# So we calculate this mask ONCE.
		self.discrete_mask = np.ones(self.datalen, dtype=bool)
		for name, dvar in rule.discrete_vars.items():
			column_data = self.data_manager.get_column(name)
			comp_value = dvar.value

			# If the column is numeric (int/float) but the rule value is a string
			if isinstance(comp_value, str) and np.issubdtype(column_data.dtype, np.number):
				try:
					# Attempt to cast the string '0' to number 0.0 or 0
					float_val = float(comp_value)
					# If column is integer, cast to integer
					if np.issubdtype(column_data.dtype, np.integer):
						comp_value = int(float_val)
					else:
						comp_value = float_val
				except ValueError:
					# If casting fails, we might be comparing specific strings to numbers.
					# As a fallback, we could cast the column to string, but that's slow.
					# Usually, this exception implies bad data alignment.
					pass
			# Logic: Mask &= (Column == Value)
			# Note: Memmaps handling string are strictly typed.
			self.discrete_mask &= column_data == comp_value

	def _clear_optimization_context(self):
		self.np_target = None
		self.np_cont_vars = {}
		self.discrete_mask = None
		self.fitness_lookup = {}  # Clear cache between runs

	def _mutate(self, evo_candidates: list[Individual]) -> list[Individual]:
		"""Mutates the offspring efficiently."""
		number_to_mutate = math.ceil(len(evo_candidates) * self.mutation_rate)

		# Optimization: Use random.sample on INDICES to avoid slow list.remove()
		indices = random.sample(range(len(evo_candidates)), number_to_mutate)

		for i in indices:
			evo_candidates[i] = self._mutate_individual(evo_candidates[i])

		return evo_candidates

	def _mutate_individual(self, individual: Individual) -> Individual:
		cvars = []
		if self.aggressive_mutation:
			cvars = individual.rule.continuous_vars.values()
		else:
			if not individual.rule.continuous_vars:
				return individual
			cvars = [random.choice(list(individual.rule.continuous_vars.values()))]

		for cvar in cvars:
			# Safety check: ensure mutation doesn't cross bounds illogically
			val_range = (cvar.lbound, cvar.ubound)

			value = self._generate_random_bound(val_range[0], val_range[1], self.force_int_bounds)

			if cvar.correlation > 0:
				# Mutate Upper Bound
				# Ensure new ubound >= lbound
				if value < cvar.lbound:
					value = cvar.lbound
				ncvar = ContinuousVariable(
					name=cvar.name, lbound=cvar.lbound, ubound=value, correlation=cvar.correlation
				)
				individual.rule.update_continuous_variable(ncvar)
			elif cvar.correlation < 0:
				# Mutate Lower Bound
				# Ensure new lbound <= ubound
				if value > cvar.ubound:
					value = cvar.ubound
				ncvar = ContinuousVariable(
					name=cvar.name, lbound=value, ubound=cvar.ubound, correlation=cvar.correlation
				)
				individual.rule.update_continuous_variable(ncvar)

		return individual

	def _crossover(self, evo_candidates: list[Individual]) -> list[Individual]:
		"""Crossover the selected individuals.

		Args:
			evo_candidates (list[Individual]): The selected individuals.

		Returns:
			list[Individual]: The offspring of the selected individuals.
		"""
		offspring = []
		for i in range(0, len(evo_candidates) - 2, 2):
			offspring.extend(self._crossover_pair(evo_candidates[i], evo_candidates[i + 1]))

		return offspring

	def _get_fitness(self, rule: Rule, eval_params: dict) -> float:
		# Start with the pre-computed discrete mask
		current_mask = self.discrete_mask.copy()

		# Apply continuous vairable constraints using NumPy
		for cvar in rule.continuous_vars.values():
			col_data = self.np_cont_vars[cvar.name]
			# Mask &= (Data >= Lower) & (Data <= Upper)
			current_mask &= (col_data >= cvar.lbound) & (col_data <= cvar.ubound)

		subpop_target = self.np_target[current_mask]
		subpop_len = subpop_target.size

		rule.numrows = subpop_len
		rule.support = subpop_len / self.datalen if self.datalen > 0 else 0.0

		is_valid = False
		dist = 0.0

		if subpop_len > 0:
			# Use NumPy for percentiles (Handles NaNs automatically if np.nanpercentile used,
			# but assuming clean data for speed. Use np.nanpercentile if data has NaNs)
			q1 = np.percentile(subpop_target, 25)
			q3 = np.percentile(subpop_target, 75)

			rule.q1 = q1
			rule.q3 = q3

			subpopiqr = q3 - q1
			rule.target_threshold = q3 + 3 * subpopiqr

			# Determine Max Threshold
			if 'minthreshold' in eval_params:
				minthreshold = eval_params['minthreshold']
			else:
				popiqr = eval_params['minq3'] - eval_params['minq1']
				minthreshold = eval_params['minq3'] + 3 * popiqr - eval_params['delta1']

			# Check Validity
			if rule.support >= eval_params['minsup']:
				if rule.target_threshold <= minthreshold:
					is_valid = True
				else:
					dist = rule.target_threshold - minthreshold
		else:
			# Empty subpopulation
			rule.target_threshold = float('inf')
			dist = float('inf')

		range_score = self._compute_range_score(rule.continuous_vars.values())

		# Final fitness
		if is_valid:
			return 1.0 + range_score
		else:
			if rule.support == 0:
				return 0.0

		# Gradient penalty: Close to threshold = Higher score (approaching 1.0)
		return 1.0 / (1.0 + max(0, dist))

	def _compute_range_score(self, contvars: list[ContinuousVariable]):
		range_score = 0.0
		for cvar in contvars:
			meta_var = self.rulemeta.continuous_vars.get(cvar.name)
			if meta_var:
				meta_range = meta_var.ubound - meta_var.lbound
				if meta_range > 0:  # Guard against division by zero
					current_range = cvar.ubound - cvar.lbound
					# Square the ratio to favor wider ranges more aggressively
					range_score += (current_range / meta_range) ** 2
		return range_score

	def _crossover_pair(self, parent1: Individual, parent2: Individual) -> list[Individual]:
		# Create base copies
		child1_rule = parent1.rule.copy()
		child2_rule = parent2.rule.copy()

		# Identify Genes (Variables)
		p1_vars = parent1.rule.continuous_vars
		p2_vars = parent2.rule.continuous_vars
		all_var_names = set(p1_vars.keys()) | set(p2_vars.keys())

		for var_name in all_var_names:
			# Case 1: Variable exists in both parents (Swap Bounds)
			if var_name in p1_vars and var_name in p2_vars:
				if random.random() > 0.5:
					# Swap the bounds for this variable between children
					child1_rule.continuous_vars[var_name] = p2_vars[var_name].model_copy()
					child2_rule.continuous_vars[var_name] = p1_vars[var_name].model_copy()

			# Variable only in Parent 2 (Inject into Child 1)
			elif var_name in p2_vars and var_name not in p1_vars:
				if random.random() > 0.5:
					child1_rule.add_continuous_variable(p2_vars[var_name].copy())

			# Variable only in Parent 1 (Inject into Child 2)
			elif var_name in p1_vars and var_name not in p2_vars:
				if random.random() > 0.5:
					child2_rule.add_continuous_variable(p1_vars[var_name].copy())

		return [Individual(rule=child1_rule, fitness=0.0), Individual(rule=child2_rule, fitness=0.0)]

	def _select(self, population: list[Individual]) -> list[Individual]:
		"""Selects individuals for crossover.

		Args:
			population: The population to select from.

		Returns:
			list: The selected individuals.
		"""
		# list is already sorted
		return population[: int(self.crossover_rate * self.population_size)]

	def _evaluate(self, population: list[Individual], eval_params: dict[str, int | float]) -> list[Individual]:
		"""Evaluates the fitness of each individual in the population.

		Args:
			population: The population to evaluate.

		Returns:
			list[Individual]: The evaluated population with fitness values
								sorted in descending order.
		"""
		for individual in population:
			if individual.rule in self.fitness_lookup:
				individual.fitness = self.fitness_lookup[individual.rule]
			else:
				individual.fitness = self._get_fitness(individual.rule, eval_params)
				self.fitness_lookup[individual.rule] = individual.fitness

		population.sort(key=lambda x: x.fitness, reverse=True)
		return population

	def _replace(self, population: list[Individual], offspring: list[Individual]) -> list[Individual]:
		"""Replaces the least fit individuals in the population with the
		offspring.

		Args:
			population: The population to replace from.
			offspring: The offspring to replace with.

		Returns:
			list[Individual]: The new population.
		"""
		# replace the last len(offspring) individuals in the population
		# with the offspring
		cutoff = int(len(offspring) / 2)
		if len(population) == len(offspring):
			offspring.sort(key=lambda x: x.fitness, reverse=True)
			population[-cutoff:] = offspring[:cutoff]
		elif len(population) > len(offspring):
			population[-len(offspring) :] = offspring
		else:
			offspring.sort(key=lambda x: x.fitness, reverse=True)
			population = offspring[: len(population)]

		return population

	def _generate_population(self, rule: Rule) -> list[Individual]:
		"""Generates a population of individuals.

		Args:
			rule: A Rule that represents the subpopulation of interest.

		Returns:
			list[Individual]: The generated population.
		"""
		existing_cont = [cvar for cvar in rule.continuous_vars.values() if not cvar.freeze]
		freezec_vars = [cvar.name for cvar in rule.continuous_vars.values() if cvar.freeze]

		population = []
		contvars = list(self.rulemeta.continuous_vars.values())

		for _ in range(self.population_size):
			base_rule = Rule(rule.itemset[:], rule.target, discrete_vars=rule.discrete_vars.copy())
			newrule = base_rule
			if len(existing_cont) > 0:
				newrule = base_rule.copy()  # Branch off

				num2updt = random.randint(1, len(existing_cont))
				for cvar in random.sample(existing_cont, num2updt):
					contvar = cvar.model_copy()
					if cvar.correlation > 0:
						contvar.ubound = self._generate_random_bound(cvar.lbound, cvar.ubound, self.force_int_bounds)
					if cvar.correlation < 0:
						contvar.lbound = self._generate_random_bound(cvar.lbound, cvar.ubound, self.force_int_bounds)
					newrule.update_continuous_variable(contvar)

				new_individual = Individual(rule=newrule, fitness=0.0)
				population.append(new_individual)
				continue

			numcontvars = random.randint(1, len(contvars))
			for contvar in random.sample(contvars, numcontvars):
				if contvar.name in freezec_vars:
					cont = contvar.copy()
				else:
					cont = ContinuousVariable(
						name=contvar.name, lbound=contvar.lbound, ubound=contvar.ubound, correlation=contvar.correlation
					)

				if contvar.correlation > 0:
					cont.ubound = self._generate_random_bound(cont.lbound, cont.ubound, self.force_int_bounds)
				elif contvar.correlation < 0:
					cont.lbound = self._generate_random_bound(cont.lbound, cont.ubound, self.force_int_bounds)

				if self.min_interval:
					if cont.ubound - cont.lbound > self.min_interval:
						newrule.add_continuous_variable(cont)
				else:
					newrule.add_continuous_variable(cont)

			new_individual = Individual(rule=newrule, fitness=0.0)
			population.append(new_individual)

		return population

	def _generate_random_bound(self, lower, upper, force_int=False):
		if force_int:
			lower_int = math.floor(lower)
			upper_int = math.ceil(upper)
			if lower_int > upper_int:
				lower_int, upper_int = upper_int, lower_int
			return float(random.randint(lower_int, upper_int))
		return random.uniform(lower, upper)
