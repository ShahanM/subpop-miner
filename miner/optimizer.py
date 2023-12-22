import math
import random
from typing import Dict, List, Union

import pandas as pd
from pydantic import BaseModel

from .rule import ContinuousVariable, Rule, RuleMeta


class Individual(BaseModel):
	rule: Rule
	fitness: float

	class Config:
		arbitrary_types_allowed = True


class OptimizerBase(object):
	def __init__(self, **kwargs) -> None:
		self.kwargs = kwargs

		for k, v in kwargs.items():
			self.__setattr__(k, v)

	def optimize(self):
		raise NotImplementedError


class GeneticOptimizer(OptimizerBase):
	fitness_lookup = dict()

	def __init__(self, population_size: int, generations: int, \
		mutation_rate: float, crossover_rate: float, rulemeta: RuleMeta, \
		data: pd.DataFrame, **kwargs) -> None:
		super(GeneticOptimizer, self).__init__(**kwargs)

		self.population_size = population_size
		self.generations = generations
		self.mutation_rate = mutation_rate
		self.crossover_rate = crossover_rate
		self.rulemeta = rulemeta
		self.data = data
		self.datalen = len(data)

		self.aggressive_mutation = kwargs.get('aggressive_mutation', False)
		self.force_int_bounds = kwargs.get('force_int_bounds', False)

		self.min_interval = None

	def optimize(self, rule:Rule, eval_params: Dict[str, Union[int, float]]) \
		-> Rule:
		"""Optimizes the rule.

		Args:
			rule (Rule): The rule to optimize.

		Returns:
			Rule: The optimized rule.
		"""
		# print("------------------------------")
		# print(rule)
		if not isinstance(rule, Rule):
			raise ValueError('Rule must be of type Rule')
		
		# if len(rule.continuous_vars.keys()) > 0:
			# raise ValueError('Rule already contains continuous variables')
			# print('Rule already contains continuous variables')
			# return rule

		# 1. Generate initial population
		population = self.__generate_population(rule)
		# 2. Evaluate population
		population = self.__evaluate(population, eval_params)
		for i in range(self.generations):
			# 3. Select parents
			evo_candidates = self.__select(population)
			# 4. Crossover parents
			new_generation = self.__crossover(evo_candidates)
			new_generation = self.__evaluate(new_generation, eval_params)
			# 5. Mutate offspring
			new_generation = self.__mutate(new_generation)
			# 6. Evaluate offspring
			new_generation = self.__evaluate(new_generation, eval_params)
			# 7. Select survivors
			new_generation = self.__select(new_generation)
			# 8. Replace population with offspring
			population = self.__replace(population, new_generation)
			population = self.__evaluate(population, eval_params)
			# 9. Repeat steps 2-7 until termination criteria is met
			
		# 10. Return best individual
		return population[0].rule

	def __mutate(self, evo_candidates:List[Individual]) -> List[Individual]:
		"""Mutates the offspring.

		Args:
			evo_candidates (List[Individual]): The offspring to mutate.

		Returns:
			List[Individual]: The mutated offspring based on the mutation rate.
		"""
		# 1. Mutate offspring
		# 2. Return offspring
		mutated = []
		number_to_mutate = math.ceil(len(evo_candidates) * self.mutation_rate)
		for i in range(number_to_mutate):
			# randomly pick 1 individual to mutate
			individual = random.choice(evo_candidates)
			# mutate the individual
			individual = self.__mutate_individual(individual)
			
			evo_candidates.remove(individual)
			mutated.append(individual)
		evo_candidates.extend(mutated)

		return evo_candidates

	def __mutate_individual(self, individual: Individual) -> Individual:
		"""Mutates an individual.

		Args:
			individual (Individual): The individual to mutate.

		Returns:
			Individual: The mutated individual.
		"""
		# 1. Mutate individual
		# 2. Return individual
		cvars = []
		if self.aggressive_mutation:
			# mutate all continuous variables
			cvars = individual.rule.continuous_vars.values()
		else:
			# randomly pick 1 continuous variable to mutate
			cvars = [random.choice(list(individual.rule.continuous_vars.values()))]
		for cvar in cvars:
			value = self.__generate_random_bound(cvar.lbound, cvar.ubound, self.force_int_bounds)
			
			# mutate upper bound if variable is positively correlated
			if cvar.correlation > 0:
				ncvar = ContinuousVariable(name=cvar.name, lbound=cvar.lbound, \
					ubound=value, correlation=cvar.correlation)
				individual.rule.update_continuous_variable(ncvar)
			# mutate lower bound if variable is negatively correlated
			elif cvar.correlation < 0:
				ncvar = ContinuousVariable(name=cvar.name, lbound=value, \
					ubound=cvar.ubound, correlation=cvar.correlation)
				individual.rule.update_continuous_variable(cvar)

		return individual

	def __crossover(self, evo_candidates:List[Individual]) -> List[Individual]:
		"""Crossover the selected individuals.

		Args:
			evo_candidates (List[Individual]): The selected individuals.

		Returns:
			List[Individual]: The offspring of the selected individuals.
		"""
		# 1. Crossover selected individuals
		# 2. Return offspring

		offspring = []
		for i in range(0, len(evo_candidates)-2, 2):
			offspring.extend(self.__crossover_pair(evo_candidates[i], \
				evo_candidates[i+1]))

		return offspring

	def __crossover_pair(self, parent1:Individual, parent2:Individual) -> \
		List[Individual]:

		# 1. Crossover parents
		# 2. Return offspring

		# randomly pick 1 continuous variable to crossover
		# if len(parent1.rule.continuous_vars) > 1:
		# 	cvar = random.choice(list(parent1.rule.continuous_vars.values()))
		# else:
		# 	cvar = list(parent1.rule.continuous_vars.values())[0]

		# create offspring 1 with bounds from parent1
		offspring1 = parent1.rule.copy() # possible bug here

		# create offspring 2 with bounds from parent2
		offspring2 = parent2.rule.copy() # possible bug here

		# if cvar exists in both parents, do nothing (crossing bounds is just 
		# swapping upper bounds for positive correlation and lower bounds for
		# negative correlation)
		# So, we only need to check if cvar is missing in one of the parents
		# if cvar.name not in offspring2.continuous_vars: #FIXME: uncomment after experiment
			# if variable is positively correlated
			# if len(offspring1.continuous_vars) == 1: #FIXME: uncomment after experiment
				# if both parents have only 1 continous variable each
				# do nothing

				# if parent2 has more than 1 continuous variable
				# add one missing variable to offspring1
				# if len(offspring2.continuous_vars) > 1: #FIXME: uncomment after experiment
					# nvar = random.choice(list(offspring2.continuous_vars.values())) #FIXME: uncomment after experiment
					# offspring1.add_continuous_variable(nvar) #FIXME: uncomment after experiment
			
			# add missing variable to offspirng2
			# offspring2.add_continuous_variable(cvar) #FIXME: uncomment after experiment

			# if parent1 has more than 1 continuous variable
			# remove cvar from offspring1
			# if len(offspring1.continuous_vars) > 1 and not cvar.freeze: #FIXME: uncomment after experiment
				# offspring1.remove_continuous_variable(cvar) #FIXME: uncomment after experiment


		offspring1 = Individual(rule=offspring1, fitness=parent1.fitness)
		offspring2 = Individual(rule=offspring2, fitness=parent2.fitness)
	
		# create offspring 3 as a duplicate of parent1
		# and upper bound from parent1
		offspring3 = parent1.rule.copy()
		offspring3 = Individual(rule=offspring3, fitness=parent1.fitness)

		# create offspring 4 as a duplicate of parent2
		offspring4 = parent2.rule.copy()
		offspring4 = Individual(rule=offspring4, fitness=parent2.fitness)

		children = [offspring1, offspring2, offspring3, offspring4]

		return children
		

	def __select(self, population:List[Individual]) -> List[Individual]:
		"""Selects individuals for crossover.

		Args:
			population (list): The population to select from.

		Returns:
			list: The selected individuals.
		"""
		# 1. Select individuals based on fitness
		# 2. Return selected individuals

		# list is already sorted by fitness so just return
		# the first crossover_rate * population_size individuals
		return population[:int(self.crossover_rate * self.population_size)]

	def __evaluate(self, population:List[Individual], \
		eval_params: Dict[str, Union[int, float]]) -> List[Individual]:
		"""Evaluates the fitness of each individual in the population.

		Args:
			population (List[Individual]): The population to evaluate.

		Returns:
			List[Individual]: The evaluated population with fitness values 
								sorted in descending order.
		"""
		# 1. Evaluate each individual
		# 2. Return evaluated population
		# print("Current Fitness Lookup Table:", self.fitness_lookup)
		for individual in population:
			if individual.rule in self.fitness_lookup:
				# print('Found fitness in lookup cache.')
				# print(individual.rule)
				individual.fitness = self.fitness_lookup[individual.rule]
			else:
				individual.fitness = self.__get_fitness(individual.rule, eval_params)
				self.fitness_lookup[individual.rule] = individual.fitness
		
		population.sort(key=lambda x: x.fitness, reverse=True)

		return population

	def __get_fitness(self, rule:Rule, \
		eval_params: Dict[str, Union[int, float]]) -> float:
		"""Evaluates the fitness of an individual.

		Args:
			individual (Individual): The individual to evaluate.

		Returns:
			float: The fitness of the individual.
		"""
		# 1. Evaluate fitness of individual
		# 2. Return fitness

		# FIXME breakdown evaluate function into two parts
		# 1. evaluate on threshold
		# 2. evaluate on support
		if rule.evaluate(self.data, eval_params, self.datalen):
			fitness = 1.0
			for contvar in rule.continuous_vars.values():
				meta_range = \
					self.rulemeta.continuous_vars[contvar.name].ubound - \
					self.rulemeta.continuous_vars[contvar.name]\
					.lbound
				contvar_range = contvar.ubound - contvar.lbound
				try:
					fitness *= math.pow(contvar_range/meta_range, 2)
				except ZeroDivisionError as e:
					print(e)
					print(contvar, meta_range)
			return fitness

		return 0.0

	def __replace(self, population:List[Individual], \
		offspring:List[Individual]) -> List[Individual]:
		"""Replaces the least fit individuals in the population with the
		offspring.

		Args:
			population (List[Individual]): The population to replace from.
			offspring (List[Individual]): The offspring to replace with.

		Returns:
			List[Individual]: The new population.
		"""
		# 1. Replace least fit individuals in population with offspring
		# 2. Return new population

		# replace the last len(offspring) individuals in the population
		# with the offspring
		if len(population) == len(offspring):
			offspring.sort(key=lambda x: x.fitness, reverse=True)
			population[-len(offspring)/2:] = offspring[:len(offspring)/2]
		elif len(population) > len(offspring):
			population[-len(offspring):] = offspring
		else:
			offspring.sort(key=lambda x: x.fitness, reverse=True)
			population = offspring[:len(population)]

		return population

	def __generate_population(self, rule:Rule) -> List[Individual]:
		"""Generates a population of individuals.
		
		Returns:
			List[Individual]: The generated population.
		"""
		# 1. Generate individuals
		# 2. Return population

		# existing_cont = set(rule.continuous_vars.keys())
		# meta_cont_set = set(self.rulemeta.continuous_vars.keys())
		# missing_cont = meta_cont_set - existing_cont
		existing_cont = [cvar for cvar in rule.continuous_vars.values()\
			if not cvar.freeze]

		freezec_vars = [cvar.name for cvar in rule.continuous_vars.values()\
			if cvar.freeze]

		population = []
		contvars = list(self.rulemeta.continuous_vars.values())
		# contvars = [v for k, v in self.rulemeta.continuous_vars.items() \
		# 	if k in missing_cont]
		for i in range(self.population_size):
			newrule = Rule(rule.itemset[:], rule.target, discrete_vars=rule.discrete_vars.copy())
			if len(existing_cont) > 0:
				# FIXME this need to be refactored
				num2updt = random.randint(1, len(existing_cont))
				for cvar in random.sample(existing_cont, num2updt):
					contvar = cvar.copy()
					if cvar.correlation > 0:
						contvar.ubound = self.__generate_random_bound(cvar.lbound, cvar.ubound, self.force_int_bounds)
					if cvar.correlation < 0:
						contvar.lbound = self.__generate_random_bound(cvar.lbound, cvar.ubound, self.force_int_bounds)
					newrule.update_continuous_variable(contvar)
					new_individual = Individual(rule=newrule, fitness=0.0)
					population.append(new_individual)
				continue
				# FIXME end of refactoring

			numcontvars = random.randint(1, len(contvars))
			for contvar in random.sample(contvars, numcontvars):
				if contvar.name in freezec_vars:
					cont = contvar.copy()
				else:
					cont = ContinuousVariable(name=contvar.name, \
						lbound=contvar.lbound, ubound=contvar.ubound, \
						correlation=contvar.correlation)

				if contvar.correlation > 0:
					cont.ubound = self.__generate_random_bound(cont.lbound, cont.ubound, self.force_int_bounds)
				elif contvar.correlation < 0:
					cont.lbound = self.__generate_random_bound(cont.lbound, cont.ubound, self.force_int_bounds)
				
				if self.min_interval:
					if cont.ubound - cont.lbound > self.min_interval:
						newrule.add_continuous_variable(cont)
				else:
					newrule.add_continuous_variable(cont)
			new_individual = Individual(rule=newrule, fitness=0.0)
			population.append(new_individual)

		return population


	def __generate_random_bound(self, lower: Union[int, float], \
		upper: Union[int, float], force_int: bool = False) \
		-> Union[int, float]:

		if force_int:
			upper = math.ceil(upper)
			lower = math.floor(lower)
			return float(random.randint(lower, upper))
		else:
			return random.uniform(lower, upper)
