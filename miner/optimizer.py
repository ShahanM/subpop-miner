from rule import Rule, RuleMeta, ContinuousVariable, DiscreteVariable
import random
import numpy as np
from pydantic import BaseModel
from typing import List
import math

from utils.data_utils import DataContext


class Individual(BaseModel):
	rule: Rule
	fitness: float


class OptimizerBase(object):
	def __init__(self, **kwargs) -> None:
		self.kwargs = kwargs

		for k, v in kwargs.items():
			self.__setattr__(k, v)

	def optimize(self):
		raise NotImplementedError


class GeneticOptimizer(OptimizerBase):
	def __init__(self, population_size: int, generations: int, \
		mutation_rate: float, crossover_rate: float, rulemeta: RuleMeta, \
		data_context: DataContext, **kwargs) -> None:
		super(GeneticOptimizer, self).__init__(**kwargs)

		self.population_size = population_size
		self.generations = generations
		self.mutation_rate = mutation_rate
		self.crossover_rate = crossover_rate
		self.rulemeta = rulemeta
		self.data_context = data_context

		self.min_interval = None

	def optimize(self, rule:Rule) -> Rule:
		"""Optimizes the rule.

		Args:
			rule (Rule): The rule to optimize.

		Returns:
			Rule: The optimized rule.
		"""

		if not isinstance(rule, Rule):
			raise ValueError('Rule must be of type Rule')
		
		if len(rule.continuous_vars) > 0:
			raise ValueError('Rule already contains continuous variables')

		# TODO - implement genetic algorithm
		# 1. Generate initial population
		population = self.__generate_population(rule)
		for i in range(self.generations):
			# 2. Evaluate population
			population = self.__evaluate(population)
			# 3. Select parents
			evo_candidates = self.__select(population)
			# 4. Crossover parents
			new_generation = self.__crossover(evo_candidates)
			# 5. Mutate offspring
			new_generation = self.__mutate(new_generation)
			# 6. Evaluate offspring
			new_generation = self.__evaluate(new_generation)
			# 7. Replace population with offspring
			population = self.__replace(population, new_generation)
			# 8. Repeat steps 2-7 until termination criteria is met

		# 9. Return best individual
		return population[0].rule

	
	def __mutate(self, evo_candidates:List[Individual]) -> List[Individual]:
		"""Mutates the offspring.

		Args:
			evo_candidates (List[Individual]): The offspring to mutate.

		Returns:
			List[Individual]: The mutated offspring based on the mutation rate.
		"""
		# TODO - implement mutation
		# 1. Mutate offspring
		# 2. Return offspring
		mutated = []
		number_to_mutate = math.ceil(len(evo_candidates) * self.mutation_rate)
		for i in range(number_to_mutate):
			# randomly pick 1 individual to mutate
			individual = random.choice(evo_candidates)
			# randomly pick 1 continuous variable to mutate
			cvar = random.choice(individual.rule.continuous_vars.keys.values())
			value = random.uniform(cvar.lbound, cvar.ubound)
			# mutate upper bound if variable is positively correlated
			if cvar.correlation > 0:
				individual.rule.continuous_vars[cvar.name].ubound = value
			# mutate lower bound if variable is negatively correlated
			elif cvar.correlation < 0:
				individual.rule.continuous_vars[cvar.name].lbound = value
			evo_candidates.remove(individual)
			mutated.append(individual)
		evo_candidates.extend(mutated)

		return evo_candidates

	def __crossover(self, evo_candidates:List[Individual]) -> List[Individual]:
		"""Crossover the selected individuals.

		Args:
			evo_candidates (List[Individual]): The selected individuals.

		Returns:
			List[Individual]: The offspring of the selected individuals.
		"""
		# TODO - implement crossover
		# 1. Crossover selected individuals
		# 2. Return offspring

		offspring = []
		for i in range(0, len(evo_candidates), 2):
			offspring.extend(self.__crossover_pair(evo_candidates[i], \
				evo_candidates[i+1]))

		return offspring

	def __crossover_pair(self, parent1:Individual, parent2:Individual) -> \
		List[Individual]:

		# TODO - implement crossover
		# 1. Crossover parents
		# 2. Return offspring

		# randomly pick 1 continuous variable to crossover
		cvar = random.choice(parent1.rule.continuous_vars.keys.values())

		# create offspring 1 with bounds from parent1
		offspring1 = parent1.rule.copy() # possible bug here

		# create offspring 2 with bounds from parent2
		offspring2 = parent2.rule.copy() # possible bug here

		# pick bounds from parent1
		# if variable is positively correlated
		if cvar.correlation > 0:
			offspring1.continuous_vars[cvar.name].ubound = \
				parent2.rule.continuous_vars[cvar.name].ubound
			offspring2.continuous_vars[cvar.name].ubound = \
				parent1.rule.continuous_vars[cvar.name].ubound

		offspring = []

		# create offspring 3 as a duplicate of parent1
		# and upper bound from parent1
		offspring3 = parent1.rule.copy()

		# create offspring 4 as a duplicate of parent2
		offspring4 = parent2.rule.copy()

		offspring.extend([offspring1, offspring2, offspring3, offspring4])

		return offspring
		

	def __select(self, population:List[Individual]) -> list:
		"""Selects individuals for crossover.

		Args:
			population (list): The population to select from.

		Returns:
			list: The selected individuals.
		"""
		# TODO - implement selection
		# 1. Select individuals based on fitness
		# 2. Return selected individuals

		# list is already sorted by fitness so just return
		# the first crossover_rate * population_size individuals

		return population[:int(self.crossover_rate * self.population_size)]

	def __evaluate(self, population:List[Individual]) -> List[Individual]:
		"""Evaluates the fitness of each individual in the population.

		Args:
			population (List[Individual]): The population to evaluate.

		Returns:
			List[Individual]: The evaluated population with fitness values 
								sorted in descending order.
		"""
		# TODO - implement evaluation
		# 1. Evaluate each individual
		# 2. Return evaluated population

		for individual in population:
			individual.fitness = self.__get_fitness(individual)
		
		population.sort(key=lambda x: x.fitness, reverse=True)

		return population

	def __get_fitness(self, individual:Individual) -> float:
		"""Evaluates the fitness of an individual.

		Args:
			individual (Individual): The individual to evaluate.

		Returns:
			float: The fitness of the individual.
		"""
		# TODO - implement fitness evaluation
		# 1. Evaluate fitness of individual
		# 2. Return fitness

		# FIXME breakdown evaluate function into two parts
		# 1. evaluate on threshold
		# 2. evaluate on support
		# if support < min_support: fitness -= len(rule data)
		if individual.rule.evaluate(self.data_context):
			fitness = 1.0
			for contvar in individual.rule.continuous_vars.values():
				meta_range = \
					self.rulemeta.continuous_vars[contvar.name].ubound - \
					self.rulemeta.continuous_vars[contvar.name]\
					.lbound
				contvar_range = contvar.ubound - contvar.lbound
				fitness *= math.pow(contvar_range/meta_range, 2)
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
		# TODO - implement replacement
		# 1. Replace least fit individuals in population with offspring
		# 2. Return new population

		# replace the last len(offspring) individuals in the population
		# with the offspring
		population[-len(offspring):] = offspring

		return population

	def __generate_population(self, rule:Rule) -> List[Individual]:
		"""Generates a population of individuals.
		
		Returns:
			List[Individual]: The generated population.
		"""
		# TODO - implement population generation
		# 1. Generate individuals
		# 2. Return population

		population = []
		for i in range(self.population_size):
			for contvar in self.rulemeta.continuous_vars.values():
				cont = ContinuousVariable(name=contvar.name, \
					lbound=contvar.lbound, ubound=contvar.ubound, \
					correlation=contvar.correlation)

				if contvar.correlation > 0:
					cont.ubound = random.uniform(contvar.lbound, contvar.ubound)
				elif contvar.correlation < 0:
					cont.lbound = random.uniform(contvar.lbound, contvar.ubound)
				
				if self.min_interval:
					if cont.ubound - cont.lbound > self.min_interval:
						rule.add_continuous_variable(cont)

			population.append(Individual(rule=rule, fitness=0.0))

		return population
