from .rule import Rule, RuleMeta, ContinuousVariable, DiscreteVariable
import random
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Union, Any
import math
import pandas as pd
from multiprocessing import Pool

from tqdm import tqdm

from utils.data_utils import DataContext


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
		self.parallel = kwargs.get('parallel', False)
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
		print("------------------------------")
		# print(rule)
		if not isinstance(rule, Rule):
			raise ValueError('Rule must be of type Rule')
		
		if len(rule.continuous_vars.keys()) > 0:
			# raise ValueError('Rule already contains continuous variables')
			print('Rule already contains continuous variables')
			# return rule

		# 1. Generate initial population
		population = self.__generate_population(rule)
		# 2. Evaluate population
		if self.parallel:
			population = self.__evaluate_parallel(population, eval_params)
		else:
			population = self.__evaluate(population, eval_params)
		for i in tqdm(range(self.generations)):
			# 3. Select parents
			evo_candidates = self.__select(population)
			# 4. Crossover parents
			# 5. Mutate offspring
			# 6. Evaluate offspring
			if self.parallel:
				new_generation = self.__crossover_parallel(evo_candidates)
				new_generation = self.__mutate_parallel(new_generation)
				new_generation = self.__evaluate_parallel(new_generation, eval_params)
			else:
				new_generation = self.__crossover(evo_candidates)
				new_generation = self.__mutate(new_generation)
				new_generation = self.__evaluate(new_generation, eval_params)

			# 7. Select survivors
			new_generation = self.__select(new_generation)
			# 8. Replace population with offspring
			population = self.__replace(population, new_generation)
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

	def __mutate_parallel(self, evo_candidates:List[Individual]) -> List[Individual]:
		"""Mutates the offspring.

		Args:
			evo_candidates (List[Individual]): The offspring to mutate.

		Returns:
			List[Individual]: The mutated offspring based on the mutation rate.
		"""
		# 1. Mutate offspring
		# 2. Return offspring

		number_to_mutate = math.ceil(len(evo_candidates) * self.mutation_rate)
		random.shuffle(evo_candidates)
		
		mutation_candidates = evo_candidates[:number_to_mutate]

		with Pool(processes=8) as pool:
			mutated = pool.map(self.__mutate_individual, mutation_candidates)

		return evo_candidates[number_to_mutate:] + mutated

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
			# value = random.uniform(cvar.lbound, cvar.ubound)
			# mutate upper bound if variable is positively correlated
			if cvar.correlation > 0:
				individual.rule.continuous_vars[cvar.name].ubound = value
			# mutate lower bound if variable is negatively correlated
			elif cvar.correlation < 0:
				individual.rule.continuous_vars[cvar.name].lbound = value

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

	def __crossover_parallel(self, evo_candidates:List[Individual]) -> List[Individual]:
		"""Crossover the selected individuals.

		Args:
			evo_candidates (List[Individual]): The selected individuals.

		Returns:
			List[Individual]: The offspring of the selected individuals.
		"""
		# 1. Crossover selected individuals
		# 2. Return offspring

		offspring =  []
		with Pool(processes=8) as pool:
			evo_pairs = [(evo_candidates[i], evo_candidates[i+1]) for i in range(0, len(evo_candidates)-2, 2)]
			result = pool.starmap(self.__crossover_pair, evo_pairs)
			for r in result:
				offspring.extend(r)
			
		return offspring

	def __crossover_pair(self, parent1:Individual, parent2:Individual) -> \
		List[Individual]:

		# 1. Crossover parents
		# 2. Return offspring

		# randomly pick 1 continuous variable to crossover
		if len(parent1.rule.continuous_vars) > 1:
			cvar = random.choice(list(parent1.rule.continuous_vars.values()))
		else:
			cvar = list(parent1.rule.continuous_vars.values())[0]

		# create offspring 1 with bounds from parent1
		offspring1 = parent1.rule.copy() # possible bug here

		# create offspring 2 with bounds from parent2
		offspring2 = parent2.rule.copy() # possible bug here

		# if cvar exists in both parents, do nothing (crossing bounds is just 
		# swapping upper bounds for positive correlation and lower bounds for
		# negative correlation)
		# So, we only need to check if cvar is missing in one of the parents
		if cvar.name not in offspring2.continuous_vars:
			# if variable is positively correlated
			# if cvar.correlation > 0:
			if len(offspring1.continuous_vars) == 1:
				# if both parents have only 1 continous variable each
				# do nothing

				# if parent2 has more than 1 continuous variable
				# add one missing variable to offspring1
				if len(offspring2.continuous_vars) > 1:
					nvar = random.choice(list(offspring2.continuous_vars.values()))
					offspring1.update_continuous_variable(nvar)
			
			# add missing variable to offspirng2
			offspring2.add_continuous_variable(cvar)

			# if parent1 has more than 1 continuous variable
			# remove cvar from offspring1
			if len(offspring1.continuous_vars) > 1:
				offspring1.remove_continuous_variable(cvar)


		offspring1 = Individual(rule=offspring1, fitness=parent1.fitness)
		offspring2 = Individual(rule=offspring2, fitness=parent2.fitness)
	
		# create offspring 3 as a duplicate of parent1
		# and upper bound from parent1
		offspring3 = parent1.rule.copy()
		offspring3 = Individual(rule=offspring3, fitness=parent1.fitness)

		# create offspring 4 as a duplicate of parent2
		offspring4 = parent2.rule.copy()
		offspring4 = Individual(rule=offspring4, fitness=parent2.fitness)

		offspring = [offspring1, offspring2, offspring3, offspring4]

		return offspring
		

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

		for individual in population:
			individual.fitness = self.__get_fitness(individual.rule, eval_params)
		
		population.sort(key=lambda x: x.fitness, reverse=True)

		return population

	def __evaluate_parallel(self, population:List[Individual], \
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

		results = []
		with Pool(processes=8) as pool:
			results = pool.starmap(self.__get_fitness, \
				[(individual.rule, eval_params) for individual in population])

		for i in range(len(population)):
			population[i].fitness = results[i]
		
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
		# if support < min_support: fitness -= len(rule data)
		# print(rule)
		if rule.evaluate(self.data, eval_params, self.datalen):
			fitness = 1.0
			for contvar in rule.continuous_vars.values():
				meta_range = \
					self.rulemeta.continuous_vars[contvar.name].ubound - \
					self.rulemeta.continuous_vars[contvar.name]\
					.lbound
				contvar_range = contvar.ubound - contvar.lbound
				fitness *= math.pow(contvar_range/meta_range, 2)
				# print(fitness)
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

		population = []
		contvars = list(self.rulemeta.continuous_vars.values())
		for i in range(self.population_size):
			newrule = Rule(rule.itemset[:], rule.target, discrete_vars=rule.discrete_vars.copy())
			numcontvars = random.randint(1, len(self.rulemeta.continuous_vars))
			for contvar in random.sample(contvars, numcontvars):
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
			return random.randint(math.floor(lower), math.ceil(upper))
		else:
			return random.uniform(lower, upper)
