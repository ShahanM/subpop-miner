import itertools
import math

from .rule import ContinuousVariable, Rule, RuleMeta


def get_covering_rules(rule: Rule, testpool: list[Rule]) -> list[Rule]:
	"""
	Finds rules in testpool that 'cover' the candidate rule.
	'Cover' means:
	1. Discrete vars are a subset (e.g., Parent has fewer/same discrete constraints).
	2. Continuous vars cover the range (e.g., Parent range includes Child range).
	"""
	covering_rules = []
	for frule in testpool:
		if frule.issubrule(rule):
			covering_rules.append(frule)
	return covering_rules


def delta_prune_empty(candidates: list[Rule], prune_delta: float) -> list[Rule]:
	# Sort shortest to longest to ensure parents (more general) are in 'finallist' first.
	# This allows the child (more specific) to be compared against them.
	sorted_candidates = sorted(candidates, key=lambda r: len(r.itemset))

	finallist = []

	for candidate in sorted_candidates:
		# Check against already accepted rules
		parents = get_covering_rules(candidate, finallist)

		should_prune = False
		for parent in parents:
			# If parent covers child AND threshold difference is small -> Prune Child
			if abs(parent.target_threshold - candidate.target_threshold) < prune_delta:
				should_prune = True
				break

		if not should_prune:
			finallist.append(candidate)

	return finallist


def delta_prune(finalcontrol: list[Rule], candidates: list[Rule], prune_delta: int) -> list[Rule]:
	outlist = []
	for candidate in candidates:
		parents = get_covering_rules(candidate, finalcontrol)

		is_pruned = False
		for parent in parents:
			if abs(parent.target_threshold - candidate.target_threshold) < prune_delta:
				is_pruned = True
				break

		if not is_pruned:
			outlist.append(candidate)

	return outlist


def prune_spanning_vars(rules: list[Rule], rulemeta: RuleMeta) -> list[Rule]:
	pruned = []
	for rule in rules:
		newrule = rule.copy()

		# Use a list to iterate safely while modifying
		for name, contvar in list(rule.continuous_vars.items()):
			meta_var = rulemeta.continuous_vars.get(name)
			if not meta_var:
				continue

			# Use isclose for float comparison safety
			is_spanning = math.isclose(contvar.lbound, meta_var.lbound) and math.isclose(
				contvar.ubound, meta_var.ubound
			)

			if is_spanning:
				newrule.remove_continuous_variable(contvar)

		pruned.append(newrule)
	return pruned


def prune_rules(rules: list[Rule]) -> list[Rule]:
	# Sort shortest to longest
	rules.sort(key=lambda rule: len(rule.itemset))
	prunedlist = []
	seen = []

	for rule in rules:
		is_covered = False
		for seen_rule in seen:
			if seen_rule.issubrule(rule):
				is_covered = True
				break

		if not is_covered:
			seen.append(rule)
			prunedlist.append(rule)

	return prunedlist


def remove_duplicate_rules(rules: list[Rule]) -> list[Rule]:
	"""
	Removes exact duplicate rules.
	A duplicate is defined as a rule with the exact same string representation/hash.
	Does NOT remove nested rules or subsets.
	"""
	seen = set()
	unique_rules = []
	for rule in rules:
		if rule not in seen:
			unique_rules.append(rule)
			seen.add(rule)
	return unique_rules


def inflate_with_cont_vars(rule: Rule, contvars: list[ContinuousVariable]):
	# Make a copy of the rule
	newrule = Rule(itemset=rule.itemset[:], target=rule.target, discrete_vars=rule.discrete_vars.copy())
	for contvar in contvars:
		ncvar = ContinuousVariable(
			name=contvar.name, lbound=contvar.lbound, ubound=contvar.ubound, correlation=contvar.correlation
		)
		newrule.add_continuous_variable(ncvar)
	return newrule


def inflate_with_quantitative_templates(candidates: list[Rule], rulemeta: RuleMeta, target_col: str) -> list[Rule]:
	_inflated: list[Rule] = []

	# Create a processing list that includes the original candidates
	#    PLUS an empty rule to generate the "Pure Continuous" templates.
	base_rules = candidates.copy() if candidates else []
	base_rules.append(Rule(itemset=[], target=target_col))

	contvars = list(rulemeta.continuous_vars.values())

	# Pre-calculate all combinations as LISTS to avoid iterator exhaustion
	all_combinations = []
	for i in range(1, len(contvars) + 1):
		all_combinations.extend(list(itertools.combinations(contvars, i)))

	# Apply every combination to every base rule
	for rule in base_rules:
		for combination in all_combinations:
			new_rule = inflate_with_cont_vars(rule, combination)
			_inflated.append(new_rule)

	return _inflated
