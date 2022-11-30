from typing import List

from .rule import Rule, RuleMeta


def delta_prune_empty(candidates: List[Rule], prune_delta: int)\
    -> List[Rule]:
    pruned = []
    finallist = []
    for candidate in candidates:
        subrules = overlap(candidate, finallist)
        for subrule in subrules:
            if abs(subrule.target_threshold - \
                candidate.target_threshold) < prune_delta:
                pruned.append(candidate)
                break
        else:
            finallist.append(candidate)
    return finallist

def overlap(rule: Rule, testpool: List[Rule]) -> List[Rule]:
    # First test discrete vars
    dvarset = set(rule.discrete_vars.values())
    cvarset = set(rule.continuous_vars.values())

    itemset = dvarset.union(cvarset)
    overlappingrules = []
    for frule in testpool:
        fdvarset = set(frule.discrete_vars.values())
        fcvarset = set(frule.continuous_vars.values())
        fitemset = fdvarset.union(fcvarset)
        if fitemset.issubset(itemset):
            overlappingrules.append(frule)
    return overlappingrules

def prune_rules(rules: List[Rule]) -> List[Rule]:
    rules.sort(key=lambda rule: len(rule.itemset))
    prunedlist = []
    seen = []
    for rule in rules:
        dvarset = set(rule.discrete_vars.values())
        cvarset = set(rule.continuous_vars.values())

        rset = dvarset.union(cvarset)
        # rset = set(rule.itemset)
        if not skiprule(rset, seen):
            seen.append(rset)
            prunedlist.append(rule)
    return prunedlist


def skiprule(rset: set, flst: List[set]) -> bool:
	if len(rset) <= 1:
		return False
	for frule in flst:
		if frule.issubset(rset):
			return True
	else:
		return False


def delta_prune(finalcontrol: List[Rule], candidates: List[Rule], \
    prune_delta: int) -> List[Rule]:
    outlist = []
    for candidate in candidates:
        subrules = overlap(candidate, finalcontrol)
        pruned = filter(lambda subrule: abs(subrule.target_threshold - \
            candidate.target_threshold) < prune_delta, subrules)
        if len(list(pruned)) == 0:
            outlist.append(candidate)
    return outlist


def prune_spanning_vars(rules: List[Rule], rulemeta: RuleMeta) -> List[Rule]:
    pruned = []
    for rule in rules:
        newrule = rule.copy()
        contvars = rule.continuous_vars.values()
        print(contvars)
        for contvar in contvars:
            print('rule cont: {} \n meta cont: {}'.format(contvar, rulemeta.continuous_vars[contvar.name]))
            if contvar.ubound == rulemeta.continuous_vars[contvar.name].ubound \
                and contvar.lbound == rulemeta.continuous_vars[contvar.name].lbound:
                newrule.remove_continuous_variable(contvar)
        print('newrule:', newrule)
        pruned.append(newrule)
    return pruned


def remove_duplicate_rules(rules: List[Rule]) -> List[Rule]:
    seen = []
    for rule in rules:
        unseen = True
        for srule in seen:
            if srule.issubrule(rule):
                unseen = False
                break
            if rule.issubrule(srule):
                seen.remove(srule)
                break
        else:
            if unseen:
                seen.append(rule)
    return seen

