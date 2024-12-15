import re

class Grammar:
    def __init__(self):
        self.rules = {}
    def add_rule(self, lhs, rhs):
        self.rules[lhs] = rhs  # Добавляем правило как список
    def to_formatted_string(self):
        max_lhs_len = max(len(lhs) for lhs in self.rules)
        result = ""
        for lhs, rhs_list in self.rules.items():
            rhs_str = " | ".join(rhs_list)
            result += f"{lhs:<{max_lhs_len}} -> {rhs_str}\n"
        return result
def eliminate_chain_rules(grammar):
    # 1. Build N sets
    N = {}
    for nonterminal in grammar.rules:
        N[nonterminal] = set()
        visited = set()
        stack = [nonterminal]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                N[nonterminal].add(current)
            if current in grammar.rules:
                for next_nonterminal in grammar.rules[current]:
                    if next_nonterminal.isupper(): #check if it's nonterminal
                        stack.append(next_nonterminal)
    
    # 2. Redefine rules
    new_rules = {}
    for nonterminal in grammar.rules:
        new_rhs = []
        for rhs in grammar.rules[nonterminal]:
            if rhs.isupper(): #check if it's nonterminal
                for n in N[rhs]:
                    if n in grammar.rules:
                        for r in grammar.rules[n]:
                            if not r.isupper(): #check if it's terminal
                                new_rhs.append(r)
            else:
                new_rhs.append(rhs)
        new_rules[nonterminal] = list(set(new_rhs))
    
    new_grammar = Grammar()
    for nonterminal, rhs in new_rules.items():
        new_grammar.add_rule(nonterminal, rhs)                   
    
    # 3. Remove unreachable rules
    start_symbol = list(grammar.rules.keys())[0] # Assuming start symbol is the first key
    reachable = set()
    stack = [start_symbol]
    while stack:
        current = stack.pop()
        if current not in reachable:
            reachable.add(current)
            if current in new_grammar.rules:
                for rule in new_grammar.rules[current]:
                    if rule.isupper():
                        stack.append(rule)
    
    
    final_rules = Grammar()
    for lhs in reachable:
        if lhs in new_grammar.rules:
            final_rules.add_rule(lhs, new_grammar.rules[lhs])

    return final_rules
def read_grammar2(input_str):
    grammar = Grammar()
    for line in input_str.split("\n"):
        line = line.strip()
        if line != "":
            # Парсим правило
            parts = line.split("->")
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs = [r.strip() for r in parts[1].split("|")]
                grammar.add_rule(lhs, rhs)
    return grammar
input_grammar = """
S -> a | A
A -> B | d
B->D
D-> H| g
H->k
"""
def belongs_to_grammar(grammar, input_string, start_symbol='S'):
    stack = [([start_symbol], 0)]  # Stack of (derivation, index_in_input_string) pairs
    
    while stack:
        derivation, index = stack.pop()
        
        if index == len(input_string): #If derivation matches the input_string length, it's valid
            return True
        
        if not derivation: # If derivation is empty and we didn't match the string, it's not valid.
            continue
            

        current_symbol = derivation[-1]

        if current_symbol.islower():  # Terminal symbol
            if current_symbol == input_string[index]:
                stack.append((derivation[:-1], index + 1))
            
        elif current_symbol in grammar.rules:  # Non-terminal symbol
            for next_symbols in grammar.rules[current_symbol]:
                new_derivation = derivation[:-1] + list(next_symbols) #Replace nonterminal with its derivations
                stack.append((new_derivation, index)) # Add to stack for backtracking
'''[
A
−
Z
]
[
0
−
9
]
?
[A−Z][0−9]?', rhs_str)'''

    return False  # No successful derivation found
grammar = read_grammar2(input_grammar)
new_grammar = eliminate_chain_rules(grammar)
print(new_grammar.to_formatted_string())
def get_bigram_matrix(grammar,FIRST,FOLLOW,LAST,PRECEDES):
        # Получаем все правила
        all_rules = set()
        for rhs_list in grammar.rules.values():
            for rhs in rhs_list:
                all_rules.add(tuple(rhs))

        print(all_rules)

        # Создаем биграмм-матрицу
        bigram_matrix = {}
        for y1 in FIRST.keys():
            for y2 in FIRST.keys():
                bigram_matrix[(y1, y2)] = any((
                    any((y1, y2) in zip(rule[:-1], rule[1:]) for rule in all_rules),
                    any(y1 in LAST[nt] and y2 in FOLLOW[nt] for nt in grammar.rules.keys()),
                    any(y1 in PRECEDES[p2] and y2 in FIRST[p1] for p1 in grammar.rules.keys() for p2 in grammar.rules.keys()),
                    any(y1 in LAST[p1] and y2 in FIRST[p1] and y2 in FOLLOW[p1]
                        for p1 in grammar.rules.keys() for p2 in grammar.rules.keys())
                ))
        return bigram_matrix Он должен строить матрицу пы