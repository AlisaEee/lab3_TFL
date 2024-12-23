# Структура для хранения грамматики
import re
import random
MAX_DEPTH=3
class Grammar:
    def __init__(self):
        self.rules = {}
        self.first = {}
        self.follow = {}
        self.last = {}
        self.name_index = 0
        self.precedes = {}
        self.start_symbol = None

        self.isGenerating = {}
        self.counter = {}
        self.concernedRules = {}

    def add_rule(self, lhs, rhs):
        if lhs not in self.rules:
            self.rules[lhs] = []
        self.rules[lhs].extend(rhs)
         # Обновляем concernedRules для всех нетерминалов в правой части
        for rule in rhs:
            for symbol in rule:
                if symbol.isupper():  # Если это нетерминал
                    if symbol not in self.concernedRules:
                        self.concernedRules[symbol] = []
                    self.concernedRules[symbol].append(lhs)

                   
    def to_formatted_string(self):
        max_lhs_len = max(len(lhs) for lhs in self.rules)
        result = ""
        for lhs, rhs_list in self.rules.items():
            rhs_str = " | ".join("".join(rhs) for rhs in rhs_list)
            result += f"{lhs:<{max_lhs_len}} -> {rhs_str}\n"
        return result
    def PrepareHNF(self):
        self.long_delete()
        self = eliminate_chain_rules(self)
        gen = self.find_generating_non_terminals()
        nonreach = self.find_unreachable_rules()
        print(self.to_formatted_string())
    def find_generating_non_terminals(self):
        """Находит все порождающие нетерминалы."""
        # Инициализируем все нетерминалы как непорождающие
        for lhs in self.rules.keys():
            self.isGenerating[lhs] = False
            self.counter[lhs] = 0

        # Счетчик для каждого правила
        for lhs, rhs_list in self.rules.items():
            for rhs in rhs_list:
                for symbol in rhs:
                    if symbol.isupper():  # Если это нетерминал
                        self.counter[lhs] += 1

        queue = []
        for lhs in self.rules.keys():
            if self.counter[lhs] == 0:
                queue.append(lhs)
                self.isGenerating[lhs] = True
        
        for lhs, rhs_list in self.rules.items():
            for rhs in rhs_list:
                if all(not symbol.isupper() for symbol in rhs): 
                    self.isGenerating[lhs] = True
                    queue.append(lhs)

        while queue:
           current_non_terminal = queue.pop(0)
           for lhs in self.concernedRules.get(current_non_terminal, []):
                self.counter[lhs] -= 1  # Уменьшаем счетчик для всех для нетер

                if self.counter[lhs] == 0 and not self.isGenerating[lhs]: #Если счётчик порождающих обнулился, то пометим его порождающим.
                    self.isGenerating[lhs] = True
                    queue.append(lhs)
        gen = {non_term for non_term, is_gen in self.isGenerating.items() if is_gen}
        rules = self.rules.items()
        self.rules = {}
        for nonterminal, rightRules in rules:
            for rightRule in rightRules:
                flag = True
                if nonterminal in gen:
                    for symbol in rightRule:
                        if symbol.isupper() and symbol not in gen:
                            flag = False
                else:
                    flag = False
                if flag:       
                    if nonterminal not in self.rules:
                        self.rules[nonterminal] = []
                    self.rules[nonterminal].extend([rightRule])

        return gen
    def long_delete_rec(self, NT,name_index):
        for i, rule in enumerate(self.rules[NT]):
            if len(rule) > 2:
                new_name = f"[EXTRA-{NT + str(name_index)}]"
                name_index += 1
                #CONVERT OLD
                self.rules[NT][i] = [rule[0], new_name]
                # MAKE NEW
                self.rules[new_name] = [rule[1:]]
                self.long_delete_rec(new_name,name_index)
    def long_delete(self):
        index = 0 # start naming id(create new states unique)
        for nonterminal in list(self.rules.keys()):
            self.long_delete_rec(nonterminal,index)
        rules = self.rules.items()
        self.rules = {}
        for nonterminal, rightRules in rules:
            self.rules[nonterminal] = []
            for rightRule in rightRules:
                self.rules[nonterminal].extend([rightRule])
    def find_unreachable_rules(self):
        if not self.start_symbol or self.start_symbol not in self.rules:
            return set(self.rules.keys())  # Если начальный символ отсутствует, все правила недостижимы

        reachable = {self.start_symbol} 
        changed = True

        while changed:
            changed = False
            for lhs, rhs_list in self.rules.items():
                if lhs in reachable:  
                    for rhs in rhs_list:
                        for symbol in rhs:
                            if symbol.isupper() and symbol not in reachable:  # Если это нетерминал
                                reachable.add(symbol)  # Добавляем  в достижимые
                                changed = True

        unreachable = {lhs for lhs in self.rules.keys() if lhs not in reachable}
        rules = self.rules.items()
        self.rules = {}
        for nonterminal, rightRules in rules:
            self.rules[nonterminal] = []
            for rightRule in rightRules:
                if nonterminal not in unreachable:
                    self.rules[nonterminal].extend([rightRule])
        return unreachable
    def cyk(self, w):
        n = len(w)
        # Initialize the table
        T = [[set([]) for j in range(n)] for i in range(n)]
    
        for j in range(n):
            for lhs, rhs_list in self.rules.items():
                for rhs in rhs_list:
                    if len(rhs) == 1 and rhs[0] == w[j]:
                        T[j][j].add(lhs)
    
        # Fill the table for substrings of length 2 to n
        for length in range(2, n + 1):  
            for i in range(n - length + 1):  
                j = i + length - 1 
                for k in range(i, j):
                    for lhs, rhs_list in self.rules.items():
                        for rhs in rhs_list:
                            if len(rhs) == 2 and rhs[0] in T[i][k] and rhs[1] in T[k + 1][j]:
                                T[i][j].add(lhs)
        #for row in T:
         #   print(row)
        #print(T[0][n-1])
        if self.start_symbol in T[0][n-1]:
            return True
        else:
            return False

    def remove_left_rec(self):
        new_rules = {}
        N = self.rules.keys()
        
        for i, Ai in enumerate(N):
            productions = self.rules[Ai]
            non_left_recursive = []
            left_recursive = []

            for production in productions:
                if production[0] == Ai:  # Проверка на леворекурсивность
                    left_recursive.append(production[1:])  # Убираем Ai
                    prod = production[1:]
                    prod.append(production[0]+'\'')
                    print("PP",prod)
                    left_recursive.append(prod) 
                else:
                    non_left_recursive.append(production)
            print(left_recursive)
            if left_recursive:
                new_non_terminal = Ai + '\''
                new_rules[new_non_terminal] = left_recursive

                new_rules[Ai] = non_left_recursive + [prod + list(new_non_terminal) for prod in non_left_recursive]
            else:
                new_rules[Ai] = productions  # Если нет левой рекурсии, оставляем как есть
        self.rules = new_rules

    def compute_first(self):
        # Инициализация множеств FIRST для всех нетерминалов
        for A in self.rules:
            self.first[A] = set()

        # Добавление пустой строки для нетерминалов, которые могут её генерировать
        for A, rhs_list in self.rules.items():
            for rhs in rhs_list:
                if rhs == ['']:  # Пустая строка
                    self.first[A].add('')

        changed = True
        while changed:
            changed = False
            for A, rhs_list in self.rules.items():
                for rhs in rhs_list:
                    initial_size = len(self.first[A])
                    for symbol in rhs:
                        if symbol in self.first:  # Если символ - нетерминал
                            self.first[A].update(self.first[symbol] - {''})
                            if '' not in self.first[symbol]:
                                break
                        else:  # Если символ - терминал
                            self.first[A].add(symbol)
                            break
                    else:
                        # Если вся последовательность может привести к пустой строке
                        self.first[A].add('')

                    if len(self.first[A]) > initial_size:
                        changed = True
        return self.first

    def compute_follow(self):
        # Инициализация множеств FOLLOW для всех нетерминалов
        for A in self.rules:
            self.follow[A] = set()
        
        if self.start_symbol:
            self.follow[self.start_symbol].add('$')

        changed = True
        while changed:
            changed = False
            for A, rhs_list in self.rules.items():
                for rhs in rhs_list:
                    for i, symbol in enumerate(rhs):
                        if symbol in self.rules:  # Если символ - нетерминал
                            if i + 1 < len(rhs):
                                next_symbol = rhs[i + 1]
                                if next_symbol in self.first:
                                    self.follow[symbol].update(self.first[next_symbol] - {''})
                                else:
                                    self.follow[symbol].add(next_symbol)

                                if '' in self.first.get(next_symbol, set()):
                                    self.follow[symbol].update(self.follow[A])
                            else:
                                self.follow[symbol].update(self.follow[A])

                        if symbol not in self.follow:
                            self.follow[symbol] = set()

                        initial_size = len(self.follow[symbol])
                        if len(self.follow[symbol]) > initial_size:
                            changed = True
        return self.follow
    
    def compute_last(self):
        # Инициализация множеств LAST для всех нетерминалов
        for A in self.rules:
            self.last[A] = set()

        changed = True
        while changed:
            changed = False
            for A, rhs_list in self.rules.items():
                for rhs in rhs_list:
                    if rhs:
                        last_symbol = rhs[-1]
                        if last_symbol.islower():  # Если терминал
                            if last_symbol not in self.last[A]:
                                self.last[A].add(last_symbol)
                                changed = True
                        elif last_symbol.isupper():  # Если нетерминал
                            for symbol in self.last[last_symbol]:
                                if symbol not in self.last[A]:
                                    self.last[A].add(symbol)
                                    changed = True
        return self.last
    def reverse(self):
        # Функция для реверсирования правил грамматики
        reversed_rules = {}
        for lhs in self.rules:
            reversed_rules[lhs] = []
        for A, productions in self.rules.items():
            for production in productions:
                reversed_production = production[::-1]
                reversed_rules[A].append(reversed_production)
        grammar = Grammar()
        grammar.rules = reversed_rules
        return grammar
    
    def compute_precedes(self):
        # Функция для вычисления precedes
        precedes = self.reverse().compute_follow() 
        return precedes
    
    def set_start_symbol(self, symbol):
        self.start_symbol = symbol
    '''
    def match_string(self, word):
        # Начинаем с начального символа и проверяем слово
        return self.match(self.start_symbol, word, 0)

    def match(self, symbol, word,depth):
        for rule in self.rules[symbol]:
            if self.try_rule(rule, word,depth):  
                return True
        return False

    def try_rule(self, rule, word, depth=0):
        if depth > MAX_DEPTH:  # MAX_DEPTH - максимальная глубина рекурсии
            return False

        if len(rule) > len(word):
            return False

        index = 0
        for symbol in rule:
            if symbol in self.rules:  # Если символ - нетерминал
                found = False
                for i in range(len(word) - index + 1):  # Перебираем возможные длины совпадения
                    if self.match(symbol, word[index:index + i], depth + 1):  # Рекурсивно проверяем
                        found = True
                        index += i
                        break
                if not found:
                    return False
            else:  # Если символ - терминал
                if index < len(word) and symbol == word[index]:
                    index += 1
                else:
                    return False

        return index == len(word)
    '''


# Функция для чтения грамматики из строки
def has_brackets(symbol, input_str):
    found = False
    for line in input_str.split("\n"):
        line = line.strip()
        if line != "":
            # Парсим правило
            parts = line.split("->")
            if len(parts) == 2:
                lhs = parts[0].strip()
                if lhs == symbol: # Нашли символ с []
                    found = True
                elif lhs == symbol[1:-1]: # Нашли символ без []
                    found = False
    return found
def read_grammar(input_str):
    grammar = Grammar()
    for line in input_str.split("\n"):
        line = line.strip()
        if line != "":
            # Парсим правило
            parts = line.split("->")
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs_str = parts[1].strip()
                rhs = re.findall(r'[a-z]+|[A-Z][0-9]?|\[[A-Z]+[0-9]*\]', rhs_str)
                combined_rhs = []
                current_sequence = []
                for symbol in rhs:
                    #print("fdf",symbol,has_brackets(symbol,input_str))
                    if symbol.startswith('[') and symbol.endswith(']'):
                        if current_sequence:
                            combined_rhs.append(current_sequence)
                            combined_rhs=[]
                        if has_brackets(symbol,input_str):
                            current_sequence.append(symbol) 
                        else:
                            current_sequence.append(symbol[1:-1])# Убираем скобки
                    
                    else:
                        current_sequence.append(symbol)  # Добавляем терминал или одиночный нетерминал
                if current_sequence:
                    combined_rhs.append(current_sequence)
                # Добавляем в грамматику
                grammar.add_rule(lhs, combined_rhs)
    return grammar

def get_bigram_matrix(grammar,FIRST,FOLLOW,LAST,PRECEDES):
    all_rules = set()
    for rhs_list in grammar.rules.values():
        for rhs in rhs_list:
            all_rules.add(tuple(rhs))
    # Извлекаем терминалы
    terminals = {symbol for symbol in FOLLOW.keys() if symbol.islower()} 
    bigram_matrix={}
    
    # Создаем биграмм-матрицу
    for x in terminals:
        bigram_matrix[x] = set()
    for y1 in terminals:
        for y2 in terminals:
            pattern = re.escape(y1) + re.escape(y2)
            exists_in_rules = any(
                re.search(pattern, ''.join(rule)) for rule in all_rules
            )
            
            if exists_in_rules:
                #print("Found in rules:", (y1, y2))
                bigram_matrix[y1].add(y2)
            elif any((
                any(y1 in LAST[nt] and y2 in FOLLOW[nt] for nt in grammar.rules.keys()),
                any(y1 in PRECEDES[p1] and y2 in FIRST[p1] for p1 in grammar.rules.keys()),
                any(y1 in LAST[p1] and y2 in FIRST[p2] and y2 in FOLLOW[p1]
                    for p1 in grammar.rules.keys() for p2 in grammar.rules.keys())
            )):
                bigram_matrix[y1].add(y2)

    return bigram_matrix

def display(array,type_task):
    print(type_task)
    for symbol, curr_set in array.items():
        print(f"{type_task}({symbol}) = {curr_set}")

def eliminate_chain_rules(grammar):
    # 1. Построение множеств N
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
                    for rhs in grammar.rules[current]:
                        for next_nonterminal in rhs: 
                            if next_nonterminal.isupper() and len(rhs) == 1:
                                stack.append(next_nonterminal)
    # 2. Переопределение правил для каждого множества
    def add_rule(new_rhs, nonterm):
        check=True
        curr_rules = grammar.rules[nonterm]
        for rule in curr_rules:
            if all(symbol.isupper() for symbol in rule) and len(rule) == 1:  # Если правило состоит только из ожиночного нетерминалов
                for n in rule:
                    if nonterm==n:
                        check = False
                    if n.isupper()and nonterm!=n:
                        add_rule(new_rhs, n)  # Рекурсивно добавляем
            else:
                if check:
                    new_rhs.append(rule)  # Добавляем правило, если оно содержит терминалы
    new_rules = {}
    for nonterminal in N:
        new_rhs = []
        add_rule(new_rhs, nonterminal)
        new_rules[nonterminal] = [list(rule) for rule in set(tuple(r) for r in new_rhs)]
    
    # Создаем новую грамматику
    new_grammar = Grammar()
    for nonterminal, rhs in new_rules.items():
        new_grammar.add_rule(nonterminal, rhs)
    

    # 3. Удаление недостижимых правил
    start_symbol = grammar.start_symbol
    reachable = set()
    stack = [start_symbol]
    while stack:
        current = stack.pop()
        if current not in reachable:
            reachable.add(current)
            if current in new_grammar.rules:
                for rule in new_grammar.rules[current]:
                    for symbol in rule:
                        if symbol.isupper():  # Если символ - нетерминал, добавляем его в стек
                            stack.append(symbol)
    
    final_rules = Grammar()
    for lhs in reachable:
        if lhs in new_grammar.rules:
            final_rules.add_rule(lhs, new_grammar.rules[lhs])
    
    return final_rules

def convert(grammar):
    new_grammar = Grammar()

    for nonterminal in grammar.rules:
        for rule in grammar.rules[nonterminal]:
            new_rule = []
            for part_rule in rule:
                if part_rule.islower():
                    new_rule.extend([char for char in part_rule]) 
                else:
                    if grammar.rules[part_rule]:
                        new_rule.append(part_rule)
            # Добавляем новое правило в новую грамматику
            new_grammar.add_rule(nonterminal, [new_rule])
    return new_grammar

def random_walk(bigram_matrix,FIRST,LAST, start_symbol):
    current_symbol = start_symbol
    follow = FIRST[start_symbol]
    current_symbol = random.choice(list(follow))
    sequence = [current_symbol]
    while bigram_matrix[current_symbol] and random.random() >= 0.07:
        next_symbols = bigram_matrix[current_symbol]

        if not next_symbols:
            break

        current_symbol = random.choice(list(next_symbols))
        sequence.append(current_symbol)
    result_string = ''.join(sequence)

    return result_string

def genString(new_grammar):
    FIRST = new_grammar.compute_first()
    FOLLOW = new_grammar.compute_follow()
    LAST = new_grammar.compute_last()
    PRECEDES = new_grammar.compute_precedes()

    # Выводим результаты
    '''
    display(FIRST,"FIRST")
    display(FOLLOW,"FOLLOW")
    display(LAST,"LAST")
    display(PRECEDES,"PRECEDES")
    '''
    bigram_matrix = get_bigram_matrix(new_grammar,FIRST,FOLLOW,LAST,PRECEDES)
    #print(bigram_matrix)
    sequence = random_walk(bigram_matrix,FIRST,LAST, new_grammar.start_symbol)
    return sequence

def generateTests(grammar,number):
    with open('tests.txt', 'w') as f:
        for i in range(number):
            string = genString(grammar)
            f.write(string+' '+str(grammar.cyk(string))+'\n')
    f.close()
def read_grammar_from_file(filepath):
    with open(filepath, 'r') as f:
        grammar_string = f.read()
    return grammar_string

# Example usage
file_path = "grammar.txt" # Replace with the actual path to your file
grammar_string = read_grammar_from_file(file_path)
'''
input_grammar = """
S -> acd[BC9] 
S -> [BC9]ddA 
A -> h 
BC9 -> d
"""
'''
string1 = "babbbbababaaaaaabaa"
string2 = "acdh"
string3 = "acddg"
string4 = "dddh"
# Читаем грамматику
grammar = read_grammar(grammar_string)
grammar.set_start_symbol('S')
grammar.PrepareHNF()
grammar.to_formatted_string()
print(grammar.cyk(list("bbb")))

s = genString(grammar)
print("Сгенерированная строка:",s)
print("Принадлежит грамматике?")
print("String",len('abababaababa'),grammar.cyk('abababaababa'))

generateTests(grammar,15)
