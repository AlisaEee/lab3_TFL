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
        self.precedes = {}
        self.start_symbol = None

    def add_rule(self, lhs, rhs):
        if lhs not in self.rules:
            self.rules[lhs] = []
        self.rules[lhs].extend(rhs)
    def to_formatted_string(self):
        max_lhs_len = max(len(lhs) for lhs in self.rules)
        result = ""
        for lhs, rhs_list in self.rules.items():
            #print(lhs, rhs_list)
            rhs_str = " | ".join(" ".join(rhs) for rhs in rhs_list)
            result += f"{lhs:<{max_lhs_len}} -> {rhs_str}\n"
        return result
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

    def match_string(self, word):
        # Начинаем с начального символа и проверяем слово
        return self.match(self.start_symbol, word, 0)

    def match(self, symbol, word,depth):
        if not word: 
            return symbol == ''  

        if symbol not in self.rules:
            return symbol == word

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



# Функция для чтения грамматики из строки
def has_brackets(symbol, input_str):
    for line in input_str.split("\n"):
        line = line.strip()
        if line != "":
            # Парсим правило
            parts = line.split("->")
            if len(parts) == 2:
                lhs = parts[0].strip()
                if lhs == symbol:
                    return True
    
    return False
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
                    if symbol.startswith('[') and symbol.endswith(']') and has_brackets(symbol,input_str):
                        if current_sequence:
                            combined_rhs.append(current_sequence)
                            combined_rhs=[]
                        current_sequence.append(symbol)  # Убираем скобки
                    
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
                            if next_nonterminal.isupper():
                                stack.append(next_nonterminal)
    # 2. Переопределение правил для каждого множества
    def add_rule(new_rhs, nonterm):
        check=True
        curr_rules = grammar.rules[nonterm]
        for rule in curr_rules:
           # print('rule',rule,len(rule),all(symbol.isupper() for symbol in rule))
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
'''def matches(grammar, symbol, string, index,result):

    if index == len(string):
        return True
    
    for rule in grammar.rules[symbol]:
        current_index = index

        for part in rule:
            if part.islower(): 
                if current_index < len(string) and string[current_index] == part:
                    current_index += 1
                else:
                    break
            else:
                if not matches(grammar, part, string, current_index, result):
                    # Если не удалось сопоставить, выходим из цикла, но продолжаем с другим правилом
                    break
                current_index = current_index+1
        else:
            if current_index == len(string):  # Проверяем, достигли ли конца строки
                return True

    return False'''

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

#def belongs_to_language(grammar, string):
    #return matches(grammar, 'S', string, 0,0)

def random_walk(bigram_matrix,FIRST,LAST, start_symbol):
    current_symbol = start_symbol
    follow = FIRST[start_symbol]
    current_symbol = random.choice(list(follow))
    sequence = [current_symbol]
    while bigram_matrix[current_symbol] and random.random() >= 0.01:
        next_symbols = bigram_matrix[current_symbol]

        if not next_symbols:
            break

        current_symbol = random.choice(list(next_symbols))
        sequence.append(current_symbol)
    result_string = ''.join(sequence)

    return result_string

def genString():
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
            string = genString()
            f.write(string+' '+str(grammar.match_string(string))+'\n')
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
new_grammar = eliminate_chain_rules(grammar)
print(new_grammar.to_formatted_string())
new_grammar = convert(new_grammar)
new_grammar.set_start_symbol('S')

s = genString()
print("Сгенерированная строка:",s)
print("Принадлежит грамматике?")
print("String",s,new_grammar.match_string(s))

#generateTests(new_grammar,5)

print("String",string1,new_grammar.match_string(string1))
print(new_grammar.to_formatted_string())