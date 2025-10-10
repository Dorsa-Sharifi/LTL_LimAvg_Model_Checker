#!/usr/bin/env python3

import ply.lex as lex
import ply.yacc as yacc
from itertools import product


class LTLimProcessor:
    def __init__(self):
        self.tokens = [
            'IDENTIFIER', 'REAL',
            'NEG', 'AND', 'OR', 'IMPLIES', 'IFF',
            'NEXT', 'FINALLY', 'GLOBALLY', 'UNTIL', 'RELEASE',
            'SUM', 'AVG', 'LIM_INF_AVG', 'LIM_SUP_AVG',
            'GEQ', 'LEQ', 'EQ', 'GT', 'LT',
            'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
            'LPAREN', 'RPAREN'
        ]

        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(module=self, debug=False)
        self.variables = set()
        self.propositions = set()
        self.limit_avg_assertions = []

    # Lexer (same as before)
    def t_LIM_INF_AVG(self, t):
        r'LimInfAvg'
        return t

    def t_LIM_SUP_AVG(self, t):
        r'LimSupAvg'
        return t

    def t_SUM(self, t):
        r'Sum'
        return t

    def t_AVG(self, t):
        r'Avg'
        return t

    def t_NEXT(self, t):
        r'\bX\b'
        return t

    def t_FINALLY(self, t):
        r'\bF\b'
        return t

    def t_GLOBALLY(self, t):
        r'\bG\b'
        return t

    def t_UNTIL(self, t):
        r'\bU\b'
        return t

    def t_RELEASE(self, t):
        r'\bR\b'
        return t

    def t_NEG(self, t):
        r'¬|!'
        return t

    def t_AND(self, t):
        r'∧|&&'
        return t

    def t_OR(self, t):
        r'∨|\|\|'
        return t

    def t_IMPLIES(self, t):
        r'->|=>|→'
        return t

    def t_IFF(self, t):
        r'<->|<=>|↔'
        return t

    def t_GEQ(self, t):
        r'>='
        return t

    def t_LEQ(self, t):
        r'<='
        return t

    def t_EQ(self, t):
        r'='
        return t

    def t_GT(self, t):
        r'>'
        return t

    def t_LT(self, t):
        r'<'
        return t

    def t_PLUS(self, t):
        r'\+'
        return t

    def t_MINUS(self, t):
        r'-'
        return t

    def t_TIMES(self, t):
        r'\*'
        return t

    def t_DIVIDE(self, t):
        r'/'
        return t

    def t_LPAREN(self, t):
        r'\('
        return t

    def t_RPAREN(self, t):
        r'\)'
        return t

    def t_REAL(self, t):
        r'\d+\.\d+|\d+'
        t.value = float(t.value) if '.' in t.value else int(t.value)
        return t

    def t_IDENTIFIER(self, t):
        r'[a-zA-Z][a-zA-Z0-9_]*'
        return t

    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}'")
        t.lexer.skip(1)

    t_ignore = ' \t\n'

    # Precedence rules
    precedence = (
        ('left', 'OR'),
        ('left', 'AND'),
        ('right', 'IMPLIES'),
        ('right', 'IFF'),
        ('right', 'NEG'),
        ('left', 'UNTIL', 'RELEASE'),
        ('left', 'NEXT', 'FINALLY', 'GLOBALLY'),
    )

    # Parser (same as before)
    def p_formula(self, p):
        """
        formula : atomic_formula
                | LPAREN formula RPAREN
                | NEG formula
                | NEXT formula
                | FINALLY formula
                | GLOBALLY formula
                | formula AND formula
                | formula OR formula
                | formula IMPLIES formula
                | formula IFF formula
                | formula UNTIL formula
                | formula RELEASE formula
        """
        if len(p) == 2:
            p[0] = p[1]
        elif p[1] == '(':
            p[0] = ('paren', p[2])
        elif p[1] in ['¬', '!', 'X', 'F', 'G']:
            p[0] = (p[1], p[2])
        else:
            op_map = {
                '->': '→', '=>': '→', '→': '→',
                '<->': '↔', '<=>': '↔', '↔': '↔',
                '∧': '∧', '&&': '∧',
                '∨': '∨', '||': '∨',
                'U': 'U', 'R': 'R'
            }
            canonical_op = op_map.get(p[2], p[2])
            p[0] = (canonical_op, p[1], p[3])

    def p_atomic_formula(self, p):
        """
        atomic_formula : proposition
                      | path_assertion
                      | prefix_assertion
        """
        p[0] = p[1]

    def p_proposition(self, p):
        """proposition : IDENTIFIER"""
        self.propositions.add(p[1])
        p[0] = ('prop', p[1])

    def p_path_assertion(self, p):
        """
        path_assertion : LIM_INF_AVG LPAREN IDENTIFIER RPAREN GEQ REAL
                      | LIM_SUP_AVG LPAREN IDENTIFIER RPAREN GEQ REAL
        """
        self.variables.add(p[3])
        assertion = (p[1], p[3], p[5], p[6])
        self.limit_avg_assertions.append(assertion)
        p[0] = assertion

    def p_prefix_assertion(self, p):
        """
        prefix_assertion : arithmetic_expr GEQ arithmetic_expr
                        | arithmetic_expr LEQ arithmetic_expr
                        | arithmetic_expr GT arithmetic_expr
                        | arithmetic_expr LT arithmetic_expr
                        | arithmetic_expr EQ arithmetic_expr
        """
        p[0] = ('assert', p[1], p[2], p[3])

    def p_arithmetic_expr(self, p):
        """
        arithmetic_expr : term
                       | arithmetic_expr PLUS term
                       | arithmetic_expr MINUS term
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = (p[2], p[1], p[3])

    def p_term(self, p):
        """
        term : factor
             | term TIMES factor
             | term DIVIDE factor
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = (p[2], p[1], p[3])

    def p_factor(self, p):
        """
        factor : REAL
               | variable
               | sum_expr
               | avg_expr
               | LPAREN arithmetic_expr RPAREN
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = p[2]

    def p_variable(self, p):
        """variable : IDENTIFIER"""
        self.variables.add(p[1])
        p[0] = ('var', p[1])

    def p_sum_expr(self, p):
        """sum_expr : SUM LPAREN IDENTIFIER RPAREN"""
        self.variables.add(p[3])
        p[0] = ('Sum', p[3])

    def p_avg_expr(self, p):
        """avg_expr : AVG LPAREN IDENTIFIER RPAREN"""
        self.variables.add(p[3])
        p[0] = ('Avg', p[3])

    def p_error(self, p):
        if p:
            raise SyntaxError(f"Syntax error at '{p.value}'")
        else:
            raise SyntaxError("Syntax error at EOF")

    def parse(self, formula):
        """Parse formula and return parse tree"""
        self.variables.clear()
        self.propositions.clear()
        self.limit_avg_assertions.clear()
        return self.parser.parse(formula, lexer=self.lexer)

    def negate_formula(self, tree):
        """Step 1: Negate the parse tree to get ϕ = ¬ψ"""
        if tree is None:
            return None

        if isinstance(tree, tuple):
            op = tree[0]

            if op in ['prop', 'var', 'real', 'Sum', 'Avg']:
                return ('¬', tree)

            elif op in ['LimInfAvg', 'LimSupAvg']:
                # Negate path assertion: ¬(LimXAvg(v) ≥ c) ≡ LimXAvg(v) < c
                return (tree[0], tree[1], '<', tree[3])

            elif op == 'assert':
                # Negate comparison operators
                old_op = tree[2]
                neg_ops = {'>=': '<', '<=': '>', '>': '<=', '<': '>=', '=': '!='}
                new_op = neg_ops.get(old_op, f'¬{old_op}')
                return ('assert', tree[1], new_op, tree[3])

            elif op == '¬':
                return tree[1]  # Double negation

            elif op == 'X':
                return ('X', self.negate_formula(tree[1]))

            elif op == 'F':
                return ('G', self.negate_formula(tree[1]))

            elif op == 'G':
                return ('F', self.negate_formula(tree[1]))

            elif op == '∧':
                return ('∨', self.negate_formula(tree[1]), self.negate_formula(tree[2]))

            elif op == '∨':
                return ('∧', self.negate_formula(tree[1]), self.negate_formula(tree[2]))

            elif op == '→':
                # ¬(φ → ψ) ≡ φ ∧ ¬ψ
                return ('∧', tree[1], self.negate_formula(tree[2]))

            elif op == '↔':
                # ¬(φ ↔ ψ) ≡ (φ ∧ ¬ψ) ∨ (¬φ ∧ ψ)
                return ('∨',
                        ('∧', tree[1], self.negate_formula(tree[2])),
                        ('∧', self.negate_formula(tree[1]), tree[2]))

            elif op == 'U':
                return ('R', self.negate_formula(tree[1]), self.negate_formula(tree[2]))

            elif op == 'R':
                return ('U', self.negate_formula(tree[1]), self.negate_formula(tree[2]))

            elif op == 'paren':
                return ('paren', self.negate_formula(tree[1]))

            else:  # Arithmetic operators
                return tree

        return tree

    def tree_to_string(self, tree):
        """Convert parse tree to string"""
        if tree is None:
            return ""

        if isinstance(tree, tuple):
            op = tree[0]

            if op == 'prop':
                return tree[1]
            elif op == 'var':
                return tree[1]
            elif op == 'real':
                return str(tree[1])
            elif op in ['LimInfAvg', 'LimSupAvg']:
                return f"{op}({tree[1]}) {tree[2]} {tree[3]}"
            elif op == 'assert':
                return f"{self.tree_to_string(tree[1])} {tree[2]} {self.tree_to_string(tree[3])}"
            elif op in ['Sum', 'Avg']:
                return f"{op}({tree[1]})"
            elif op in ['¬', 'X', 'F', 'G']:
                return f"{op}({self.tree_to_string(tree[1])})"
            elif op in ['∧', '∨', '→', '↔', 'U', 'R']:
                return f"({self.tree_to_string(tree[1])} {op} {self.tree_to_string(tree[2])})"
            elif op == 'paren':
                return f"({self.tree_to_string(tree[1])})"
            else:
                return f"({self.tree_to_string(tree[1])} {op} {self.tree_to_string(tree[2])})"

        else:
            return str(tree)

    def extract_limit_avg_assertions(self, tree):
        """Extract all limit-average assertions from the parse tree"""
        assertions = []

        def traverse(node):
            if isinstance(node, tuple):
                if node[0] in ['LimInfAvg', 'LimSupAvg']:
                    assertions.append(node)
                else:
                    for child in node[1:]:
                        if isinstance(child, tuple) or isinstance(child, (int, float, str)):
                            traverse(child)

        traverse(tree)
        return assertions

    def replace_assertions_with_truth_values(self, tree, truth_assignments):
        """Replace limit-average assertions with truth values based on assignments"""
        if not isinstance(tree, tuple):
            return tree

        op = tree[0]

        # If this is a limit-average assertion, replace with truth value
        if op in ['LimInfAvg', 'LimSupAvg']:
            for assertion, truth_value in truth_assignments:
                if tree == assertion:
                    return 'T' if truth_value else 'F'
            return tree  # Should not happen

        # Recursively process children
        new_children = []
        for child in tree[1:]:
            if isinstance(child, tuple):
                new_children.append(self.replace_assertions_with_truth_values(child, truth_assignments))
            else:
                new_children.append(child)

        return (op,) + tuple(new_children)

    def build_boolean_combination(self, assertions, truth_values):
        """Build the Boolean combination of limit-average assertions"""
        if not assertions:
            return "T"  # No assertions, always true

        terms = []
        for i, (assertion, truth_value) in enumerate(zip(assertions, truth_values)):
            assertion_str = self.tree_to_string(assertion)
            if truth_value:
                terms.append(assertion_str)
            else:
                terms.append(f"¬{assertion_str}")

        if len(terms) == 1:
            return terms[0]
        else:
            return " ∧ ".join(f"({term})" for term in terms)

    def process_formula_negation(self, formula_psi):
        """Complete pipeline: Given formula ψ, process ¬ψ and detach limit-average assertions"""

        print("=" * 80)
        print(f"PROCESSING FORMULA ψ: {formula_psi}")
        print("=" * 80)

        # Step 1: Parse the original formula ψ
        print("Step 1: Parsing original formula ψ")
        tree_psi = self.parse(formula_psi)
        if tree_psi is None:
            raise ValueError("Failed to parse formula ψ")

        parsed_psi = self.tree_to_string(tree_psi)
        print(f"Parsed ψ: {parsed_psi}")

        # Step 2: Negate to get ϕ = ¬ψ
        print("\nStep 2: Negating formula to get ϕ = ¬ψ")
        tree_phi = self.negate_formula(tree_psi)
        formula_phi = self.tree_to_string(tree_phi)
        print(f"Negated formula ϕ: {formula_phi}")

        # Step 3: Extract limit-average assertions from ϕ
        assertions = self.extract_limit_avg_assertions(tree_phi)
        print(f"\nStep 3: Found {len(assertions)} limit-average assertions in ϕ:")
        for i, assertion in enumerate(assertions):
            print(f"  θ{i + 1}: {self.tree_to_string(assertion)}")

        if not assertions:
            print("No limit-average assertions found in ϕ. Formula is already standard LTL.")
            return [("T", formula_phi)]  # Single disjunct with no limit-average part

        # Step 4: Generate all possible truth assignments (2^n combinations)
        n = len(assertions)
        truth_assignments_list = list(product([True, False], repeat=n))
        print(f"\nStep 4: Generating {len(truth_assignments_list)} truth assignments")

        # Step 5: Build the disjunction for ϕ
        disjuncts = []
        print("Step 5: Building disjuncts for ϕ:")

        for truth_values in truth_assignments_list:
            # Build the LTL formula with assertions replaced by truth values
            ltl_formula_tree = self.replace_assertions_with_truth_values(tree_phi,
                                                                         list(zip(assertions, truth_values)))
            ltl_formula = self.tree_to_string(ltl_formula_tree)

            # Build the Boolean combination of limit-average assertions
            limit_avg_formula = self.build_boolean_combination(assertions, truth_values)

            disjunct = (limit_avg_formula, ltl_formula)
            disjuncts.append(disjunct)

            print(f"  Disjunct: {limit_avg_formula} ∧ {ltl_formula}")

        # Step 6: Return the final disjunction
        print(f"\nStep 6: Final disjunction has {len(disjuncts)} disjuncts")
        return disjuncts, formula_phi

    def complete_pipeline(self, formula_psi):
        """Complete processing pipeline as described in the paper"""
        print("COMPLETE MODEL CHECKING PIPELINE")
        print("=" * 80)
        print(f"Input formula ψ: {formula_psi}")
        print("=" * 80)

        try:
            # Process: ψ → ϕ = ¬ψ → disjunction of (χ ∧ ξ)
            disjuncts, negated_formula = self.process_formula_negation(formula_psi)

            print("\n" + "=" * 80)
            print("FINAL RESULT:")
            print("=" * 80)
            print(f"Original formula ψ: {formula_psi}")
            print(f"Negated formula ϕ = ¬ψ: {negated_formula}")
            print(f"Equivalent disjunction for model checking:")

            for i, (chi, xi) in enumerate(disjuncts):
                print(f"  Disjunct {i + 1}: {chi} ∧ {xi}")

            print(f"\nNext steps for each disjunct:")
            print(f"  1. Convert LTL formula ξ to Büchi automaton A")
            print(f"  2. Build product K × A (quantitative Kripke structure)")
            print(f"  3. Check if K × A has fair computation satisfying limit-average formula χ")

            return disjuncts, negated_formula

        except Exception as e:
            print(f"ERROR: {e}")
            return None, None


# Test cases
if __name__ == "__main__":
    processor = LTLimProcessor()

    test_formulas = [
        # Simple cases
        "LimInfAvg(x) >= 5",
        "LimSupAvg(y) >= 3.5 ∧ F p",

        # Multiple assertions
        "LimInfAvg(x) >= 5 ∧ LimSupAvg(y) >= 3.0",
        "F(LimInfAvg(y) >= 2.0) ∨ G(LimSupAvg(y) >= 4.0)",

        # Complex cases with implications
        "(LimInfAvg(x) >= 5 → F p) ∧ LimSupAvg(y) >= 3.0",
        "G((LimInfAvg(x) >= 2.0 ∧ p) → F q)",

        # Standard LTL formulas (should still work)
        "p U q",
        "F p ∧ G q",

        # Real-world example
        "G(request → (F response ∧ LimInfAvg(ServiceTime) >= 0.9))"
    ]

    for formula in test_formulas:
        disjuncts, negated = processor.complete_pipeline(formula)
        print("\n" + "=" * 80 + "\n")

        # Reset for next formula
        processor.variables.clear()
        processor.propositions.clear()
        processor.limit_avg_assertions.clear()