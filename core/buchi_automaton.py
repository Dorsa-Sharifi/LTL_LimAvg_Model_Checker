#!/usr/bin/env python3
from core.ltllim_parser import LTLimProcessor


class NBWState:
    """Represents a state in the NBW"""

    def __init__(self, name, labels, incoming=None, outgoing=None):
        self.name = name
        self.labels = labels  # Set of propositions that must be true
        self.incoming = incoming if incoming else set()
        self.outgoing = outgoing if outgoing else set()
        self.is_initial = False
        self.is_accepting = False

    def __repr__(self):
        return f"State({self.name}, labels={self.labels}, accepting={self.is_accepting})"


class NBW:
    """Nondeterministic Büchi Automaton"""

    def __init__(self):
        self.states = set()
        self.initial_states = set()
        self.alphabet = set()  # Set of all possible atomic propositions
        self.transitions = {}  # dict: (state, symbol) -> set of next states

    def add_state(self, state):
        self.states.add(state)
        if state.is_initial:
            self.initial_states.add(state)

    def add_transition(self, from_state, symbol, to_state):
        key = (from_state, frozenset(symbol))
        if key not in self.transitions:
            self.transitions[key] = set()
        self.transitions[key].add(to_state)

        # Update state connections
        from_state.outgoing.add(to_state)
        to_state.incoming.add(from_state)

    def __repr__(self):
        return f"NBW(states={len(self.states)}, initial={len(self.initial_states)}, transitions={len(self.transitions)})"


class LTLToNBWConverter:
    """Converts LTL formulas to Nondeterministic Büchi Automata"""

    def __init__(self):
        self.state_counter = 0
        self.formula_cache = {}  # Cache for formula expansions

    def get_new_state_name(self):
        name = f"q{self.state_counter}"
        self.state_counter += 1
        return name

    def expand_formula(self, formula):
        """Recursively expand LTL formula using tableau rules"""
        if formula in self.formula_cache:
            return self.formula_cache[formula]

        if isinstance(formula, str):
            # Atomic proposition or truth value
            result = {frozenset([formula])}

        elif formula[0] == '¬':
            # Negation
            inner = formula[1]
            if isinstance(inner, tuple) and inner[0] == '¬':
                # ¬¬φ ≡ φ
                result = self.expand_formula(inner[1])
            elif inner == 'T':
                result = {frozenset(['F'])}
            elif inner == 'F':
                result = {frozenset(['T'])}
            else:
                result = {frozenset([f'¬{inner}'])}

        elif formula[0] == '∧':
            # Conjunction: φ ∧ ψ
            left, right = formula[1], formula[2]
            left_expansion = self.expand_formula(left)
            right_expansion = self.expand_formula(right)

            result = set()
            for l_set in left_expansion:
                for r_set in right_expansion:
                    # Merge sets, handling contradictions
                    merged = l_set.union(r_set)
                    if not self.has_contradiction(merged):
                        result.add(frozenset(merged))

        elif formula[0] == '∨':
            # Disjunction: φ ∨ ψ
            left, right = formula[1], formula[2]
            left_expansion = self.expand_formula(left)
            right_expansion = self.expand_formula(right)
            result = left_expansion.union(right_expansion)

        elif formula[0] == 'X':
            # Next: Xφ
            inner = formula[1]
            result = {frozenset([f'X({self.tree_to_string(inner)})'])}

        elif formula[0] == 'F':
            # Finally: Fφ ≡ φ ∨ XFφ
            inner = formula[1]
            inner_expansion = self.expand_formula(inner)
            next_expansion = self.expand_formula(('X', formula))
            result = inner_expansion.union(next_expansion)

        elif formula[0] == 'G':
            # Globally: Gφ ≡ φ ∧ XGφ
            inner = formula[1]
            inner_expansion = self.expand_formula(inner)
            next_expansion = self.expand_formula(('X', formula))

            result = set()
            for i_set in inner_expansion:
                for n_set in next_expansion:
                    merged = i_set.union(n_set)
                    if not self.has_contradiction(merged):
                        result.add(frozenset(merged))

        elif formula[0] == 'U':
            # Until: φ U ψ ≡ ψ ∨ (φ ∧ X(φ U ψ))
            left, right = formula[1], formula[2]
            right_expansion = self.expand_formula(right)
            left_expansion = self.expand_formula(left)
            next_expansion = self.expand_formula(('X', formula))

            # ψ ∨ (φ ∧ X(φ U ψ))
            result = right_expansion.copy()

            for l_set in left_expansion:
                for n_set in next_expansion:
                    merged = l_set.union(n_set)
                    if not self.has_contradiction(merged):
                        result.add(frozenset(merged))

        else:
            # For other operators, treat as atomic for now
            formula_str = self.tree_to_string(formula)
            result = {frozenset([formula_str])}

        self.formula_cache[formula] = result
        return result

    def has_contradiction(self, label_set):
        """Check if a set of labels contains contradictions"""
        labels = set(label_set)

        # Check for p and ¬p
        for label in labels:
            if label.startswith('¬'):
                positive = label[1:]
                if positive in labels:
                    return True
            else:
                if f'¬{label}' in labels:
                    return True

        # Check for T and F
        if 'T' in labels and 'F' in labels:
            return True

        return False

    def tree_to_string(self, tree):
        """Convert parse tree to string (same as previous implementation)"""
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

    def ltl_to_nbw(self, ltl_formula):
        """Main method: Convert LTL formula to NBW"""
        print(f"Converting LTL formula to NBW: {ltl_formula}")

        # Reset state
        self.state_counter = 0
        self.formula_cache = {}

        # Parse the formula if it's a string
        if isinstance(ltl_formula, str):
            # For now, we'll work with the string representation
            # In a full implementation, we'd parse it to a tree structure
            formula_tree = self.simplify_ltl_formula(ltl_formula)
        else:
            formula_tree = ltl_formula

        # Create NBW
        nbw = NBW()

        # Create initial state with the full formula
        initial_state = NBWState(
            self.get_new_state_name(),
            self.expand_formula(formula_tree)
        )
        initial_state.is_initial = True
        nbw.add_state(initial_state)

        # Build the automaton using tableau construction
        self.build_automaton(nbw, initial_state, formula_tree)

        # Set accepting states (for Büchi condition)
        self.set_accepting_states(nbw, formula_tree)

        print(f"NBW construction complete: {nbw}")
        return nbw

    def simplify_ltl_formula(self, formula_str):
        """Simplify LTL formula string for processing"""
        # Remove unnecessary parentheses and normalize
        # This is a simplified version - full implementation would parse properly
        return formula_str

    def build_automaton(self, nbw, current_state, formula):
        """Recursively build the automaton structure"""
        # This is a simplified implementation
        # Full implementation would use the tableau method with eventualities

        current_expansion = self.expand_formula(formula)

        for label_set in current_expansion:
            # Create new state for this expansion
            new_state = NBWState(
                self.get_new_state_name(),
                label_set
            )
            nbw.add_state(new_state)

            # Add transition
            symbol = self.extract_symbol_from_labels(label_set)
            nbw.add_transition(current_state, symbol, new_state)

            # Recursively build from new state if needed
            if self.has_temporal_operators(label_set):
                self.build_automaton(nbw, new_state, formula)

    def extract_symbol_from_labels(self, label_set):
        """Extract the current symbol (atomic propositions) from label set"""
        symbol = set()
        for label in label_set:
            if not (label.startswith('X(') or label.startswith('F(') or label.startswith('G(') or label.startswith(
                    'U(')):
                if label not in ['T', 'F'] and not label.startswith('¬'):
                    symbol.add(label)
        return symbol

    def has_temporal_operators(self, label_set):
        """Check if label set contains temporal operators"""
        for label in label_set:
            if label.startswith('X(') or label.startswith('F(') or label.startswith('G(') or label.startswith('U('):
                return True
        return False

    def set_accepting_states(self, nbw, formula):
        """Set accepting states based on Büchi conditions"""
        # For F(φ): states where φ is satisfied
        # For φ U ψ: states where ψ is satisfied
        # This is a simplified version

        formula_str = self.tree_to_string(formula)

        for state in nbw.states:
            # Check if this state satisfies all eventualities
            if self.satisfies_eventualities(state, formula):
                state.is_accepting = True

    def satisfies_eventualities(self, state, formula):
        """Check if a state satisfies all eventualities in the formula"""
        # Simplified implementation
        # Full version would track eventualities and ensure they're fulfilled

        labels_str = [str(label) for label in state.labels]

        # For F(φ), check if φ is in current labels
        if isinstance(formula, tuple) and formula[0] == 'F':
            inner_str = self.tree_to_string(formula[1])
            if inner_str in labels_str:
                return True

        # For φ U ψ, check if ψ is in current labels
        if isinstance(formula, tuple) and formula[0] == 'U':
            right_str = self.tree_to_string(formula[2])
            if right_str in labels_str:
                return True

        # Default: accept all states for simplicity
        return True

    def print_nbw(self, nbw):
        """Print the NBW in a readable format"""
        print("\n" + "=" * 50)
        print("NONDETERMINISTIC BÜCHI AUTOMATON")
        print("=" * 50)

        print(f"States: {len(nbw.states)}")
        print(f"Initial states: {[s.name for s in nbw.initial_states]}")
        print(f"Accepting states: {[s.name for s in nbw.states if s.is_accepting]}")

        print("\nTransitions:")
        for (from_state, symbol), to_states in nbw.transitions.items():
            for to_state in to_states:
                print(f"  {from_state.name} --[{set(symbol)}]--> {to_state.name}")

        print("\nState details:")
        for state in nbw.states:
            accepting = " (accepting)" if state.is_accepting else ""
            initial = " (initial)" if state.is_initial else ""
            print(f"  {state.name}: labels={set(state.labels)}{initial}{accepting}")


# Integration with our previous processor
class EnhancedLTLimProcessor(LTLimProcessor):
    """Enhanced processor that includes NBW conversion"""

    def __init__(self):
        super().__init__()
        self.nbw_converter = LTLToNBWConverter()

    def complete_pipeline_with_nbw(self, formula_psi):
        """Complete pipeline including NBW conversion"""
        print("COMPLETE MODEL CHECKING PIPELINE WITH NBW CONVERSION")
        print("=" * 80)

        # Steps 1-6: Process formula and detach limit-average assertions
        disjuncts, negated_formula = self.complete_pipeline(formula_psi)

        if not disjuncts:
            return

        print("\n" + "=" * 80)
        print("NBW CONVERSION STEP")
        print("=" * 80)

        # Convert each LTL formula ξ to NBW
        nbw_automata = []
        for i, (chi, xi) in enumerate(disjuncts):
            print(f"\n--- Converting disjunct {i + 1} to NBW ---")
            print(f"LTL formula ξ: {xi}")

            try:
                nbw = self.nbw_converter.ltl_to_nbw(xi)
                nbw_automata.append((chi, xi, nbw))
                self.nbw_converter.print_nbw(nbw)
            except Exception as e:
                print(f"Error converting to NBW: {e}")
                nbw_automata.append((chi, xi, None))

        return nbw_automata


# Test the NBW conversion
if __name__ == "__main__":

    from pymodelchecking import LTLLogic, Kripke

    processor = EnhancedLTLimProcessor()

    # Test with simple LTL formulas first
    test_formulas = [
        "F p",  # Eventually p
        "G p",  # Always p
        "p U q",  # p until q
        "F G p",  # Infinitely often p
        "p ∧ q",  # p and q
        "X p",  # Next p
    ]

    print("TESTING NBW CONVERSION WITH SIMPLE LTL FORMULAS")
    print("=" * 80)

    for formula in test_formulas:
        print(f"\n>>> Processing formula: {formula}")
        print("-" * 40)

        # Test direct NBW conversion
        converter = LTLToNBWConverter()
        nbw = converter.ltl_to_nbw(formula)
        converter.print_nbw(nbw)

        print("-" * 40)

    # Test full pipeline with limit-average assertions
    print("\n\nTESTING FULL PIPELINE WITH LIMIT-AVERAGE ASSERTIONS")
    print("=" * 80)

    complex_formulas = [
        "LimInfAvg(X) >= 5",
        "F p ∧ LimSupAvg(Y) >= 3.0",
    ]

    for formula in complex_formulas:
        nbw_automata = processor.complete_pipeline_with_nbw(formula)
        print("\n" + "=" * 80 + "\n")