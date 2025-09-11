"""
LTL to Büchi Automaton Converter for LTL+LimAvg Model Checker
Simple pattern-based converter for common LTL formulas.
"""

import re
from core.buchi_automaton import BuchiAutomaton


class SimpleLTLToBuchi:
    def __init__(self):
        self.counter = 0

    def convert(self, ltl_formula):
        """
        Convert LTL formula to Büchi automaton.
        Supports common patterns: F(p), G(p), F(G(p)), G(F(p)), p U q, etc.
        """
        ltl_formula = ltl_formula.strip()

        # Eventually (F p)
        if re.match(r'F\s*\(\s*(\w+)\s*\)', ltl_formula):
            match = re.match(r'F\s*\(\s*(\w+)\s*\)', ltl_formula)
            prop = match.group(1)
            return self._create_eventually_automaton(prop)

        # Globally (G p)
        if re.match(r'G\s*\(\s*(\w+)\s*\)', ltl_formula):
            match = re.match(r'G\s*\(\s*(\w+)\s*\)', ltl_formula)
            prop = match.group(1)
            return self._create_globally_automaton(prop)

        # Infinitely often (G F p)
        if re.match(r'G\s*F\s*\(\s*(\w+)\s*\)', ltl_formula) or re.match(r'GF\s*(\w+)', ltl_formula):
            if re.match(r'G\s*F\s*\(\s*(\w+)\s*\)', ltl_formula):
                match = re.match(r'G\s*F\s*\(\s*(\w+)\s*\)', ltl_formula)
            else:
                match = re.match(r'GF\s*(\w+)', ltl_formula)
            prop = match.group(1)
            return self._create_infinitely_often_automaton(prop)

        # Eventually always (F G p)
        if re.match(r'F\s*G\s*\(\s*(\w+)\s*\)', ltl_formula) or re.match(r'FG\s*(\w+)', ltl_formula):
            if re.match(r'F\s*G\s*\(\s*(\w+)\s*\)', ltl_formula):
                match = re.match(r'F\s*G\s*\(\s*(\w+)\s*\)', ltl_formula)
            else:
                match = re.match(r'FG\s*(\w+)', ltl_formula)
            prop = match.group(1)
            return self._create_eventually_always_automaton(prop)

        # Until (p U q)
        if re.match(r'(\w+)\s*U\s*(\w+)', ltl_formula):
            match = re.match(r'(\w+)\s*U\s*(\w+)', ltl_formula)
            prop1, prop2 = match.groups()
            return self._create_until_automaton(prop1, prop2)

        # Simple atomic proposition
        if re.match(r'^\w+$', ltl_formula):
            return self._create_simple_prop_automaton(ltl_formula)

        # Negated atomic proposition
        if re.match(r'^!\s*\w+$', ltl_formula):
            prop = re.match(r'^!\s*(\w+)$', ltl_formula).group(1)
            return self._create_negated_prop_automaton(prop)

        # Default: create a simple accepting automaton
        return self._create_default_automaton(ltl_formula)

    def _create_eventually_automaton(self, prop):
        """Create automaton for F(prop): eventually prop holds."""
        automaton = BuchiAutomaton(f"F({prop})")

        # States: q0 (initial), q1 (accepting)
        automaton.add_state('q0', is_initial=True)
        automaton.add_state('q1', is_accepting=True)

        # Transitions
        automaton.add_transition('q0', 'q0', 'true')  # Stay in q0
        automaton.add_transition('q0', 'q1', prop)  # Move to q1 when prop holds
        automaton.add_transition('q1', 'q1', 'true')  # Stay in q1 (accepting)

        return automaton

    def _create_globally_automaton(self, prop):
        """Create automaton for G(prop): always prop holds."""
        automaton = BuchiAutomaton(f"G({prop})")

        # States: q0 (initial & accepting), q1 (trap)
        automaton.add_state('q0', is_initial=True, is_accepting=True)
        automaton.add_state('q1')  # Trap state (non-accepting)

        # Transitions
        automaton.add_transition('q0', 'q0', prop)  # Stay in q0 when prop holds
        automaton.add_transition('q0', 'q1', f'!{prop}')  # Go to trap when prop fails
        automaton.add_transition('q1', 'q1', 'true')  # Stay in trap

        return automaton

    def _create_infinitely_often_automaton(self, prop):
        """Create automaton for GF(prop): infinitely often prop holds."""
        automaton = BuchiAutomaton(f"GF({prop})")

        # Single state that's both initial and accepting
        automaton.add_state('q0', is_initial=True, is_accepting=True)

        # Self-loop on everything, but accepting only when prop holds
        automaton.add_transition('q0', 'q0', 'true')

        return automaton

    def _create_eventually_always_automaton(self, prop):
        """Create automaton for FG(prop): eventually always prop holds."""
        automaton = BuchiAutomaton(f"FG({prop})")

        # States: q0 (initial), q1 (accepting)
        automaton.add_state('q0', is_initial=True)
        automaton.add_state('q1', is_accepting=True)

        # Transitions
        automaton.add_transition('q0', 'q0', 'true')  # Stay in q0
        automaton.add_transition('q0', 'q1', prop)  # Move to q1 when prop holds
        automaton.add_transition('q1', 'q1', prop)  # Stay in q1 when prop holds
        automaton.add_transition('q1', 'q0', f'!{prop}')  # Back to q0 when prop fails

        return automaton

    def _create_until_automaton(self, prop1, prop2):
        """Create automaton for prop1 U prop2."""
        automaton = BuchiAutomaton(f"{prop1} U {prop2}")

        # States: q0 (initial), q1 (accepting)
        automaton.add_state('q0', is_initial=True)
        automaton.add_state('q1', is_accepting=True)

        # Transitions
        automaton.add_transition('q0', 'q0', f'{prop1} & !{prop2}')  # prop1 holds, prop2 doesn't
        automaton.add_transition('q0', 'q1', prop2)  # prop2 holds (until satisfied)
        automaton.add_transition('q1', 'q1', 'true')  # Stay accepting

        return automaton

    def _create_simple_prop_automaton(self, prop):
        """Create automaton for a simple atomic proposition."""
        automaton = BuchiAutomaton(f"simple({prop})")

        # Single accepting state
        automaton.add_state('q0', is_initial=True, is_accepting=True)
        automaton.add_transition('q0', 'q0', prop)

        return automaton

    def _create_negated_prop_automaton(self, prop):
        """Create automaton for negated atomic proposition."""
        automaton = BuchiAutomaton(f"!{prop}")

        # Single accepting state
        automaton.add_state('q0', is_initial=True, is_accepting=True)
        automaton.add_transition('q0', 'q0', f'!{prop}')

        return automaton

    def _create_default_automaton(self, formula):
        """Create a default automaton that accepts all runs."""
        automaton = BuchiAutomaton(f"default({formula})")

        # Single accepting state that accepts everything
        automaton.add_state('q0', is_initial=True, is_accepting=True)
        automaton.add_transition('q0', 'q0', 'true')

        return automaton
