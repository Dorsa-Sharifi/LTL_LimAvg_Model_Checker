"""
Büchi Automaton Implementation for LTL+LimAvg Model Checker
Represents ω-regular properties with accepting states.
"""



class BuchiAutomaton:
    def __init__(self, name=""):
        self.name = name
        self.states = set()
        self.initial_states = set()
        self.accepting_states = set()
        self.transitions = {}  # state -> list of (condition, next_state)

    def add_state(self, state, is_initial=False, is_accepting=False):
        """Add a state to the automaton."""
        self.states.add(state)
        if state not in self.transitions:
            self.transitions[state] = []
        if is_initial:
            self.initial_states.add(state)
        if is_accepting:
            self.accepting_states.add(state)

    def add_transition(self, from_state, to_state, condition):
        """Add a transition with a condition."""
        self.add_state(from_state)
        self.add_state(to_state)
        self.transitions[from_state].append((condition, to_state))

    def evaluate_condition(self, condition, atomic_props):
        """
        Evaluate a transition condition against atomic propositions.
        Supports: 'true', 'prop', '!prop', 'prop1 & prop2', 'prop1 | prop2'
        """
        if condition == 'true':
            return True
        if condition == 'false':
            return False

        # Handle negation
        if condition.startswith('!'):
            prop = condition[1:]
            return prop not in atomic_props

        # Handle conjunctions and disjunctions
        if ' & ' in condition:
            parts = condition.split(' & ')
            return all(self.evaluate_condition(part.strip(), atomic_props) for part in parts)

        if ' | ' in condition:
            parts = condition.split(' | ')
            return any(self.evaluate_condition(part.strip(), atomic_props) for part in parts)

        # Simple atomic proposition
        return condition in atomic_props

    def get_enabled_transitions(self, state, atomic_props):
        """Get all enabled transitions from a state given atomic propositions."""
        enabled = []
        for condition, next_state in self.transitions.get(state, []):
            if self.evaluate_condition(condition, atomic_props):
                enabled.append((condition, next_state))
        return enabled

    def __str__(self):
        result = f"Büchi Automaton '{self.name}':\n"
        result += f"States: {self.states}\n"
        result += f"Initial: {self.initial_states}\n"
        result += f"Accepting: {self.accepting_states}\n"
        result += "Transitions:\n"
        for state, trans_list in self.transitions.items():
            for condition, next_state in trans_list:
                result += f"  {state} --[{condition}]--> {next_state}\n"
        return result
