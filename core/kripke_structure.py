"""
Kripke Structure Implementation for LTL+LimAvg Model Checker
Represents system models with states, transitions, and atomic propositions.
"""


class KripkeStructure:
    def __init__(self):
        self.states = set()
        self.initial_states = set()
        self.transitions = {}  # state -> set of next states
        self.atomic_props = {}  # state -> set of atomic propositions
        self.variables = {}  # state -> dict of variable assignments

    def add_state(self, state):
        """Add a state to the Kripke structure."""
        self.states.add(state)
        if state not in self.transitions:
            self.transitions[state] = set()
        if state not in self.atomic_props:
            self.atomic_props[state] = set()
        if state not in self.variables:
            self.variables[state] = {}

    def add_initial_state(self, state):
        """Mark a state as initial."""
        self.add_state(state)
        self.initial_states.add(state)

    def add_transition(self, from_state, to_state):
        """Add a transition between states."""
        self.add_state(from_state)
        self.add_state(to_state)
        self.transitions[from_state].add(to_state)

    def add_atomic_prop(self, state, prop):
        """Add an atomic proposition to a state."""
        self.add_state(state)
        self.atomic_props[state].add(prop)

    def set_variable(self, state, var_name, value):
        """Set a variable value for a state."""
        self.add_state(state)
        self.variables[state][var_name] = value

    def get_successors(self, state):
        """Get all successor states."""
        return self.transitions.get(state, set())

    def get_atomic_props(self, state):
        """Get atomic propositions for a state."""
        return self.atomic_props.get(state, set())

    def get_variable_value(self, state, var_name):
        """Get variable value for a state."""
        return self.variables.get(state, {}).get(var_name, 0)

    def __str__(self):
        result = "Kripke Structure:\n"
        result += f"States: {self.states}\n"
        result += f"Initial States: {self.initial_states}\n"
        result += f"Transitions: {dict(self.transitions)}\n"
        result += f"Atomic Props: {dict(self.atomic_props)}\n"
        result += f"Variables: {dict(self.variables)}\n"
        return result
