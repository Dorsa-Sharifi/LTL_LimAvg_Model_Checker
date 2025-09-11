"""
Product Kripke Structure Implementation for LTL+LimAvg Model Checker
Constructs the product K × A of a Kripke structure and Büchi automaton.
"""

from core.kripke_structure import KripkeStructure


class ProductKripkeStructure(KripkeStructure):
    def __init__(self, kripke, buchi):
        super().__init__()
        self.original_kripke = kripke
        self.buchi = buchi
        self.product_states = {}  # (k_state, a_state) -> product_state_id
        self.reverse_mapping = {}  # product_state_id -> (k_state, a_state)
        self._construct_product()

    def _construct_product(self):
        """Construct the product automaton K × A."""
        state_counter = 0

        # Create product states
        for k_state in self.original_kripke.states:
            for a_state in self.buchi.states:
                product_state = f"({k_state},{a_state})"
                self.product_states[(k_state, a_state)] = product_state
                self.reverse_mapping[product_state] = (k_state, a_state)

                # Add to product structure
                self.add_state(product_state)

                # Copy atomic propositions from Kripke state
                for prop in self.original_kripke.get_atomic_props(k_state):
                    self.add_atomic_prop(product_state, prop)

                # Copy variables from Kripke state
                for var_name, value in self.original_kripke.variables.get(k_state, {}).items():
                    self.set_variable(product_state, var_name, value)

                # Mark as initial if both components are initial
                if (k_state in self.original_kripke.initial_states and
                        a_state in self.buchi.initial_states):
                    self.add_initial_state(product_state)

                state_counter += 1

        # Create product transitions
        for (k_state, a_state), product_state in self.product_states.items():
            k_atomic_props = self.original_kripke.get_atomic_props(k_state)

            # Get enabled automaton transitions
            enabled_a_transitions = self.buchi.get_enabled_transitions(a_state, k_atomic_props)

            # For each Kripke transition
            for k_next in self.original_kripke.get_successors(k_state):
                # For each enabled automaton transition
                for condition, a_next in enabled_a_transitions:
                    next_product_state = self.product_states.get((k_next, a_next))
                    if next_product_state:
                        self.add_transition(product_state, next_product_state)

    def is_accepting_state(self, product_state):
        """Check if a product state is accepting (Büchi accepting component)."""
        if product_state not in self.reverse_mapping:
            return False
        k_state, a_state = self.reverse_mapping[product_state]
        return a_state in self.buchi.accepting_states

    def get_accepting_states(self):
        """Get all accepting states in the product."""
        accepting = set()
        for product_state in self.states:
            if self.is_accepting_state(product_state):
                accepting.add(product_state)
        return accepting

    def get_kripke_state(self, product_state):
        """Get the Kripke component of a product state."""
        if product_state not in self.reverse_mapping:
            return None
        return self.reverse_mapping[product_state][0]

    def get_buchi_state(self, product_state):
        """Get the Büchi component of a product state."""
        if product_state not in self.reverse_mapping:
            return None
        return self.reverse_mapping[product_state][1]

    def __str__(self):
        result = f"Product Structure (K × A):\n"
        result += f"Original Kripke states: {len(self.original_kripke.states)}\n"
        result += f"Büchi states: {len(self.buchi.states)}\n"
        result += f"Product states: {len(self.states)}\n"
        result += f"Accepting states: {len(self.get_accepting_states())}\n"
        return result
