from typing import Dict, Set, Tuple, List, Optional
from graphviz import Digraph

class QuantitativeKripkeStructure:
    def __init__(self,
                 states: Set[str],
                 init_state: str,
                 edges: Set[Tuple[str, str]],
                 boolean_vars: Set[str],
                 logical_formulas: Dict[str, Set[str]],
                 numeric_values: Dict[str, Dict[str, float]]):

        self.states = states
        self.init_state = init_state
        self.edges = edges
        self.boolean_vars = boolean_vars
        self.logical_formulas = logical_formulas
        self.numeric_values = numeric_values

        self.accepting_states = set()
        self.fairness_sets = []

        self.V = set()
        for var_dict in numeric_values.values():
            self.V.update(var_dict.keys())

        self.validate()

    def validate(self):
        if self.init_state not in self.states:
            raise ValueError(f"Initial state {self.init_state} not in states")

        for src, dst in self.edges:
            if src not in self.states:
                raise ValueError(f"Source state {src} in edge doesn't exist")
            if dst not in self.states:
                raise ValueError(f"Target state {dst} in edge doesn't exist")

        for state, true_vars in self.logical_formulas.items():
            for var in true_vars:
                if var not in self.boolean_vars:
                    raise ValueError(f"Boolean variable {var} in state {state} not declared")

        for state in self.states:
            if state not in self.numeric_values:
                raise ValueError(f"State {state} missing numeric values")
            for var in self.V:
                if var not in self.numeric_values[state]:
                    raise ValueError(f"State {state} missing value for numeric variable {var}")

        outgoing = {s: False for s in self.states}
        for (src, _) in self.edges:
            outgoing[src] = True

        for s, has_out in outgoing.items():
            if not has_out:
                raise ValueError(f"State {s} has no outgoing transitions")

    def get_boolean_valuation(self, state: str) -> Dict[str, bool]:
        return {var: var in self.logical_formulas.get(state, set())
                for var in self.boolean_vars}

    def get_numeric_valuation(self, state: str) -> Dict[str, float]:
        return self.numeric_values[state].copy()

    def get_label(self, state: str) -> Tuple[Dict[str, bool], Dict[str, float]]:
        return (self.get_boolean_valuation(state),
                self.get_numeric_valuation(state))

    def __str__(self) -> str:
        lines = [
            f"States: {self.states}",
            f"Initial state: {self.init_state}",
            f"Edges: {self.edges}",
            "Boolean variables:",
            "\n".join(f"  {s}: {self.get_boolean_valuation(s)}" for s in self.states),
            "Numeric variables:",
            "\n".join(f"  {s}: {self.numeric_values[s]}" for s in self.states)
        ]
        return "\n".join(lines)

    def add_accepting_state(self, state: str) -> None:
        if state not in self.states:
            raise ValueError(f"State {state} not in QKS")
        self.accepting_states.add(state)

    def add_fairness_set(self, states: Set[str]) -> None:
        if not states.issubset(self.states):
            raise ValueError("Some states not in QKS")
        self.fairness_sets.append(states)

    def compute_limit_average(self, path: List[str], var: str) -> float:
        if not path:
            raise ValueError("Path cannot be empty")
        if var not in next(iter(self.numeric_values.values())):
            raise ValueError(f"Variable {var} not in QKS")

        return sum(self.numeric_values[s][var] for s in path) / len(path)

    def check_path_fairness(self, path: List[str]) -> bool:
        if not self.accepting_states and not self.fairness_sets:
            return True

        if self.accepting_states:
            return any(s in self.accepting_states for s in path)

        for fs in self.fairness_sets:
            if not any(s in fs for s in path):
                return False
        return True
