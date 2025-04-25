import networkx as nx
import matplotlib.pyplot as plt
from warnings import warn


class StateTransitionGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.init_state = None

    def build_from_parsed_data(self, parsed_data):
        """Build Kripke model from parsed data with visualization support"""
        try:
            # Add states with attributes
            self.graph.add_nodes_from(parsed_data['states'])

            # Set initial state
            self.init_state = parsed_data.get('initstate')
            if self.init_state:
                self.graph.nodes[self.init_state]['color'] = 'gold'
                self.graph.nodes[self.init_state]['shape'] = 'doublecircle'

            # Add edges
            self.graph.add_edges_from(parsed_data.get('edges', []))

            # Set atomic propositions
            for state, formulas in parsed_data.get('logicalformulas', []):
                self.graph.nodes[state]['formulas'] = formulas

            # Set values
            for state, value in parsed_data.get('values', []):
                clean_state = state.replace('v_', '')
                if clean_state in self.graph.nodes:
                    self.graph.nodes[clean_state]['value'] = value

            # Initialize LTL formula storage
            for node in self.graph.nodes:
                if 'ltl' not in self.graph.nodes[node]:
                    self.graph.nodes[node]['ltl'] = []

        except Exception as e:
            warn(f"Build error: {str(e)}")

    def add_ltl_formula(self, state, formula):
        """Add parsed LTL formula to a state"""
        from ltl.parser import LTLParser
        try:
            if 'ltl' not in self.graph.nodes[state]:
                self.graph.nodes[state]['ltl'] = []
            self.graph.nodes[state]['ltl'].append(LTLParser().parse(formula))
        except KeyError:
            warn(f"State {state} not found")
        except Exception as e:
            warn(f"LTL parsing failed: {str(e)}")

    def visualize(self, save_path=None, show_ltl=False):
        """Visualize the Kripke model with optional LTL formulas"""
        plt.figure(figsize=(12, 8))

        # Create layout
        pos = nx.spring_layout(self.graph, seed=42, k=0.5)

        # Draw nodes
        node_colors = [data.get('color', 'lightblue')
                       for _, data in self.graph.nodes(data=True)]
        node_shapes = [data.get('shape', 'circle')
                       for _, data in self.graph.nodes(data=True)]

        # Draw edges first
        nx.draw_networkx_edges(
            self.graph, pos,
            arrowstyle='->',
            arrowsize=20,
            width=1.5,
            edge_color='gray'
        )

        # Draw nodes with different shapes
        for shape in set(node_shapes):
            nodes = [n for n in self.graph.nodes
                     if self.graph.nodes[n].get('shape', 'circle') == shape]
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=nodes,
                node_shape='o' if shape == 'circle' else 'd',
                node_color=[self.graph.nodes[n]['color'] for n in nodes],
                node_size=2000
            )

        # Create labels
        labels = {}
        for node, data in self.graph.nodes(data=True):
            label = f"{node}"
            if 'value' in data:
                label += f"\nValue: {data['value']}"
            if 'formulas' in data and data['formulas']:
                label += f"\nPropositions: {', '.join(data['formulas'])}"
            if show_ltl and 'ltl' in data and data['ltl']:
                label += f"\nLTL: {len(data['ltl'])} formula(s)"
            labels[node] = label

        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos,
            labels=labels,
            font_size=8,
            font_family='monospace',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )

        plt.title("Kripke Model Visualization")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def visualize_counterexample(self, path, save_path=None):
        """Highlight a counterexample path"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, seed=42)

        # Color all nodes
        node_colors = ['lightgray' for _ in self.graph.nodes]

        # Highlight path nodes
        for i, node in enumerate(path):
            node_index = list(self.graph.nodes).index(node)
            intensity = 0.3 + 0.7 * (i / len(path))
            node_colors[node_index] = (1, 0.5 - intensity / 2, 0.5 - intensity / 2)

        # Draw
        nx.draw(
            self.graph, pos,
            node_color=node_colors,
            with_labels=True,
            font_weight='bold',
            arrows=True
        )

        # Highlight path edges
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(
            self.graph, pos,
            edgelist=path_edges,
            edge_color='red',
            width=2,
            arrows=True
        )

        plt.title(f"Counterexample (length {len(path)})")
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()