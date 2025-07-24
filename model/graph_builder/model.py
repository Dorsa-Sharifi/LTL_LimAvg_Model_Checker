import networkx as nx
import matplotlib.pyplot as plt


class KripkeStructure:
    def __init__(self, parsed_data):
        self.graph = nx.DiGraph()
        self.build(parsed_data)

    def build(self, parsed_data):
        # Add states
        for state in parsed_data['states']:
            self.graph.add_node(state, value=None)

        # Set initial states
        for state in parsed_data['initial']:
            if state in self.graph.nodes:
                self.graph.nodes[state]['initial'] = True
                self.graph.nodes[state]['color'] = 'gold'

        # Add transitions
        for src, dst in parsed_data['transitions']:
            if src in self.graph.nodes and dst in self.graph.nodes:
                self.graph.add_edge(src, dst)

        # Add labels (atomic propositions)
        for state, labels in parsed_data['labels'].items():
            if state in self.graph.nodes:
                self.graph.nodes[state]['labels'] = labels

        # Add values
        for state, value in parsed_data['values'].items():
            if state in self.graph.nodes:
                self.graph.nodes[state]['value'] = value

    def visualize(self, filename=None):
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(self.graph, seed=42)

        # Draw nodes
        node_colors = [
            'gold' if self.graph.nodes[node].get('initial', False)
            else 'lightblue'
            for node in self.graph.nodes()
        ]

        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9
        )

        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos,
            arrowstyle='->',
            arrowsize=20,
            width=1.5
        )

        # Create enhanced labels with all information
        labels = {}
        for node in self.graph.nodes():
            label = node
            if 'value' in self.graph.nodes[node]:
                label += f"\nValue: {self.graph.nodes[node]['value']}"
            if 'labels' in self.graph.nodes[node]:
                label += f"\nLabels: {', '.join(self.graph.nodes[node]['labels'])}"
            labels[node] = label

        nx.draw_networkx_labels(
            self.graph, pos,
            labels=labels,
            font_size=10,
            font_family='monospace',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.5')
        )

        plt.title("Kripke Structure with State Values", pad=20)
        plt.axis('off')

        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()