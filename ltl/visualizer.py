import networkx as nx
import matplotlib.pyplot as plt
from collections import deque


class ParseTreeVisualizer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0

    def visualize(self, parse_tree, filename=None):
        self.graph = nx.DiGraph()
        self.node_counter = 0
        root_id = self._build_tree(parse_tree)
        pos = self._calculate_positions(root_id)

        plt.figure(figsize=(12, 8))
        nx.draw(self.graph, pos,
                labels=nx.get_node_attributes(self.graph, 'label'),
                with_labels=True,
                node_size=2500,
                node_color='lightblue',
                font_size=10,
                arrows=True)

        plt.title("LTL Parse Tree")
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()

    def _build_tree(self, node, parent=None):
        node_id = self.node_counter
        self.node_counter += 1

        if isinstance(node, tuple):
            if len(node) == 2:  # Unary operator
                label = node[0]
                self.graph.add_node(node_id, label=label)
                if parent is not None:
                    self.graph.add_edge(parent, node_id)
                child_id = self._build_tree(node[1], node_id)
            else:  # Binary operator
                label = node[0]
                self.graph.add_node(node_id, label=label)
                if parent is not None:
                    self.graph.add_edge(parent, node_id)
                left_id = self._build_tree(node[1], node_id)
                right_id = self._build_tree(node[2], node_id)
        else:
            self.graph.add_node(node_id, label=str(node))
            if parent is not None:
                self.graph.add_edge(parent, node_id)

        return node_id

    def _calculate_positions(self, root_id):
        pos = {}
        queue = deque([(root_id, 0, 0, 1.0)])  # (node_id, x, y, width)

        while queue:
            node_id, x, y, width = queue.popleft()
            pos[node_id] = (x, -y)

            children = list(self.graph.successors(node_id))
            if children:
                child_width = width / len(children)
                for i, child in enumerate(children):
                    child_x = x - width / 2 + (i + 0.5) * child_width
                    queue.append((child, child_x, y + 1, child_width))

        return pos