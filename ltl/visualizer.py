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

        # Build the tree and get root node ID
        root_id = self._build_tree(parse_tree)

        # Calculate positions for all nodes
        pos = self._calculate_positions(root_id)

        plt.figure(figsize=(12, 8))

        # Draw with consistent ordering
        node_order = list(self.graph.nodes())
        nx.draw(self.graph, pos,
                nodelist=node_order,
                labels=nx.get_node_attributes(self.graph, 'label'),
                with_labels=True,
                node_size=2500,
                node_color='lightblue',
                font_size=10,
                arrows=True)

        plt.title("LTL Parse Tree")
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()

    def _build_tree(self, node, parent=None):
        node_id = self.node_counter
        self.node_counter += 1
        self.graph.add_node(node_id)

        # Set node label - modified to remove 'atomic' prefix
        if isinstance(node, tuple):
            if node[0] == 'atomic':
                # Display just the proposition name for atomic nodes
                self.graph.nodes[node_id]['label'] = str(node[1])
            elif len(node) == 2 and node[0] in ['LimSupAvg', 'LimInfAvg']:
                self.graph.nodes[node_id]['label'] = f"{node[0]}(...)"
                self._build_tree(node[1], node_id)
            elif len(node) == 2:
                self.graph.nodes[node_id]['label'] = node[0]
                self._build_tree(node[1], node_id)
            else:
                self.graph.nodes[node_id]['label'] = node[0]
                self._build_tree(node[1], node_id)
                self._build_tree(node[2], node_id)
        else:
            self.graph.nodes[node_id]['label'] = str(node)

        if parent is not None:
            self.graph.add_edge(parent, node_id)

        return node_id

    def _calculate_positions(self, root_id):
        pos = {}
        queue = deque([(root_id, 0, 0, 1.0)])  # (node_id, x, y, width)
        visited = set()

        while queue:
            node_id, x, y, width = queue.popleft()

            if node_id in visited:
                continue
            visited.add(node_id)

            pos[node_id] = (x, -y)
            children = list(self.graph.successors(node_id))

            if children:
                child_width = width / len(children)
                for i, child in enumerate(children):
                    child_x = x - width / 2 + (i + 0.5) * child_width
                    queue.append((child, child_x, y + 1, child_width))

        return pos