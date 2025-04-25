# from model.graph_builder import StateTransitionGraph
# from model.lexer_parser import parser
#
# def parse_input(input_text):
#     """Parse input text handling multiple formulas per state"""
#     parsed_data = {}
#     for line in input_text.strip().split('\n'):
#         if not line.strip():
#             continue
#         result = parser.parse(line)
#         if result:
#             key, value = result
#             # Convert v_s1 to s1 in values
#             if key == 'values':
#                 value = [(k.replace('v_', ''), v) for k, v in value]
#             parsed_data[key] = value
#     return parsed_data
#
# if __name__ == "__main__":
#     # Example with multiple formulas
#     input_text = """
#     states = {s1, s2, s3}
#     initstate = s1
#     edges = {(s1, s2), (s1, s3)}
#     logicalformulas = {s1: {p, q}, s2: {p, r}, s3: {t}}
#     v_s1 = 3, v_s2 = -5, v_s3 = 1
#     """
#
#     parsed_data = parse_input(input_text)
#     print("Parsed Data:", parsed_data)
#
#     graph = StateTransitionGraph()
#     graph.build_from_parsed_data(parsed_data)
#     result = graph.visualize("graph.png")
#     if result is None:  # Visualization failed
#         print("Rendered text representation instead")
#         graph.text_display()


from ltl.parser import LTLParser
from ltl.visualizer import ParseTreeVisualizer
import os


def test_formula(formula):
    parser = LTLParser()
    visualizer = ParseTreeVisualizer()

    try:
        print(f"\nFormula: {formula}")

        # Create safe filename
        safe_name = "".join(c if c.isalnum() else "_" for c in formula)[:20]

        # Parse original
        tree = parser.parse(formula)
        print("Original:", tree)
        visualizer.visualize(tree, f"original_{safe_name}.png")

        # Parse negated
        negated = parser.negate(tree)
        print("Negated:", negated)
        visualizer.visualize(negated, f"negated_{safe_name}.png")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    test_formula("F(p && Xq)")
    test_formula("G(p -> Fq)")
    test_formula("(p U q) && (r || s)")
    test_formula("!(Xp || Gq)")