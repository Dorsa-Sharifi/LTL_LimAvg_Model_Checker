#TODO: Testing Kripke Structure
from model import KripkeStructure, kripke_parser
def parse_kripke(input_text):
    try:
        return kripke_parser.parse(input_text)
    except Exception as e:
        print(f"Parsing error: {e}")
        return None


if __name__ == "__main__":
    input_text = """
    STATES: {s0, s1, s2, s3};
    INIT: {s0, s3};
    TRANS: {(s0, s1), (s1, s2), (s2, s0), (s3, s2)};
    LABEL: {
        s0: {p, q},
        s1: {q}
    };
    VALUES: {
        s0: 3,
        s1: -5,
        s2: 3
    };
    """

    parsed = parse_kripke(input_text)
    if parsed:
        print("Successfully parsed Kripke structure:")
        print(parsed)

        kripke = KripkeStructure(parsed)
        kripke.visualize("kripke_structure.png")
    else:
        print("Failed to parse input")

#TODO: Testing LTL Parser
from ltl import LTLimModelChecker
from ltl.parser import LTLParser
from ltl.visualizer import ParseTreeVisualizer
import os

# #
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
    test_formula("G(p || LimSupAvg(u))")
    test_formula("F(p && Xq)")
    test_formula("G(p -> Fq)")
    test_formula("(p U q) && (r || s)")
    test_formula("!(Xp || Gq)")

