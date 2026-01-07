import spot
import json
import sys
import re

def ltl_to_nbw_hoa(ltl_formula):
    try:
        print(f"Original formula: {ltl_formula}", file=sys.stderr)
        
        formula = ltl_formula
        formula = formula.replace('∧', '&').replace('∨', '|').replace('¬', '!')
        formula = formula.replace('→', '->').replace('↔', '<->')
        
        print(f"Converted formula: {formula}", file=sys.stderr)
        
        automaton = spot.translate(formula, 'BA')
        
        return {
            'success': True,
            'hoa_format': automaton.to_str('hoa'),
            'states': automaton.num_states(),
            'edges': automaton.num_edges(),
            'acceptance': str(automaton.get_acceptance()),
            'is_deterministic': automaton.is_deterministic(),
            'original_formula': ltl_formula,
            'spot_formula': formula
        }
    except Exception as e:
        return {
            'success': False, 
            'error': str(e), 
            'original_formula': ltl_formula
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ltl_formula = sys.argv[1]
        result = ltl_to_nbw_hoa(ltl_formula)
        print(json.dumps(result))
    else:
        print(json.dumps({'success': False, 'error': 'No formula provided'}))
