#!/home/otebook/.local/share/pipx/venvs/z3-solver/bin/python
import sys
import json
import traceback

try:
    import z3
except ImportError as e:
    print(json.dumps({'success': False, 'error': f'Z3 import failed: {e}'}))
    sys.exit(1)

def parse_limit_avg_formula(formula, variables):
    """Parse limit-average formula into Z3 constraints - COMPLETE VERSION"""
    print(f"DEBUG: Parsing formula: {formula}", file=sys.stderr)
    
    constraints = []
    
    if formula == "true":
        print("DEBUG: Formula is 'true' - no constraints", file=sys.stderr)
        return constraints
    elif formula == "false":
        print("DEBUG: Formula is 'false' - always unsatisfiable", file=sys.stderr)
        return [('contradiction',)]  
    
    is_negated = formula.startswith('¬')
    if is_negated:
        formula = formula[1:]  
        print(f"DEBUG: Processing negated formula: {formula}", file=sys.stderr)
    
    if "LimInfAvg" in formula or "LimSupAvg" in formula:
        import re
        
        pattern = r'(LimInfAvg|LimSupAvg)\((\w+)\)\s*([\^<>=]+)\s*([\d.]+)'
        matches = re.findall(pattern, formula)
        
        for match in matches:
            avg_type, var, operator, value_str = match
            value = float(value_str)
            
            operator = operator.replace('^<', '<').replace('^>', '>')
            
            print(f"DEBUG: Found constraint: {avg_type}({var}) {operator} {value}", file=sys.stderr)
            
            if is_negated:
                print(f"DEBUG: Negating constraint", file=sys.stderr)
                if operator == '<':
                    operator = '>='
                elif operator == '<=':
                    operator = '>'
                elif operator == '>':
                    operator = '<='
                elif operator == '>=':
                    operator = '<'
                elif operator == '=':
                    operator = '!='
                print(f"DEBUG: Negated constraint: {avg_type}({var}) {operator} {value}", file=sys.stderr)
            
            # Convert to Z3 constraints
            if operator == '<':
                constraints.append(('lt', var, value))
            elif operator == '<=':
                constraints.append(('le', var, value))
            elif operator == '>':
                constraints.append(('gt', var, value))
            elif operator == '>=':
                constraints.append(('ge', var, value))
            elif operator == '=':
                constraints.append(('eq', var, value))
            elif operator == '!=':
                constraints.append(('ne', var, value))
    
    if " ∧ " in formula and not constraints:
        print(f"DEBUG: Boolean combination detected: {formula}", file=sys.stderr)
        parts = formula.split(" ∧ ")
        for part in parts:
            part_constraints = parse_limit_avg_formula(part.strip(), variables)
            constraints.extend(part_constraints)
    
    return constraints

def solve_lp_feasibility(cycle_vectors, variables, constraints):
    try:
        print(f"DEBUG: Starting LP with {len(cycle_vectors)} vectors", file=sys.stderr)
        solver = z3.Solver()
        
        if any(constraint[0] == 'contradiction' for constraint in constraints):
            print("DEBUG: Contradiction found - returning unsatisfiable", file=sys.stderr)
            return {'feasible': False}
        
        weights = [z3.Real(f'w_{i}') for i in range(len(cycle_vectors))]
        
        for w in weights:
            solver.add(w >= 0)
        solver.add(z3.Sum(weights) == 1)
        
        result_vars = {var: z3.Real(var) for var in variables}
        
        for var in variables:
            weighted_sum = z3.Sum([weights[i] * z3.RealVal(cycle_vectors[i][var]) 
                                 for i in range(len(cycle_vectors))])
            solver.add(result_vars[var] == weighted_sum)
        
        for constraint in constraints:
            if len(constraint) == 3:  
                op, var, value = constraint
                if op == 'lt':
                    solver.add(result_vars[var] < value)
                elif op == 'le':
                    solver.add(result_vars[var] <= value)
                elif op == 'gt':
                    solver.add(result_vars[var] > value)
                elif op == 'ge':
                    solver.add(result_vars[var] >= value)
                elif op == 'eq':
                    solver.add(result_vars[var] == value)
                elif op == 'ne':
                    solver.add(result_vars[var] != value)
        
        print("DEBUG: Checking satisfiability...", file=sys.stderr)
        result = solver.check()
        print(f"DEBUG: Z3 result: {result}", file=sys.stderr)
        
        if result == z3.sat:
            model = solver.model()
            weights_result = []
            for w in weights:
                try:
                    val = model.eval(w)
                    weights_result.append(float(val.as_fraction()))
                except:
                    weights_result.append(float(str(val)))
            
            result_point = {}
            for var in variables:
                try:
                    val = model.eval(result_vars[var])
                    result_point[var] = float(val.as_fraction())
                except:
                    result_point[var] = float(str(val))
            
            return {
                'feasible': True,
                'weights': weights_result,
                'result_point': result_point
            }
        else:
            return {'feasible': False}
            
    except Exception as e:
        print(f"DEBUG: LP solving error: {e}", file=sys.stderr)
        return {'feasible': False, 'error': str(e)}

if __name__ == "__main__":
    try:
        print("DEBUG: Script started", file=sys.stderr)
        
        if len(sys.argv) < 2:
            error_result = {'success': False, 'error': 'No arguments provided'}
            print(json.dumps(error_result))
            sys.exit(1)
            
        problem_data = json.loads(sys.argv[1])
        print("DEBUG: JSON parsed successfully", file=sys.stderr)
        
        cycle_vectors = problem_data['cycle_vectors']
        variables = problem_data['variables'] 
        formula = problem_data['limit_avg_formula']
        
        formula = formula.replace('^<', '<').replace('^>', '>')
        
        print(f"DEBUG: Received {len(cycle_vectors)} cycle vectors for variables {variables}", file=sys.stderr)
        print(f"DEBUG: Formula to check: {formula}", file=sys.stderr)
        
        constraints = parse_limit_avg_formula(formula, variables)
        
        result = solve_lp_feasibility(cycle_vectors, variables, constraints)
        result['success'] = True
        
        print(json.dumps(result))
        
    except Exception as e:
        print(f"DEBUG: Main error: {e}", file=sys.stderr)
        error_result = {'success': False, 'error': str(e)}
        print(json.dumps(error_result))
