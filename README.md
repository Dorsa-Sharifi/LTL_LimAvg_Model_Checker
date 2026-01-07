# LTLim Model Checker

A multi-platform model checker for Linear Temporal Logic with limit-average constraints (LTLim) that works on **Windows+WSL** and **Native Linux**.

##  Quick Start

### Windows with WSL
```bash
# Clone and enter directory
git clone <repository-url>
cd LTL_LimAvg_Model_Checker

# Setup configuration (press Enter 3 times for defaults)
python main.py --setup

# Verify installation
python main.py --show-platform

# Run your first verification
python main.py "F p"
```

### Native Linux
```bash
# Clone and install dependencies
git clone <repository-url>
cd LTL_LimAvg_Model_Checker

# Setup configuration
python3 main.py --setup

# Run verification
python3 main.py "LimInfAvg(x) > 2.0"
```
## Prerequisites

### Windows + WSL
1. **Windows 10/11** with WSL2 installed
   ```powershell
   # In PowerShell (Admin)
   wsl --install
   ```
2. **Python 3.8+** on Windows
3. **In WSL**: Install required tools
   ```bash
   # Inside WSL terminal
   sudo apt update
   sudo apt install python3 python3-pip
   pip3 install spot  # Spot library for LTL to NBW conversion
   sudo apt install z3  # Z3 theorem prover
   ```

### Native Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv
sudo apt install python3-spot z3

# Install Python dependencies
pip3 install numpy scipy
```

##  Installation & Setup

### 1. Initial Configuration
Run the interactive setup wizard:
```bash
python main.py --setup
```

You'll be prompted for three paths:
- **Spot script**: Path to `ltl_to_nbw.py` (default: `/home/otebook/ltl_to_nbw.py`)
- **Z3 LP script**: Path to `z3_lp_solver.py` (default: `/home/otebook/z3_lp_solver.py`)
- **Z3 Python**: Python executable for Z3 (default: your WSL Python path)

**For Windows+WSL users**: Just press **Enter** three times to accept defaults.

### 2. Verify Installation
```bash
# Check platform detection
python main.py --show-platform

# Expected output for Windows+WSL:
# Platform: Windows with WSL
# System: windows
# Tool Strategy: wsl_bridge
# Python: python
```

### 3. Test with Example
```bash
# Simple LTL formula
python main.py "F p"

# With limit-average constraint
python main.py "LimInfAvg(x) > 1.5"
```

##  Usage Examples

### Basic Verification
```bash
# Single formula on default system
python main.py "G(p -> F q)"

# Quiet mode (less output)
python main.py --quiet "F p"

# With custom system specification
python main.py --qks-file system.json "LimInfAvg(x) > 2.0"

# Batch processing
python main.py --batch formulas.txt
```

### Create a Custom System
Create a JSON file `my_system.json`:
```json
{
    "states": ["s0", "s1", "s2"],
    "init_state": "s0",
    "edges": [
        ["s0", "s1"],
        ["s1", "s2"],
        ["s2", "s0"]
    ],
    "boolean_vars": ["p", "q"],
    "logical_formulas": {
        "s0": ["p"],
        "s1": ["q"],
        "s2": ["p", "q"]
    },
    "numeric_values": {
        "s0": {"x": 1.0, "y": 2.0},
        "s1": {"x": 2.0, "y": 3.0},
        "s2": {"x": 3.0, "y": 1.0}
    }
}
```

Then verify:
```bash
python main.py --qks-file my_system.json "F(p ∧ q)"
```

## LTLLim Syntax Reference

### Temporal Operators
| Operator | Meaning | Example |
|----------|---------|---------|
| `F φ` | Eventually φ | `F p` |
| `G φ` | Globally φ | `G(p → q)` |
| `X φ` | Next φ | `X p` |
| `φ U ψ` | φ Until ψ | `p U q` |
| `φ R ψ` | φ Release ψ | `p R q` |

### Boolean Operators
| Operator | Alternatives | Example |
|----------|--------------|---------|
| Negation | `¬φ` or `!φ` | `¬p` |
| Conjunction | `φ ∧ ψ` or `φ && ψ` | `p ∧ q` |
| Disjunction | `φ ∨ ψ` or `φ \|\| ψ` | `p ∨ q` |
| Implication | `φ → ψ` or `φ -> ψ` | `p → q` |
| Equivalence | `φ ↔ ψ` or `φ <-> ψ` | `p ↔ q` |

### Limit-Average Constraints
| Constraint | Meaning |
|------------|---------|
| `LimInfAvg(x) ≥ c` | Limit inferior average of x ≥ c |
| `LimSupAvg(x) ≤ c` | Limit superior average of x ≤ c |
| `LimInfAvg(x) > c` | Strict inequality |
| `LimSupAvg(x) < c` | Strict inequality |

### Formula Examples
```bash
# Simple LTL
python main.py "F p"
python main.py "G(p → F q)"
python main.py "p U (q ∧ r)"

# With limit averages
python main.py "LimInfAvg(x) > 2.0"
python main.py "F p ∧ LimSupAvg(y) ≤ 3.0"
python main.py "G(p → (F q ∧ LimInfAvg(z) ≥ 1.0))"

# Complex combinations
python main.py "(F p) ∧ (G q) ∧ (LimInfAvg(x) > 1.5)"
```

### Platform Detection
The system automatically detects your environment:
- **Windows with WSL available** → Uses `wsl_bridge` strategy
- **Native Linux** → Uses `native_linux` strategy
- **WSL (inside WSL)** → Uses `native_linux` strategy

### Execution Strategies
- **`wsl_bridge`**: Calls Linux tools via `wsl` command from Windows
- **`native_linux`**: Direct execution on Linux/WSL

##  Configuration

### Configuration File
Generated at `model_checker_config.json`:
```json
{
    "platform": "Windows with WSL",
    "paths": {
        "spot_script": "/home/otebook/ltl_to_nbw.py",
        "z3_lp_script": "/home/otebook/z3_lp_solver.py",
        "z3_python": "/home/otebook/.local/share/pipx/venvs/z3-solver/bin/python"
    }
}
```

### Updating Configuration
```bash
# Run setup wizard again
python main.py --setup

# Or edit manually
nano model_checker_config.json  # Linux/WSL
notepad model_checker_config.json  # Windows
```

## Testing

### Run Test Suite
```bash
# Create test formulas
cat > test_formulas.txt << EOF
F p
G(p -> F q)
LimInfAvg(x) > 1.0
EOF

# Batch test
python main.py --batch test_formulas.txt

# Test with custom system
python main.py --qks-file test_system.json "F p"
```

### Diagnostic Commands
```bash
# Show platform info
python main.py --show-platform

# Check configuration
python -c "from config import get_config; c=get_config(); c.print_summary()"

# Test WSL connectivity (Windows only)
wsl echo "WSL is working"
```


## Understanding Output

### Successful Verification
```
-RESULT: SATISFIABLE
```
or
```
-RESULT: UNSATISFIABLE
```

### Output Sections
1. **Platform Information**: OS detection results
2. **Configuration Summary**: Loaded paths and validation
3. **LTL Parsing**: Formula parsing and transformation
4. **NBW Conversion**: LTL to nondeterministic Büchi automaton
5. **Product Construction**: QKS × NBW product automaton
6. **MSCC Analysis**: Maximal strongly connected components
7. **Z3 Checking**: Feasibility checking with Z3
8. **Final Result**: SATISFIABLE or UNSATISFIABLE

## Algorithm Overview

The model checker implements:

1. **LTLim Parsing**: Parse and negate input formula
2. **Formula Transformation**: Convert to disjunctive normal form
3. **NBW Construction**: Convert LTL to nondeterministic Büchi automaton using Spot
4. **Product Automaton**: Construct QKS × NBW product
5. **MSCC Decomposition**: Find maximal strongly connected components
6. **Fairness Checking**: Identify fair MSCCs (containing accepting states)
7. **Cycle Analysis**: Compute limit-average values for cycles
8. **Convex Hull**: Compute value regions
9. **Feasibility Checking**: Use Z3 to check limit-average constraints



