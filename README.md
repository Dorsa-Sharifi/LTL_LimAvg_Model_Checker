# LTL LimAvg Model Checker



# How to run

To run and test the model checker, follow the steps below:
- Open test.py and find function run_enhanced_test_suite().
- Define your desired Quantitative Kripke Structure in qks.
- Enter your wsl path script in this line: processor = EnhancedLTLimProcessor("/home/otebook/ltl_to_nbw.py", qks).
- Enter your desired LTL Formula in test_categories part.
- Run test.py.

