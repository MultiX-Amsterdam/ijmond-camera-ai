# General AI Instructions & Coding Philosophy

Your goal is to write code that accelerates research progress while maintaining high reusability, clarity, and independence of functions. You prioritize practical, testable research code over enterprise-level abstractions.

## Coding Paradigm
- **Functional Programming:** Write self-contained functions. Do NOT rely on or modify state that changes outside the function.
- **No OOP Unless Mandatory:** Avoid Object-Oriented Programming (classes, inheritance) unless absolutely necessary for a specific framework.
- **Pure Functions:** Avoid modifying input parameters. Minimize the use of global variables (only use them for necessary system configuration).
- ***Avoid Code Duplication:** Avoid duplicating codes. Try to reuse exisitng code or functions if possible.

## Python Style & Formatting
- **Language:** Python is the primary language.
- **Style Guide:** Strictly follow PEP 8. Leave exactly two blank lines between functions.
- **Naming Conventions:** Use `lower_snake_case` for all variables, functions, Python files (`.py`), and shell scripts (`.sh`). Never use spaces in file names.
- **Strings:** Always use double quotes (`"like this"`), never single quotes.
- **Docstrings:** Use the `numpydoc` style guide for all functions.
- **Document the Code**: Always write (or update) docstrings at the begining of the script to explain what the script is doing (and also provide example usages if it can be executed directly via terminal). Also, for each function, write (or update) docstrings to explain input and output parameters.

## Testing Philosophy
- **Script-Level Testing:** Do not write unit tests for individual functions. Instead, write test cases for the entire script (e.g., passing input parameters via terminal commands).
- **Manual Verification:** Expect that many tasks (like computer vision data preprocessing) require the user to eyeball outputs. Output intermediate results at a smaller scale for the user to visually verify.

## Environment & Dependencies
- **Virtual Environment:** Assume `conda` is used for environment management.
- **Package Installation:** Always maintain an `install_packages.sh` script.
- **Strict Versioning:** All packages in `install_packages.sh` MUST have explicit version numbers.
- **Command Preference:** Always try `pip install ==version` first. Only fall back to `conda install` if `pip` fails.

## Security
- **NO CREDENTIALS:** NEVER commit, push, or write database passwords, private keys, user tokens, or any other credentials in the code.
