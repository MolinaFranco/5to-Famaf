# Claude Development Guidelines

## Code Style Requirements

- **ALWAYS run flake8 before committing any Python changes**
- Follow PEP 8 style guidelines strictly
- Maximum line length: 79 characters (flake8 default)
- Use 4 spaces for indentation (no tabs)
- **ALWAYS run flake8 after finishing the changes and fix all issues before continuing**
- Claude must **strictly respect flake8** at all times, after every modification.

## Pre-commit Commands

Before any commit, run:
```bash
flake8 scikit-neuromsi/
```

Fix ALL flake8 issues before proceeding.

## Project Structure

- Main package: scikit-neuromsi/
- All files created for testing the changes maked, experimentation, or temporary purposes: pycloude/
- Never place testing or experimental scripts inside scikit-neuromsi/.
- ssn_inference_numerical_experiments and ssn_inference_optimizer: They are reliable repositories that do not need to be modified and you can extract information and methods to use. **You should ALWAYS respect the style and architecture of scikit-neuromsi** since all our changes will be there.

## Development Workflow

1. Make code changes
2. Check that everything is well documented and explained, with docstrings in English and comments referencing the papers.
3. Run flake8 to check style
4. Fix any style issues
5. Run tests

## Documentation and Comments

- **All docstrings must be complete and written in English.** Everything between ''' or """ must be in English.
- Inline comments using # must be written in Spanish.
- Docstrings must clearly explain the purpose, parameters, return values, and include references to the related papers or source code.

## Justification of Code and Methods

- All code, decisions, and implementations must be justified based on: The papers located in the repository, especially:
        echepaper (Echeveste et al., 2020)
        cupini2017 (Cupini et al., 2017)
- The original repositories already cloned and available in the local directory:
    ssn_inference_numerical_experiments
    ssn_inference_optimizer
- Claude must try to not look for external information on the internet to justify implementations.
- If more information or clarification is needed, it should ask the user directly instead.
- When using or adapting existing code, **always note the origin and reference the corresponding paper or source file.**

## Code preferences

- I prefer that you use descriptive variable names or at least explain them. Many times in papers they use only one letter, but it can be confusing in large quantities to use only one letter.