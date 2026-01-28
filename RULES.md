# Axios Project Rules

## Code Style

1. No emojis anywhere in the codebase
2. No comments in the code
3. No emojis or comments in README or any documentation
4. Clean, self-documenting code with clear variable and function names
5. Follow PEP 8 for Python code

## Git Workflow

1. Make frequent, granular commits
2. Each commit should represent a logical unit of work
3. Commit messages should be clear and descriptive
4. No WIP commits to main branch

## Branching Strategy

1. Treat this as a production-level project
2. All major features and changes must be developed in feature branches
3. Branch naming: feature/description, fix/description, refactor/description
4. Test thoroughly in the feature branch before merging
5. Only merge to main after verification passes
6. Delete feature branches after successful merge
7. Never push broken code to main

## Project Structure

1. All source code in src directory
2. All robot models in models directory
3. All configuration in config directory
4. All training artifacts in outputs directory
5. Virtual environment in venv directory (gitignored)

## Development

1. Always use the project virtual environment
2. Keep requirements.txt updated
3. Test before committing
4. No hardcoded paths
