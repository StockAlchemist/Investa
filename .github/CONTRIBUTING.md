# Contributing to Investa

First off, thank you for considering contributing to Investa! It's people like you that make open source such a great community.

## Where do I go from here?

If you've noticed a bug or have a feature request, [make sure to check our issues page](https://github.com/StockAlchemist/Investa/issues) to see if someone has already reported it. If not, please feel free to open a new issue.

## Fork & create a branch

If you're ready to contribute code:

1. Fork the repository to your own GitHub account.
2. Clone the project to your machine: `git clone https://github.com/YOUR_USERNAME/Investa.git`
3. Create a new branch for your changes: `git checkout -b feature/your-descriptive-feature-name` or `fix/your-bug-fix-name`.
4. Make your changes.

## Code Style and Guidelines

* **Python Code:**
  * Try to follow PEP 8 style guidelines.
  * Add comments to your code where necessary to explain complex logic.
  * If you add new features, please try to include or update relevant documentation or docstrings.
* **Commit Messages:**
  * Write clear and concise commit messages.
  * A good format is a short summary (50 chars or less), followed by a blank line, followed by a more detailed explanation if needed.
  * Reference any relevant issue numbers (e.g., "Fix #123: Improve error handling for CSV parsing").
* **Testing:**
  * If you add new functionality, consider if unit tests are appropriate. (If the project has a testing framework, please add tests for your changes.)

## Submitting a Pull Request

1. Once you're happy with your changes, push your branch to your fork: `git push origin feature/your-descriptive-feature-name`.
2. Go to the Investa repository on GitHub and click "New pull request".
3. Select your branch to compare and create the pull request.
4. Provide a clear title and description for your pull request, explaining the changes you've made and why. Reference any related issues.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct. Please make sure you are familiar with its terms.

## Getting Help

If you have questions or need help with your contribution, feel free to open an issue and tag it as a "question".

Thank you for your contribution!

---

## Development Setup (Quick Recap)

This is a brief reminder, for more details see the main `README.md`.

1. **Clone the repository (if you haven't already from your fork):**

    ```bash
    git clone https://github.com/YOUR_USERNAME/Investa.git
    cd Investa
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # .\venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies:**

    ```bash
    pip install PySide6 pandas numpy matplotlib yfinance scipy mplcursors requests numba
    ```

    *(Or `pip install -r requirements.txt` if one exists and is up-to-date)*
4. **Run the application:**

    ```bash
    python main_gui.py
    ```
