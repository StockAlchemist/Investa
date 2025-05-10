# Investa Portfolio Dashboard

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Investa is a desktop application designed to help you track, analyze, and visualize your investment portfolio. It loads transaction data from a CSV file, fetches current market prices, and provides a comprehensive overview of your holdings, performance, and historical trends.

## Features

* **Portfolio Summary:** View current market values, cost basis, unrealized/realized gains, dividends, and total returns for your holdings.
* **Detailed Holdings Table:** See a breakdown of each stock, ETF, and cash position with sortable columns and customizable visibility.
* **Account Filtering:** Focus your view on specific investment accounts or see an aggregated overview.
* **Currency Conversion:** Display portfolio values in your preferred currency (USD, THB, EUR, JPY, GBP, etc.).
* **Historical Performance Charts:**
  * Plot accumulated Time-Weighted Return (TWR) against popular benchmarks (e.g., SPY, QQQ).
  * Visualize absolute portfolio value over time.
  * Adjustable date ranges and intervals (Daily, Weekly, Monthly).
* **Periodic Returns Bar Charts:** View portfolio and benchmark returns for annual, monthly, and weekly periods.
* **Portfolio Allocation Pie Charts:** Understand your portfolio's composition by account and by individual holding.
* **Market Data Fetching:** Retrieves near real-time stock quotes, index prices, and FX rates using Yahoo Finance.
* **Data Caching:** Caches fetched market data to speed up subsequent loads and reduce API calls.
* **Transaction Management:**
  * Manually add new transactions.
  * View, edit, and delete existing transactions directly from the application (modifies the source CSV).
* **Manual Price Overrides:** Set manual prices for symbols where API data might be unavailable or incorrect.
* **Fundamental Data Viewer:** Look up and display key fundamental data for stock symbols.
* **Configuration Persistence:** Saves UI settings (file paths, currency, column visibility, etc.) for convenience.
* **CSV Format Help:** In-app guide for the required transaction CSV format.
* **Numba Optimization:** Utilizes Numba for accelerating historical portfolio value calculations.

## Getting Started Tutorial

For a step-by-step guide on how to set up and use Investa, please see our detailed tutorial:

➡️ **[Investa User Tutorial](TUTORIAL.md)**

## Screenshots

*(Placeholder: Add screenshots of the application here to give users a visual idea.)*

* *Main Dashboard View*
* *Historical Performance Graph*
* *Transaction Management Dialog*

## Technology Stack

* **GUI:** PySide6 (Qt for Python)
* **Data Handling & Analysis:** pandas, NumPy
* **Financial Calculations:** SciPy (for IRR/NPV), Numba
* **Market Data:** yfinance
* **Charting:** Matplotlib (embedded in PySide6), mplcursors (for interactive tooltips)
* **Concurrency:** QThreadPool, QRunnable, multiprocessing

## Prerequisites

* Python 3.8 or higher
* pip (Python package installer)

## Installation & Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/StockAlchemist/Investa
    cd Investa # Or your repository name
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # .\venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies:**
    A `requirements.txt` file should be created. If not present, you can install the main packages:

    ```bash
    pip install PySide6 pandas numpy matplotlib yfinance scipy mplcursors requests numba
    ```

    *(It's recommended to generate a `requirements.txt` file for easier dependency management: `pip freeze > requirements.txt`)*

## Configuration

* **Application Settings (`gui_config.json`):**
  * This file is automatically created in your user's application configuration directory (e.g., `~/Library/Application Support/Investa/` on macOS).
  * It stores the path to your last used transactions CSV, display preferences, column visibility, and graph settings.
* **Manual Prices (`manual_prices.json`):**
  * Also stored in the user's application configuration directory.
  * Allows you to define manual price overrides for specific symbols. This can be managed via `Settings > Manual Prices...` in the app.
* **Account Currencies:**
  * Managed via `Settings > Account Currencies...` in the app. This allows you to specify the local currency for each of your investment accounts.
* **Advanced Configuration (`config.py`):**
  * Contains constants for logging levels, cache durations, default benchmark symbols, and internal symbol mappings. Modify this file directly for advanced tweaks.
* **API Keys:**
  * The application primarily uses `yfinance` which generally does not require an API key for public data.
  * An environment variable `FMP_API_KEY` is checked (see `config.py`) but currently not actively used in the core portfolio summary logic.

## Input Data Format (Transactions CSV)

The application requires a CSV file with your transaction history. The expected columns are:

1. `Date (MMM DD, YYYY)` (e.g., *Jan 01, 2023*)
2. `Transaction Type` (e.g., *Buy, Sell, Dividend, Split, Deposit, Withdrawal, Fees*)
3. `Stock / ETF Symbol` (e.g., *AAPL, GOOG*. Use **`$CASH`** for cash-related transactions like deposits/withdrawals into an account, or cash dividends/fees not tied to a specific holding.)
4. `Quantity of Units`
5. `Amount per unit`
6. `Total Amount` (Optional for Buy/Sell if Quantity and Amount per unit are provided)
7. `Fees` (Commissions)
8. `Investment Account` (Name of your brokerage account, e.g., *Brokerage A, IRA*)
9. `Split Ratio (new shares per old share)` (Required only for 'Split' type, e.g., *2* for a 2-for-1 split)
10. `Note` (Optional)

For detailed examples and specific requirements for each transaction type, please refer to the **Help > CSV Format Help...** menu within the application.

## Usage

For a detailed step-by-step guide on using Investa, check out our **[User Tutorial](TUTORIAL.md)**.

Here's a quick overview:

1. **Run the application:**

    ```bash
    python main_gui.py
    ```

2. **Select Transactions File:**
    * On first run, or if the previously used file is not found, you'll be prompted to select your transactions CSV file via `File > Open Transactions File...` or the "Select CSV" button.
    * You can also create a new, empty transactions file via `File > New Transactions File...`.

3. **Refresh Data:**
    * Click the "Refresh All" button (or press F5) to load data from the selected CSV, fetch market prices, and calculate portfolio metrics.
    * Use "Update Accounts" if you only change account filters.
    * Use "Update Graphs" if you only change graph parameters (dates, interval, benchmarks).

4. **Interact with the Dashboard:**
    * Use the controls to change display currency, show/hide closed positions, filter by account, and adjust graph settings.
    * Right-click on the holdings table header to customize visible columns.
    * Right-click on a holding in the table for context menu options like viewing its transaction history or charting its price.

## Building a Native macOS App (Optional)

If you wish to package Investa as a standalone macOS application (`.app` bundle), you can use PyInstaller:

1. **Install PyInstaller:**

    ```bash
    pip install pyinstaller
    ```

2. **Prepare an App Icon:** Create an icon in `.icns` format (e.g., `app_icon.icns`).
3. **Generate a Spec File:**

    ```bash
    pyi-makespec --name Investa --windowed --noconfirm main_gui.py
    ```

4. **Modify `Investa.spec`:**
    * Add necessary `datas` (e.g., `style.qss`, your icon file).
    * Include `hiddenimports` for libraries PyInstaller might miss (especially for pandas, numpy, scipy, matplotlib, PySide6, numba).
    * Configure the `BUNDLE` section for macOS specifics (bundle identifier, `Info.plist` details).
    * Refer to the PyInstaller documentation for detailed spec file options.
5. **Build the App:**

    ```bash
    pyinstaller Investa.spec
    ```

    The bundled app will be in the `dist` folder.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these general steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Create a new Pull Request.

Please ensure your code follows existing style and includes relevant updates to documentation if applicable.

## License

This project is licensed under the MIT License - see the LICENSE file for details (or include the MIT license text directly if no separate file).

SPDX-License-Identifier: MIT

## Author

* **Kit Matan** - <kittiwit@gmail.com>

---

*Disclaimer: This software is for informational and educational purposes only. It is not financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions. Market data is typically provided by Yahoo Finance and may be delayed or contain inaccuracies.*
