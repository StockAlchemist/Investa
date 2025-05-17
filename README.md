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
* **Periodic Returns Bar Charts:** View portfolio and benchmark returns for annual, monthly, and weekly periods with adjustable lookback.
* **Dividend History:** Visualize dividend income over time (Annual, Quarterly, Monthly) and view a detailed dividend transaction table.
* **Portfolio Allocation Pie Charts:**
  * Understand your portfolio's composition by account and by individual holding.
  * View allocation by Asset Type, Sector, Geography, and Industry (requires fundamental data).
* **Market Data Fetching:** Retrieves near real-time stock quotes, index prices, and FX rates using Yahoo Finance.
* **Data Caching:** Caches fetched market data to speed up subsequent loads and reduce API calls.
* **Transaction Management:**
  * Manually add new transactions.
  * View, edit, and delete existing transactions directly from the application (modifies the source CSV).
  * Utility to standardize CSV headers to the application's preferred internal format.
* **Manual Price Overrides:** Set manual prices for symbols where API data might be unavailable or incorrect.
* **Fundamental Data Viewer:** Look up and display key fundamental data for stock symbols.
  * **Financials Tab:** Income Statement, Balance Sheet, Cash Flow statements.
* **Configuration Persistence:** Saves UI settings (file paths, currency, column visibility, account currencies, manual overrides, etc.) for convenience.
* **Account Currency Management:** Assign specific currencies to different investment accounts.
* **Manual Overrides:** Beyond just price, manually override Asset Type, Sector, Geography, and Industry for holdings.
* **Table Filtering:** Live text-based filtering for the main holdings table.
* **CSV Format Help:** In-app guide for the required transaction CSV format.
* **Numba Optimization:** Utilizes Numba for accelerating historical portfolio value calculations.

## Getting Started Tutorial

For a step-by-step guide on how to set up and use Investa, please see our detailed tutorial:

➡️ **[Investa User Tutorial](TUTORIAL.md)**

## Screenshots

![Main Dashboard View](screenshots/Investa_screen_1.png)
![Transaction Log](screenshots/Investa_screen_2.png)
![Asset Allocation](screenshots/Investa_screen_3.png)

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

User-specific configuration files (`gui_config.json`, `manual_prices.json`), cache files, and transaction backups are stored in a standard application data directory. The exact location depends on your operating system:

* **macOS:** `~/Library/Application Support/StockAlchemist/Investa/`
* **Windows:** `C:\Users\<YourUserName>\AppData\Local\StockAlchemist\Investa\` (or potentially in `AppData\Roaming`)
* **Linux:** `~/.local/share/StockAlchemist/Investa/` (or `~/.config/StockAlchemist/Investa/`)

Key files in this directory:

* **`gui_config.json`**: Stores your UI preferences, such as the path to your last used transactions CSV, display currency, selected accounts, graph settings, and column visibility.
* **`manual_prices.json`**: Contains any manual price overrides you've set for specific symbols (managed via `Settings > Manual Prices...` in the app).
* **`csv_backups/` (subfolder)**: Stores timestamped backups of your transactions CSV file, created automatically when you add, edit, or delete transactions through the application.
* Cache files (e.g., for market data) are also stored here to speed up loading times.

* **Account Currencies:**
  * Managed via `Settings > Account Currencies...` in the app. This allows you to specify the local currency for each of your investment accounts.
* **Advanced Configuration (`config.py`):**
  * Contains constants for logging levels, cache durations, default benchmark symbols, and internal symbol mappings. Modify this file directly for advanced tweaks.
* **API Keys:**
  * The application primarily uses `yfinance` which generally does not require an API key for public data.
  * An environment variable `FMP_API_KEY` is checked (see `config.py`) but currently not actively used in the core portfolio summary logic.

## Input Data Format (Transactions CSV)

The application requires a CSV file with your transaction history.
Investa can read CSV files with either verbose, descriptive headers or its preferred cleaned, internal headers.
A utility is provided within the application (`Settings > Standardize CSV Headers...`) to convert your CSV to the cleaned header format, which is recommended for optimal consistency.

**Preferred (Cleaned) CSV Headers:**

1. `Date` (e.g., *Jan 01, 2023* or other common date formats)
2. `Type` (e.g., *Buy, Sell, Dividend, Split, Deposit, Withdrawal, Fees*)
3. `Symbol` (e.g., *AAPL, GOOG*. Use **`$CASH`** for cash-related transactions.)
4. `Quantity`
5. `Price/Share`
6. `Total Amount` (Optional for Buy/Sell if Quantity and Price/Share are provided)
7. `Commission` (Fees)
8. `Account` (Name of your brokerage account, e.g., *Brokerage A, IRA*)
9. `Split Ratio` (Required only for 'Split' type, e.g., *2* for a 2-for-1 split)
10. `Note` (Optional)

**Compatible (Verbose) CSV Headers (will be mapped internally):**

* `Date (MMM DD, YYYY)` maps to `Date`
* `Transaction Type` maps to `Type`
* `Stock / ETF Symbol` maps to `Symbol`
* `Quantity of Units` maps to `Quantity`
* `Amount per unit` maps to `Price/Share`
* `Fees` maps to `Commission`
* `Investment Account` maps to `Account`
* `Split Ratio (new shares per old share)` maps to `Split Ratio`
    *(`Total Amount` and `Note` are typically the same)*

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

We welcome contributions to Investa! If you're interested in helping out, please take a look at our **[Contributing Guidelines](CONTRIBUTING.md)** for more information on how to get started, coding standards, and how to submit your changes.

We appreciate your help in making Investa better!

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details (or include the MIT license text directly if no separate file).

SPDX-License-Identifier: MIT

## Author

* **Google Gemini and Kit Matan** - <kittiwit@gmail.com>

---

*Disclaimer: This software is for informational and educational purposes only. It is not financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions. Market data is typically provided by Yahoo Finance and may be delayed or contain inaccuracies.*
