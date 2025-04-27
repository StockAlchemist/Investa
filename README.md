# Investa Portfolio Dashboard

## Description

Investa is a desktop application built with Python and PySide6 for tracking and analyzing personal investment portfolios. It reads transaction data from a CSV file, fetches current market data and historical prices using the `yfinance` library, calculates portfolio holdings, performance metrics (including Total Return % and Time-Weighted Return - TWR), and visualizes data through tables and charts.

## Features

* Reads transactions from a user-specified CSV file.
* Calculates current portfolio holdings, cost basis, market value, gains/losses.
* Fetches near real-time stock/ETF quotes and FX rates (via Yahoo Finance).
* **Supports manual price overrides** for assets without live data via `manual_prices.json`.
* Calculates historical portfolio value and performance (TWR), using last transaction prices as fallbacks for historical points.
* Displays performance against selected benchmark symbols (e.g., SPY, QQQ).
* Supports multiple accounts and currencies (with automatic FX conversion via USD bridge).
* Filters display by selected accounts.
* Option to show/hide closed positions.
* Visualizations:
  * Summary metrics dashboard.
  * Detailed holdings table with sortable columns and configurable visibility.
  * Pie charts showing portfolio allocation by account and holding.
  * Line charts showing historical absolute value and accumulated TWR vs. benchmarks.
* Configurable display currency.
* Configurable historical graph date range and interval (Daily, Weekly, Monthly).
* Caching for faster loading of market data and historical results.
* Ability to manually add new transactions via a dialog.

## Screenshots (Optional)

*Replace this section with actual screenshots*

`[Screenshot of main window with data loaded]`
*Caption: Main dashboard view showing summary, charts, and table.*

`[Screenshot of historical performance graph]`
*Caption: Historical performance graph comparing portfolio TWR against benchmarks.*

## Setup Instructions

1. **Prerequisites:**
    * Python 3.8 or higher recommended.
    * `pip` (Python package installer).

2. **Clone Repository (if applicable):**

    ```bash
    git clone <your-repository-url>
    cd Investa # Or your project directory name
    ```

3. **Create Virtual Environment:**
    * It's highly recommended to use a virtual environment.

    ```bash
    python -m venv .venv
    ```

    * Activate the environment:
        * Windows: `.\.venv\Scripts\activate`
        * macOS/Linux: `source .venv/bin/activate`

4. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. **Transactions File (`my_transactions.csv`):**
    * Place your transaction data in a CSV file. By default, the app looks for `my_transactions.csv`.
    * You can select a different file using the "Select CSV" button in the app.
    * **Required Columns (exact headers):**
        * `Date (MMM DD, YYYY)` (e.g., Aug 31, 2020)
        * `Transaction Type` (e.g., Buy, Sell, Dividend, Split, Deposit, Withdrawal, Fees)
        * `Stock / ETF Symbol` (Use `$CASH` for cash transactions)
        * `Quantity of Units`
        * `Amount per unit`
        * `Total Amount` (Optional, useful for some dividends)
        * `Fees` (Commissions)
        * `Investment Account` (Name of the account)
        * `Split Ratio (new shares per old share)` (Required only for 'Split' type, e.g., 4 for 4:1)
        * `Note` (Optional)

2. **Manual Prices File (`manual_prices.json`):** (Optional)
    * Create this file in the same directory as `main_gui.py` if you need to provide *current estimated prices* for certain assets.
    * **Purpose:** Used when a live price cannot be fetched from Yahoo Finance (e.g., for private funds, delisted stocks, or symbols added to `YFINANCE_EXCLUDED_SYMBOLS` in `portfolio_logic.py`).
    * **Usage:** When calculating the **current summary**, the application will look in this file first before attempting the "Last Transaction Price" fallback. **This file does NOT affect historical graph calculations.**
    * **Format:** A simple JSON object (key-value pairs):
        * **Key:** The exact symbol string (case-sensitive) as used in your transactions CSV.
        * **Value:** The estimated current price as a number (integer or float). **This price MUST be in the asset's local currency** (e.g., THB for SET funds, USD for US stocks).
    * **Example `manual_prices.json`:**

        ```json
        {
          "PRIVATE_FUND": 18.75,
          "MY_PRIVATE_CO": 55.10
        }
        ```

3. **Configuration File (`gui_config.json`):**
    * This file is automatically created/updated when the application closes to save your settings.
    * Key settings include:
        * `transactions_file`: Path to the last used CSV.
        * `display_currency`: Default currency for UI display (e.g., "USD", "EUR").
        * `account_currency_map`: Defines the local currency for specific accounts (e.g., `"SET": "THB"`). Crucial for multi-currency portfolios.
        * `default_currency`: The base currency used if an account isn't in `account_currency_map` (usually "USD").
        * `selected_accounts`: List of accounts currently selected for display (empty list means all).
        * `graph_start_date`, `graph_end_date`, `graph_interval`, `graph_benchmarks`: Saved graph settings.
        * `column_visibility`: Saves which columns are visible in the table.

4. **API Keys (Optional):**
    * Currently, the FMP API key is loaded from config/environment but not actively used in the Yahoo Finance version.

## Running the Application

1. Ensure your virtual environment is activated.
2. Make sure your transaction CSV (e.g., `my_transactions.csv`), `portfolio_logic.py`, and optionally `manual_prices.json` are in the same directory as `main_gui.py`.
3. Run the main GUI script from the terminal:

    ```bash
    python main_gui.py
    ```

## Dependencies

* PySide6
* pandas
* numpy
* matplotlib
* yfinance
* scipy
* requests (usually installed as a dependency of yfinance)

(See `requirements.txt` for specific versions).
