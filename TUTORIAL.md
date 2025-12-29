# Welcome to Investa v1.0! A Quick Tutorial

Investa is your personal desktop assistant for keeping a close eye on your investment portfolio. It uses a local SQLite database to store your transaction history, fetches the latest market data, and then presents you with a clear, detailed picture of how your investments are doing.

Let's dive in!

## Part 1: Getting Set Up

Before you can start crunching numbers, there are a couple of preliminary steps:

1. **Installation (If you haven't already):**
    * Make sure you have Python (3.8 or newer) on your system.
    * Clone the Investa project from its GitHub repository.
    * It's a good idea to set up a virtual environment for Python projects. In your terminal, navigate to the cloned project folder and run:

        ```bash
        python3 -m venv venv
        source venv/bin/activate # For macOS/Linux
        # .\venv\Scripts\activate # For Windows
        ```

    * Install the necessary Python packages. If there's a `requirements.txt` file, you can use:

        ```bash
        pip install -r requirements.txt
        ```

        Otherwise, install them manually (core dependencies):

        ```bash
        pip install PySide6 pandas numpy matplotlib yfinance scipy mplcursors requests numba
        ```

2. **Understanding Data Storage: The SQLite Database**
    Investa uses a local SQLite database file (typically named `investa_transactions.db`) to store all your transaction data. This file is the heart of your Investa setup.
    * **Location:** This database file, along with configuration files (`gui_config.json`, `manual_overrides.json`), cache, and backups, is stored in a standard application data directory. The exact path depends on your operating system:
        * **macOS:** `~/Library/Application Support/StockAlchemist/Investa/`
        * **Windows:** `C:\Users\<YourUserName>\AppData\Local\StockAlchemist\Investa\` (or `AppData\Roaming\StockAlchemist\Investa\`)
        * **Linux:** `~/.local/share/StockAlchemist/Investa/` (or `~/.config/StockAlchemist/Investa/`)
    * You will typically create or open this database file when you first run Investa.

3. **Preparing Your Transaction Data for Import (Optional CSV):**
    If you have existing transaction data in a CSV file, you can import it into the Investa database. Investa is flexible with CSV headers but recommends a "cleaned" format for clarity. An in-app utility (**File > Import Transactions from CSV...** then **Standardize Headers**) can help map your CSV columns.

    **Preferred (Cleaned) CSV Headers for Import:**
    1. `Date`: e.g., *Jan 01, 2023* (common date formats are supported)
    2. `Type`: *Buy, Sell, Dividend, Split, Deposit, Withdrawal, Fees*
    3. `Type`: *Buy, Sell, Dividend, Split, Deposit, Withdrawal, Fees, Transfer*
    3. `Symbol`: e.g., *AAPL, VTI*. Use the special symbol **`$CASH`** for all cash transactions.
    4. `Quantity`
    5. `Price/Share`
    6. `Total Amount`: (Optional for Buy/Sell if Quantity and Price/Share are provided). For dividends, this is the total dividend amount.
    7. `Commission`: Any transaction fees.
    8. `Account`: Name of your brokerage account (e.g., *Brokerage A, Roth IRA*).
    9. `Split Ratio`: Only for 'Split' type (e.g., *2* for a 2-for-1 split).
    10. `Note`: Optional notes.

    **Compatible (Verbose) CSV Headers (mapped internally during import):**
    * `Date (MMM DD, YYYY)` -> `Date`
    * `Transaction Type` -> `Type`
    * `Stock / ETF Symbol` -> `Symbol`
    * `Quantity of Units` -> `Quantity`
    * `Amount per unit` -> `Price/Share`
    * `Fees` -> `Commission`
    * `Investment Account` -> `Account`
    * `Split Ratio (new shares per old share)` -> `Split Ratio`

### Understanding and Using `$CASH` Transactions

The special cash symbol **`$CASH`** is crucial for accurately tracking your portfolio's value, cash flows, and performance, especially the Time-Weighted Return (TWR).

* **What it Represents:** The `$CASH` symbol signifies the cash component within an investment account. Its currency is determined by the currency assigned to that account (see **Settings > Account Currencies...**). Its price is always `1.00` in its respective currency.
* **When and How to Use Cash Symbols (Examples for CSV Import or Manual Entry):**

  * **Deposits (External Inflow):** Adding funds to your brokerage account.
    * `Date`: 2023-01-15
    * `Type`: Deposit
    * `Symbol`: $CASH
    * `Quantity`: 1000
    * `Price/Share`: 1
    * `Total Amount`: 1000
    * `Account`: Brokerage A
    * **TWR Impact:** This is an **external cash inflow**, increasing the investment base. TWR calculation will account for this new capital.

  * **Withdrawals (External Outflow):** Taking funds out of your brokerage account.
    * `Date`: 2023-02-20
    * `Type`: Withdrawal
    * `Symbol`: $CASH
    * `Quantity`: 500 (or -500, `Total Amount` will be negative)
    * `Price/Share`: 1
    * `Total Amount`: -500 (or 500 if Quantity is negative)
    * `Account`: Brokerage A
    * **TWR Impact:** This is an **external cash outflow**, decreasing the investment base. TWR calculation will account for this reduction of capital.

  * **Buying a Stock (Internal Conversion):** Using cash from your account to buy shares.
    * This is typically a two-part conceptual movement if you track cash with high fidelity, though Investa often handles the cash deduction implicitly if your `Buy` transaction for a stock has a `Total Amount`. For explicit cash tracking during CSV import or for clarity:
            1. **(Optional Explicit Cash Reduction)** A `$CASH` `Sell` or `Withdrawal` (less common for this specific purpose).
                *`Date`: 2023-03-01
                * `Type`: Sell (or Withdrawal, conceptually)
                *`Symbol`: $CASH
                * `Quantity`: 1000 (amount used for stock purchase)
                *`Price/Share`: 1
                * `Total Amount`: -1000
                *`Account`: Brokerage A
            2. **The Actual Stock Purchase:**
                * `Date`: 2023-03-01
                *`Type`: Buy
                * `Symbol`: AAPL
                *`Quantity`: 10
                * `Price/Share`: 100
                *`Total Amount`: 1000
                * `Account`: Brokerage A
    * **TWR Impact:** The act of buying a stock itself (converting cash to stock) is an **internal asset conversion**, not an external cash flow. The total portfolio value remains momentarily unchanged. TWR measures the performance *after* this conversion. If you explicitly log a `$CASH` `Sell` for this, it's also internal.

  * **Selling a Stock (Internal Conversion):** Receiving cash in your account from selling shares.
    * Similar to buying, this is often handled implicitly. For explicit cash tracking:
            1. **The Actual Stock Sale:**
                *`Date`: 2023-04-10
                * `Type`: Sell
                *`Symbol`: AAPL
                * `Quantity`: 5
                *`Price/Share`: 120
                * `Total Amount`: 600
                *`Account`: Brokerage A
            2. **(Optional Explicit Cash Increase)** A `$CASH` `Buy` or `Deposit`.
                * `Date`: 2023-04-10
                *`Type`: Buy (or Deposit, conceptually)
                * `Symbol`: $CASH
                *`Quantity`: 600 (proceeds from sale)
                * `Price/Share`: 1
                *`Total Amount`: 600
                * `Account`: Brokerage A
    * **TWR Impact:** Selling a stock is an **internal asset conversion**. Cash increases, stock holding decreases. Not an external flow.

  * **Dividends Received in Cash (Two-Step Process):**
    * To accurately track both the return from your stock and the corresponding increase in your cash balance, you should record **two transactions**.

    **Step 1: Record the dividend gain from the stock.** This attributes the return to the correct asset.
    * `Date`: 2023-05-15
    * `Type`: Dividend
    * `Symbol`: MSFT (the stock that paid the dividend)
    * `Total Amount`: 50 (the cash amount of the dividend)
    * `Account`: Brokerage A

    **Step 2: Record the cash increase in your account.** This reflects the cash landing in your brokerage account.
    * `Date`: 2023-05-15
    * `Type`: Buy
    * `Symbol`: $CASH
    * `Quantity`: 50
    * `Price/Share`: 1
    * `Account`: Brokerage A

    * **Why two steps?**
      * The `Dividend` transaction on `MSFT` ensures that the $50 is correctly included in MSFT's "Total Gain" and "Total Return %".
      * The `Buy` transaction on `$CASH` increases your cash balance. Using `Buy` (instead of `Deposit`) correctly tells the system this is an *internal* cash movement (a return from an existing asset), not new external capital. This is crucial for accurate Time-Weighted Return (TWR) calculation.

  * **Transferring Assets Between Accounts (ACATS):**
    * Use the `Transfer` type to move assets (stocks, ETFs, or cash) from one of your accounts to another. This is an internal movement that preserves the cost basis of the asset.

    **Example: Transferring 10 shares of VTI from Brokerage A to Roth IRA.**
    * `Date`: 2023-07-01
    * `Type`: Transfer
    * `Symbol`: VTI
    * `Quantity`: 10
    * `Account`: Brokerage A  *(This is the 'From' account)*
    * `Note`: To: Roth IRA  *(The 'To' account must be specified in the note like this)*

    * **How it works:** Investa will decrease the holding of VTI in "Brokerage A" and increase it in "Roth IRA", transferring the proportional cost basis along with the shares.
    * **TWR Impact:** This is an **internal asset conversion** between accounts. It is not an external cash flow and does not impact the TWR of the overall portfolio.
    * **Important:** For CSV imports, the "To" account must be specified in the `Note` column with the format `To: <Account Name>`. When adding manually in the app, dedicated "From" and "To" fields will appear.

  * **Fees Paid From Cash:** For general account fees not tied to a specific trade.
    * `Date`: 2023-06-01
    * `Type`: Fees
    * `Symbol`: $CASH
    * `Quantity`: 10
    * `Price/Share`: 1
    * `Total Amount`: -10 (or 10 if Quantity is negative)
    * `Account`: Brokerage A
    * **TWR Impact:** Fees paid from cash are like withdrawals for TWR purposes if they are external to the core investment activity (e.g. advisory fees). If they are transaction fees, they are part of the cost of the transaction. The `Fees` type with `$CASH` implies an expense that reduces the account's cash and is typically treated as a negative return or an outflow.

* **Key for TWR:** Accurately logging `Deposit` and `Withdrawal` transactions for `$CASH` is paramount for correct TWR calculation, as these define the external cash flows against which investment performance is measured. Internal conversions (buying/selling assets) do not count as external flows.

* **Tip:** For detailed examples and specific requirements for each transaction type, once the app is running, check out the **Help > CSV Format Help...** menu. This is especially useful before importing a CSV.

## Part 2: Your First Launch and Managing Data Files

1. **Run Investa:**
    Open your terminal, ensure your virtual environment is active (if used), navigate to the Investa project directory, and run:

    ```bash
    python src/main_gui.py
    ```

2. **Database Initialization (First Run):**
    * **Welcome Prompt:** On your very first launch, Investa will display a welcome message and prompt you to either:
        * **Create a New Database:** This is the recommended option if you're new to Investa or want to start fresh. It will create an empty `investa_transactions.db` file in the default application data directory (see Part 1 for locations) or a location you choose.
        * **Open an Existing Database:** If you have an existing Investa database file (`.db`) from a previous installation or a backup, use this option.
        * **(Migrate from CSV - Deprecated):** Older versions focused on CSVs. If Investa detects an old `gui_config.json` pointing to a CSV, it *might* offer to import it. However, the primary workflow is now database-centric. It's better to create a new database and then import your CSV.

3. **Managing Your Data Files (File Menu):**
    The **File** menu is your central hub for managing data:

    * **File > New Database File...**
        * Use this to create a brand-new, empty SQLite database (`.db` extension).
        * You'll be asked where to save this file. The default application data directory is a good choice, but you can place it elsewhere (e.g., a synced cloud folder, though ensure it's not simultaneously accessed by multiple Investa instances).
        * Once created, this new database becomes the active one.

    * **File > Open Database File...**
        * Use this to open an existing Investa SQLite database file (`.db`).
        * This is how you switch between different portfolio databases if you maintain more than one, or if you've moved your database file.
        * The path to the last successfully opened database is remembered by Investa.

    * **File > Import Transactions from CSV...**
        * This powerful utility allows you to populate your currently open SQLite database with transactions from a CSV file.
        * A dialog will appear:
            1. **Select CSV File:** Choose the CSV file containing your transactions.
            2. **Header Standardization (Important):** You'll see a preview of your CSV data and options to map your CSV columns to Investa's expected fields (`Date`, `Type`, `Symbol`, etc.). Use the dropdowns above each column in the preview to match your CSV's headers to Investa's internal names.
            3. **Standardize Headers Button:** Click this after mapping. It converts your CSV data to the preferred internal format for import.
            4. **Import Button:** Once headers are standardized, click this to import the transactions into the active database.
            5. **Backup Original CSV:** You'll usually be prompted to back up the original CSV file. It's a good idea to do so.
        * After import, it's wise to check the **Transactions Log** tab to ensure data appears as expected.

4. **Refresh and See the Magic!**
    * Once your database is set up (and optionally populated via CSV import) and you've added any new transactions directly (see Part 6), click the main **"Refresh All"** button (or press F5).
    * Investa will:
        * Load all transactions from the active SQLite database.
        * Fetch the latest market prices for your holdings using Yahoo Finance.
        * Calculate all portfolio metrics, historical performance, and generate charts.
        * Display everything in the dashboard!

## Part 3: Exploring the Main Dashboard

Once your data is loaded (from the database) and refreshed, the main dashboard comes alive. Here's a quick tour:

* **Portfolio Summary (Top Section):**
    This area provides a high-level snapshot of your entire portfolio (or the accounts selected in the filter):
  * `Net Value`: Total current market value of all your holdings.
  * `Day's G/L`: Gain or loss for the current trading day.
  * `Total G/L`: Overall profit or loss, combining realized and unrealized gains/losses and dividends.
  * `Realized G/L`: Profits locked in from sales.
  * `Unrealized G/L`: Profits or losses on paper for assets you still hold.
  * `Dividends`: Total dividends received.
  * `Fees`: Total fees paid.
  * `Cash Balance`: Total cash held across the selected accounts (derived from `$CASH` transactions).
  * `Total Ret %`: Your portfolio's total return percentage.
  * `Ann. TWR %`: Annualized Time-Weighted Return percentage, a key measure of investment performance.

* **Controls (Toolbar and Buttons below Summary):**
  * `Display Currency`: A dropdown to select your preferred currency (e.g., USD, EUR, JPY) for viewing all monetary values. Add more currencies via **Settings > Choose Currencies...**.
  * `Show Closed Pos.`: Checkbox to include or exclude assets you've completely sold off from the Holdings Table.
  * `Accounts`: Button to open a dialog where you can select which specific investment accounts (defined in your transactions) to include in the dashboard view. Click "Update Accounts" (or "Refresh All") after changing.
  * `Refresh All (F5)`: The main button to reload all data from the database, fetch fresh market prices, and recalculate everything.
  * `Update Accounts`: A quicker refresh if you've only changed the account filter.
  * A dedicated section for graph controls with date range selectors, a benchmark selection button, and an "Update Graphs" button. A "Presets..." dropdown menu is available to quickly set common time ranges (`1W`, `MTD`, `YTD`, `1Y`, `All`, etc.).
  * `Update Graphs`: A quicker refresh if you've only changed graph parameters (like date ranges or benchmarks).

* **Holdings Table (Main Area):**
    This is the heart of the dashboard, listing all your individual assets (stocks, ETFs, `$CASH`) within the currently selected accounts.
  * **Columns:** Displays comprehensive information for each holding, such as Symbol, Quantity, Current Price, Market Value, Cost Basis, Unrealized Gain/Loss (value and %), Realized Gain/Loss, Dividends, Fees, Return %, TWR %, and more.
  * **Sorting:** Click any column header to sort the table by that column. Click again to reverse the sort order.
  * **Customize Columns:** Right-click on any column header to open a context menu that allows you to show or hide specific columns, tailoring the table to your preferences.
  * **Live Filtering:**
    * "Filter: Symbol contains..." text box: Type here to quickly filter the table for symbols matching your input.
    * "Filter: Account contains..." text box: Type here to filter by account name.
  * **Context Menu (Right-Click on a Holding):** Right-clicking a specific row (holding) in the table provides quick actions:
    * `View Transactions for [Symbol]`: Opens a dialog showing all transactions for that specific symbol.
    * `View Fundamentals for [Symbol]`: Opens the Fundamental Data Viewer for that symbol.
    * `Chart Price for [Symbol]`: (If charting feature for individual symbols is enabled) Plots the historical price of the selected symbol.

* **Reordering Columns:** You can customize the layout of the scrollable part of the table. Simply click and drag a column header to a new position. Your custom column order and widths will be saved and restored the next time you open Investa.

## Part 4: Analyzing Your Performance with Charts

Investa's charting capabilities are grouped into tabs, providing rich visualizations of your portfolio's performance and composition. Remember to click "Update Graphs" if you change date ranges, intervals, or benchmark selections on the "Performance & Summary" tab.

### "Performance & Summary" Tab

This is your primary tab for performance analysis:

* **Historical Performance Line Graphs (Top Left):**
  * **Accumulated Gain (TWR %):** This chart is crucial. It shows your portfolio's Time-Weighted Return percentage over your chosen `Date Range` and `Interval` (Daily, Weekly, Monthly). TWR measures performance by neutralizing the effects of external cash flows (deposits/withdrawals), making it ideal for comparing your investment strategy against `Benchmark(s)` like SPY (S&P 500) or QQQ (NASDAQ 100). You can select multiple benchmarks to overlay.
  * **Absolute Portfolio Value:** This chart displays the total market value of your portfolio in your selected `Display Currency` over the same `Date Range` and `Interval`. It gives you a clear view of your portfolio's growth in monetary terms.

* **Periodic Returns Bar Charts (Right Side):**
    These charts break down percentage returns for discrete periods, allowing you to see performance trends:
  * **Annual Returns:** Shows year-by-year TWR for your portfolio and the selected primary benchmark. Use the "Periods:" spinbox to control how many past years are displayed.
  * **Monthly Returns:** Shows month-by-month TWR.
  * **Weekly Returns:** Shows week-by-week TWR.

* **Portfolio Allocation Pie Charts (Bottom Left):**
    These provide a quick visual breakdown of your portfolio's current composition based on the accounts selected in the main filter:
  * **Value by Account:** Illustrates how your total portfolio value is distributed across your different investment accounts (e.g., Brokerage A, IRA).
  * **Value by Holding:** Shows the relative weight (percentage of total portfolio value) of each individual stock, ETF, and your `$CASH` balance.

### Other Chart Tabs (Covered in Later Sections)

* **"Dividend History" Tab:** Focuses on visualizing dividend income over time.
* **"Asset Allocation" Tab:** Provides pie charts for allocation by Asset Type, Sector, Geography, and Industry.

## Part 5: Deep Dive into Dividend History

The **"Dividend History"** tab is dedicated to tracking and visualizing your dividend income, offering both aggregated views and detailed transaction lists.

1. **Accessing the Tab:** Click on the "Dividend History" tab within the main interface.
2. **Account Filtering:** The dividend data presented is filtered by the accounts selected in the main "Accounts" filter (top control bar). This ensures you see dividend information relevant to your current view (e.g., "All Accounts" or a specific brokerage).
3. **Controls for Visualization:**
    * **Aggregate by:** This dropdown menu allows you to group your dividend income into:
        * `Annual`: Shows total dividends received per year.
        * `Quarterly`: Shows total dividends received per quarter.
        * `Monthly`: Shows total dividends received per month.
        This selection drives the bar chart and the summary table.
    * **Periods to Show:** This spinbox lets you define how many of the chosen aggregation periods (e.g., the last 10 years, the last 12 quarters) are displayed in the chart and summary table.
    * **Update Chart Button:** After changing "Aggregate by" or "Periods to Show," click this button to refresh the dividend visualizations and tables.

4. **Dividend Bar Chart:**
    * This chart visually represents the total dividend amounts for each aggregated period you've selected (e.g., a bar for each year showing total annual dividends).
    * The Y-axis displays the total dividend amount in your globally selected `Display Currency`.
    * The X-axis represents the periods (years, quarters, or months).
    * Hovering over a bar often shows the exact amount for that period.

5. **Dividend Summary Table:**
    * Located typically below the bar chart, this table provides the same aggregated data in a numerical format.
    * It has columns like "Period" (e.g., "2023", "2023-Q1") and "Total Dividends" (in the `Display Currency`).

6. **Detailed Dividend Transaction History Table:**
    * This comprehensive table, usually found to the right or below the summary, lists every individual dividend transaction loaded from your database that falls within the selected account scope.
    * Key columns include:
        * `Date`: Date the dividend was received.
        * `Symbol`: The stock or ETF that paid the dividend.
        * `Account`: The account that received the dividend.
        * `LocalCurrency`: The currency in which the dividend was originally paid (asset's local currency).
        * `DividendAmountLocal`: The dividend amount in that local currency.
        * `FXRateUsed`: The foreign exchange rate applied if the local currency differs from your display currency.
        * `DividendAmountDisplayCurrency`: The dividend amount converted to your chosen global `Display Currency`.
    * This table is sortable by clicking on its column headers, allowing you to easily find specific transactions or view trends.

This tab provides a clear and detailed overview of your dividend earnings, helping you understand this important component of your investment returns.

## Part 6: Managing Your Transactions Directly in the Database

With Investa, your transaction data lives in the SQLite database. You have full control to add, edit, and delete transactions directly within the application. Changes are saved immediately to the database file.

### "Transactions Log" Tab

Before making changes, it's often helpful to review your existing data. The **"Transactions Log"** tab offers a comprehensive view:

* It displays two main tables:
  * **Stock/ETF Transactions Table:** Lists all transactions for your actual investment assets (shares, ETFs, etc.).
  * **$CASH Transactions Table:** Specifically lists all transactions related to the `$CASH` symbol (deposits, withdrawals, cash movements).
* Both tables are sortable by clicking on their column headers, making it easy to find specific entries.
* This tab provides a read-only view, perfect for reviewing data before or after making modifications.

### Adding, Editing, and Deleting Transactions

All modifications to your transaction history are done via the **Transactions** menu or corresponding toolbar buttons:

1. **Adding a New Transaction:**
    * Go to **Transactions > Add Transaction...** (or click the "Add Tx" button on the toolbar).
    * A dialog window will appear, allowing you to enter all the details for a new transaction:
        * `Date`, `Type` (Buy, Sell, Dividend, Transfer, etc.), `Symbol`, `Quantity`, `Price/Share`, `Total Amount`, `Commission`, `Account` (or From/To Accounts for transfers), `Split Ratio` (if applicable), and `Note`.
    * Click "Save Transaction" to add this new record directly to your SQLite database.

2. **Managing Existing Transactions (Edit/Delete):**
    * Go to **Transactions > Manage Transactions...** (or click the "Manage Tx" button on the toolbar).
    * This opens the "Manage Transactions" dialog, which is your primary interface for editing or deleting records.
    * **Viewing Transactions:** The dialog displays a table of all transactions currently in your database. You can filter this table by `Symbol` or `Account` using the input fields at the top to quickly find the transaction(s) you're interested in.
    * **Editing a Transaction:**
        1. Select the transaction row in the table that you wish to modify.
        2. Click the "Edit Selected" button.
        3. A dialog, pre-filled with the selected transaction's data, will appear.
        4. Make your necessary changes and click "Save Changes". The record in the database will be updated.
    * **Deleting a Transaction:**
        1. Select the transaction row in the table you want to remove.
        2. Click the "Delete Selected" button.
        3. You'll be asked for confirmation. If you confirm, the transaction will be permanently removed from the database.
            * **Caution:** Deleting transactions is a permanent action in the database. While Investa *may* still create CSV backups during certain operations (like CSV import/export), the primary record is in the database. Always be sure before deleting. It's good practice to occasionally back up your `investa_transactions.db` file itself.

* **Impact of Changes:** After adding, editing, or deleting any transactions, always click the **"Refresh All"** button on the main dashboard to ensure all calculations and views are updated based on the modified data.

This direct database management provides a seamless and integrated way to keep your portfolio records accurate and up-to-date.

## Part 7: Asset Allocation Insights

The **"Asset Allocation"** tab offers valuable pie charts that visually break down your portfolio's diversification across several key dimensions. These charts help you understand your exposure and concentration in different areas. The data primarily comes from the fundamental information fetched for your holdings via Yahoo Finance, which can be supplemented or corrected with manual overrides.

* **Allocation by Asset Type:**
  * Displays a pie chart categorizing your portfolio by asset classes such as "Stock", "ETF", "Cash", "Mutual Fund", "Currency", etc.
  * This classification is typically derived from the `quoteType` provided by Yahoo Finance for each symbol.
  * `$CASH` holdings are explicitly shown as "Cash".

* **Allocation by Sector:**
  * Shows a pie chart of your portfolio's distribution across various market sectors (e.g., "Technology", "Healthcare", "Financial Services", "Consumer Cyclical").
  * This relies on the "Sector" information fetched for your individual stock and ETF holdings.

* **Allocation by Geography:**
  * Visualizes the geographical spread of your investments (e.g., "United States", "Canada", "United Kingdom", "India").
  * This uses the "Country" information associated with your holdings. For ETFs, this often reflects the domicile of the ETF itself, though the underlying assets might be global.

* **Allocation by Industry:**
  * Provides a more granular breakdown than Sector, showing a pie chart of your portfolio's allocation across specific industries (e.g., "Software - Infrastructure", "Banks - Regional", "Auto Manufacturers").
  * This relies on the "Industry" information fetched for your holdings.

**Important Notes on Allocation Data:**

* **Data Source:** The accuracy and completeness of these charts depend heavily on the fundamental data available for your holdings through Yahoo Finance.
* **Missing Data:** If Yahoo Finance does not provide data for a specific category (like Sector or Industry) for some of your holdings, those holdings might be grouped under an "Unknown" or "N/A" slice in the pie chart, or they might not contribute to that specific chart if the information is entirely absent.
* **Manual Overrides:** You have the power to correct or specify these classifications. Use **Settings > Symbol Settings...** to manually set the `Asset Type`, `Sector`, `Geography`, and `Industry` for any symbol. These overrides are stored in `manual_overrides.json` and will be used by the allocation charts. This is particularly useful for:
  * Assets not well-covered by Yahoo Finance.
  * Correcting data that you believe is misclassified by the API.
  * Defining custom classifications that suit your analysis style.
* **Account Filtering:** Like other dashboard elements, these allocation charts respect the currently selected account filter.

By regularly reviewing these charts and ensuring your data (including manual overrides) is accurate, you can gain significant insights into your investment strategy and risk exposures.

## Part 8: Understanding Capital Gains

The **"Capital Gains"** tab provides a focused view of the profits or losses you've realized from selling assets (stocks, ETFs). This is distinct from unrealized gains/losses on assets you still hold.

1. **Accessing the Tab:** Click on the "Capital Gains" tab in the main interface.

2. **Account Filtering:** Similar to other analytical tabs, the capital gains data displayed is filtered by the accounts selected in the main "Accounts" filter on the top control bar. This allows you to see realized gains specific to certain accounts or an aggregated view.

3. **Controls for Visualization:**
    * **Aggregate by:** This dropdown menu allows you to group your realized capital gains into:
        * `Annual`: Shows total realized gains/losses per year.
        * `Quarterly`: Shows total realized gains/losses per quarter.
        This selection drives the bar chart and the summary table below it.
    * **Periods to Show:** This spinbox lets you define how many of the chosen aggregation periods (e.g., the last 10 years, the last 8 quarters) are displayed in the chart and summary table.
    * *(Note: Changes to these controls will automatically update the chart and summary table.)*

4. **Capital Gains Bar Chart:**
    * This chart visually represents the total realized capital gains or losses for each aggregated period you've selected.
    * The Y-axis displays the gain/loss amount in your globally selected `Display Currency`.
    * The X-axis represents the periods (years or quarters).
    * Bars are color-coded: typically green for gains and red for losses.
    * Hovering over a bar often shows the exact amount for that period.

5. **Summary of Plotted Gains Table:**
    * Located directly below the bar chart, this table provides a numerical breakdown of the data shown in the chart.
    * It typically has columns like "Period" (e.g., "2023", "2023-Q2") and "Realized Gain/Loss (Display Currency)".

6. **Detailed Capital Gains History Table:**
    * This comprehensive table, found at the bottom of the tab, lists every individual realized gain/loss event calculated from your transaction history that falls within the selected account scope.
    * It provides a granular view of each sale that resulted in a capital gain or loss.
    * Key columns often include:
        * `Date`: Date the sale (realization event) occurred.
        * `Symbol`: The stock or ETF that was sold.
        * `Account`: The account where the sale took place.
        * `Quantity Sold`: The number of units sold.
        * `Avg Sale Price (Local)`: Average price per unit at which the asset was sold, in its local currency.
        * `Total Proceeds (Local)`: Total cash received from the sale, in local currency.
        * `Total Cost Basis (Local)`: The original cost of the units sold, in local currency.
        * `Realized Gain (Local)`: The profit or loss from the sale in local currency (Proceeds - Cost Basis).
        * `FX Rate`: The foreign exchange rate applied if the local currency of the asset differs from your global display currency.
        * `Realized Gain (Display)`: The profit or loss from the sale, converted to your chosen global `Display Currency`.
        * `Term`: Indicates if the gain/loss is Short-Term or Long-Term (this depends on holding period calculation logic, which may vary).
    * This table is sortable by clicking on its column headers.

**How Capital Gains are Calculated:**

* Investa calculates realized capital gains when you record a "Sell" transaction for a stock/ETF or a "Buy to Cover" transaction for a short position.
* It uses a cost basis accounting method (commonly First-In, First-Out or FIFO, though the specific method might be configurable or fixed in the backend logic) to determine the cost of the shares being sold.
* The realized gain/loss is then the difference between the sale proceeds (net of any fees if accounted for separately) and the calculated cost basis of those specific shares.
* Foreign exchange effects are also considered if the asset's local currency differs from your display currency.

This tab is essential for understanding the actual profits you've locked in from your trading activities and can be very useful for tax estimation purposes (though always consult with a tax professional for official advice).

## Part 9: Rebalancing Your Portfolio

The **"Rebalancing"** tab is a powerful tool that helps you align your current portfolio with your desired target allocations. It calculates the specific buy and sell trades needed to achieve your goals.

1. **Accessing the Tab:** Click on the "Rebalancing" tab in the main interface.

2. **How It Works:**
    * **Load Current Holdings:** The tab automatically loads your current holdings based on the accounts selected in the main filter, displaying each asset's symbol, current quantity, price, and market value.
    * **Define Target Allocations:** For each asset listed (including `$CASH`), enter your desired target allocation as a percentage in the "Target %" column. The total of all target percentages must equal 100%.
    * **Calculate Trades:** Click the "Calculate Rebalance" button. Investa will compute the difference between your current allocation and your target allocation for each asset.

3. **Interpreting the Results:**
    * The table will update to show:
        * `Target Value`: The ideal market value for each asset based on your target percentage.
        * `Action`: The required action, which will be "Buy", "Sell", or "Hold".
        * `Trade Value`: The total dollar amount to buy or sell for that asset.
        * `Trade Quantity`: The number of shares to buy or sell.
    * A summary at the bottom shows the total amount of funds that need to be freed up from sales and the total amount required for purchases, helping you manage the cash flow for the rebalancing process.

This feature simplifies the often-complex task of rebalancing, providing clear, actionable steps to maintain your investment strategy.

## Part 10: Advanced Portfolio Analysis

The **"Advanced Analysis"** tab provides sophisticated tools to gain deeper insights into your portfolio's risk and return characteristics. It contains three sub-tabs.

### Correlation Matrix

* **Purpose:** This tab visualizes how the daily returns of your assets move in relation to one another.
* **How to Use:** Simply click the "Calculate Correlation" button. The assets in your currently filtered portfolio will be analyzed.
* **Interpretation:** A heatmap is generated.
  * A value of **1** (dark green/blue) means the assets are perfectly positively correlated (they move in the same direction).
  * A value of **-1** (dark red) means they are perfectly negatively correlated (they move in opposite directions).
  * A value near **0** (neutral color) means there is little to no correlation.
* This is useful for understanding diversification. A portfolio with many highly correlated assets may carry more concentrated risk.

### Factor Analysis

* **Purpose:** This tool performs a regression analysis to determine what market factors (e.g., the overall market, company size, value vs. growth) are driving your portfolio's returns.
* **How to Use:** Select a model (`Fama-French 3-Factor` or `Carhart 4-Factor`) and click "Run Factor Analysis".
* **Interpretation:** The results table shows key regression statistics:
  * `alpha`: Represents the excess return of your portfolio above what would be expected based on the market factors. A positive alpha is often considered a sign of skill.
  * `beta` (e.g., `Mkt-RF`, `SMB`, `HML`): Measures your portfolio's sensitivity to each factor. For example, a market beta greater than 1 indicates higher volatility than the market.
  * `R-squared`: Shows the percentage of your portfolio's returns that can be explained by the model.

### Scenario Analysis

* **Purpose:** This tab allows you to simulate how your portfolio might perform under specific hypothetical market shocks.
* **How to Use:** Select a scenario from the dropdown (e.g., "S&P 500 drops 10%") and click "Run Scenario".
* **Interpretation:** The results table shows the `Estimated Impact` on the market value of each of your holdings and the portfolio as a whole, based on their historical correlation to the selected market index. This provides a tangible estimate of potential risk.

## Part 11: Handy Extras & Advanced Settings

Investa packs several additional features and settings to enhance your portfolio management experience:

* **Symbol Settings (Overrides, Mapping, Exclusions) via `Settings > Symbol Settings...`:**
    This powerful dialog is your go-to for managing how individual symbols are treated within Investa. It directly modifies the `manual_overrides.json` file.
  * **Manual Overrides Tab:**
    * If data fetched from Yahoo Finance for a symbol (like its current `price`, `asset_type`, `sector`, `geography`, or `industry`) is missing, incorrect, or you wish to use a custom value, you can set it here.
    * Enter the `Symbol`, select the `Field to Override` (e.g., price, sector), and provide the `New Value`. Click "Set Override".
    * **Editing Overrides:** You can now see a table of your existing overrides. Click the "Edit" button next to any entry to populate the form fields with its current values, making it easy to adjust and update without starting from scratch.
    * This gives you fine-grained control over how your assets are classified and valued, which is especially useful for non-standard assets or correcting API data.
  * **Symbol Mapping Tab:**
    * Useful if a ticker symbol has changed or if you use an alternative symbol in your records (e.g., `BRK.B` vs. `BRK-B`).
    * Enter the `Original Symbol` (as it might appear in your transactions or from an old data source) and the `Mapped Symbol` (the one Yahoo Finance recognizes or your preferred primary symbol).
    * Investa will then treat all instances of the original symbol as the mapped symbol for data fetching and calculations.
  * **Excluded Symbols Tab:**
    * If you have symbols you want Investa to completely ignore for market data fetching and calculations (e.g., delisted stocks you only keep for historical record, private assets you track manually), add them here.

* **Fundamental Data Viewer (Toolbar & Context Menu):**
  * Quickly look up detailed financial information for any stock. Enter a symbol in the "Symbol for Fundamentals" box on the toolbar and click "Get Fundamentals", or right-click a holding in the Holdings Table and select "View Fundamentals".
  * The viewer is organized into tabs:
    * **Summary:** Company profile, key statistics (Market Cap, P/E, EPS), dividend information, and price history.
    * **Financials:** Annual and quarterly Income Statements. Now includes Dividend Yield and Rate.
    * **Balance Sheet:** Annual and quarterly Balance Sheet data.
    * **Cash Flow:** Annual and quarterly Cash Flow statements.
    * *(Options Tab might be present but is typically for future development).*

* **Account Currencies (`Settings > Account Currencies...`):**
  * If you have multiple investment accounts and they operate in different local currencies (e.g., a US brokerage in USD, a UK brokerage in GBP, a Thai brokerage in THB), this dialog allows you to assign a specific currency to each account.
  * This is crucial for accurate cost basis, gain/loss calculations, and proper aggregation before conversion to your main `Display Currency`.

* **Choose Currencies for Display (`Settings > Choose Currencies...`):**
  * Customize the list of currencies available in the main `Display Currency` dropdown on the dashboard.
  * Select your frequently used currencies from a comprehensive list to keep the dropdown tidy and relevant.

* **Data Caching & Clearing Cache:**
  * Investa automatically caches market data (prices, FX rates, fundamental info) to speed up loading times and reduce API calls to Yahoo Finance. Cache settings (like duration) are generally managed in `config.py`.
  * If you suspect your cached data is stale or corrupted, you can clear it via **Settings > Clear Cache Files...**. This will delete the cached market data, and Investa will fetch fresh data on the next "Refresh All".

* **Configuration Persistence (`gui_config.json` & `manual_overrides.json`):**
  * **`gui_config.json`:** Stores your UI preferences and operational settings. This includes:
      * Path to the currently loaded SQLite database file (`investa_transactions.db` or user-specified).
      * Selected global display currency.
      * List of active/selected investment accounts for filtering.
      * Current graph settings (date ranges, intervals, selected benchmarks).
      * Visibility status for columns in the main holdings table.
      * User-defined list of currencies for quick selection in dropdowns.
  * **`manual_overrides.json`:** Stores symbol-specific configurations and data overrides. This JSON file contains:
      * **`manual_price_overrides`**: Manually set values for `price`, `asset_type`, `sector`, `geography`, and `industry` for specific symbols. Example: `{"AAPL": {"price": 175.00, "asset_type": "Stock", "sector": "Technology", "geography": "United States", "industry": "Consumer Electronics"}}`
      * **`user_symbol_map`**: User-defined mappings from alternative ticker symbols to a primary symbol. Example: `{"BRK.B": "BRK-B", "MSFT.NE": "MSFT"}`
      * **`excluded_symbols`**: A list of symbols to be excluded from market data fetching or certain calculations.
  * These files are located in the application data directory. You generally don't need to edit them manually, as settings are managed through the application's UI.

* **Application Data Directory:**
    As mentioned in Part 1, this directory is vital. It holds your database, config files, cache, and CSV backups.
  * **macOS:** `~/Library/Application Support/StockAlchemist/Investa/`
  * **Windows:** `C:\Users\<YourUserName>\AppData\Local\StockAlchemist\Investa\`
  * **Linux:** `~/.local/share/StockAlchemist/Investa/`

## Part 12: Tips & Troubleshooting

Here are some general tips and common troubleshooting steps, reflecting Investa's database-centric approach:

* **Your Database is Primary:** Remember that `investa_transactions.db` (or your custom-named `.db` file) is your primary data source. While CSV import is supported, all ongoing transaction management (adds, edits, deletes) happens directly in the database.
  * **Backup your `.db` file regularly!** While Investa might create CSV backups during certain operations, direct backup of the database file itself is the most robust way to protect your data.
* **Accuracy is Paramount:** The insights Investa provides are only as good as the data you feed it.
  * Double-check every transaction you enter or import for correct dates, types, symbols, quantities, prices, and especially account assignments.
  * Pay close attention to `$CASH` transactions (Deposits, Withdrawals) as they are critical for accurate Time-Weighted Return (TWR) calculations.
* **"Refresh All" (F5) is Your Best Friend:**
  * After adding/editing/deleting transactions directly in the database.
  * After importing a CSV file.
  * After changing symbol settings (overrides, mappings, exclusions).
  * When you want the absolute latest market prices and FX rates.
  * If something just doesn't look right, a "Refresh All" is often the first step.
* **Use the Status Bar:** The bottom of the Investa window often displays messages about current operations (e.g., "Fetching prices...", "Calculations complete...", "Database loaded"). This can give you an idea of what the application is doing, especially during longer operations.
* **CSV Format Help (`Help > CSV Format Help...`):**
  * Even though the database is primary, this in-app guide is still very useful if you are preparing a CSV file for import. It details the expected headers and data for each transaction type.
* **Troubleshooting Steps:**
  * **Market Data Not Loading (e.g., prices show as 0 or NaN):**
    * **Check Internet Connection:** Ensure you have an active internet connection. `yfinance` needs to fetch data online.
    * **Symbol Validity:** Verify that the stock/ETF symbols in your transactions are correct and recognized by Yahoo Finance. Some symbols might have changed or been delisted. Use the `Settings > Symbol Settings...` to map old symbols to new ones if needed (e.g., `BRK.B` to `BRK-B`).
    * **API Limits/Issues:** Yahoo Finance might occasionally have temporary issues or impose rate limits. Try refreshing data after some time.
    * **Firewall/VPN:** Your firewall or VPN might be blocking requests to Yahoo Finance. Try temporarily disabling them to check.
    * **Excluded Symbols:** Ensure the symbol is not in the exclusion list under `Settings > Symbol Settings...`.
    * **Cache:** Try clearing the cache via `File > Clear Cache and Restart` (if available) or by manually deleting cache files from the application data directory.

  * **CSV Import Errors:**
    * **Incorrect Format:** Ensure your CSV file strictly follows the format described in Part 1 or use the "Standardize Headers" utility during import. Pay close attention to date formats, transaction types, and required fields like `Symbol`, `Quantity`, `Price/Share`.
    * **Special Characters/Encoding:** Save your CSV as UTF-8 encoded. Unusual characters in notes or symbols might cause issues.
    * **Large Files:** For very large CSV files, import in smaller chunks if possible.
    * **Header Row:** Ensure your CSV has a header row that matches either the "Preferred (Cleaned)" or "Compatible (Verbose)" headers.

  * **Data Inaccuracies (e.g., incorrect cost basis, gains, TWR):**
    * **Transaction Data Entry:** Double-check all your transactions in the "Transaction Log" tab for accuracy. Errors in dates, quantities, prices, types (Buy/Sell/Dividend/Split), or fees will lead to miscalculations.
    * **`$CASH` Transactions:** Ensure all cash movements (deposits, withdrawals, fees paid from cash, inter-account transfers) are correctly logged using the `$CASH` symbol as described. Missing cash flow data is a common source of TWR discrepancies.
    * **Splits and Dividends:** Verify that stock splits are entered correctly with the right ratio and that all dividends (especially reinvested ones) are recorded.
    * **Currency Settings:** Ensure your global display currency and account-specific currencies are set correctly.
    * **Manual Overrides:** Check if any manual overrides for prices in `Settings > Symbol Settings...` are affecting calculations unexpectedly.

  * **Application Slowdown:**
    * **Large Database:** A very large number of transactions over many years can slow down calculations.
    * **Market Data Fetching:** Initial data fetch for many symbols can be slow. Subsequent loads should be faster due to caching.
    * **Numba JIT Compilation:** The first time calculations are run (e.g., historical TWR), Numba compiles functions, which might take a moment. Subsequent runs are faster.

  * **"File not found" error for database on startup:**
    * This usually means the database file (`investa_transactions.db` or a custom-named one) that was last opened cannot be found at its previous location.
    * The application will prompt you to either locate the existing `.db` file or create a new one.
    * Ensure the path stored in `gui_config.json` under `database_path` is correct or select the correct file when prompted.

  * **Error related to `matplotlib` or `PySide6` backend:**
    * Ensure these libraries are correctly installed in your Python environment. Reinstalling them (`pip install --force-reinstall PySide6 matplotlib`) might help.
    * Conflicts with other GUI libraries or incorrect environment setup can sometimes cause these. Using a clean virtual environment is recommended.

## Part 13: The Investa Web Dashboard

Investa now includes a modern Web Dashboard that allows you to view your portfolio from any device on your local network (e.g., your smartphone, tablet, or another computer).

### Getting Started with the Web App

1.  **Start the Application:**
    We've made this easy with a single script that launches both the backend and frontend for you. Open your terminal in the project root and run:
    
    ```bash
    ./start_investa.sh
    ```
    
    This command starts the API server (port 8000) and the web interface (port 3000) simultaneously. It will also print your machine's local network IP address (e.g., providing a Tailscale IP or local IP) for remote access.

2.  **Accessing the Dashboard:**
    *   **Local Machine:** Open your browser and go to `http://localhost:3000`.
    *   **Remote Device:** Find the IP address of your computer running Investa (e.g., `192.168.1.15`). On your mobile device connected to the same Wi-Fi, go to `http://192.168.1.15:3000`.

### Features

The Web Dashboard mirrors many of the key features of the desktop application:

*   **Dashboard Summary:** View your Net Value, Daily P&L, Total Return, and key metrics at a glance.
*   **Performance Graph:** Interactive charts for TWR and Portfolio Value.
*   **Markets Tab (Mobile):** Dedicated tab to track major market indices (Dow, Nasdaq, S&P 500) while on the go.
*   **Holdings:** A sorted list of your current positions.
*   **Transactions Log:** View your history of trades and cash movements.
*   **Asset Allocation & Analysis:** Visual breakdowns of your portfolio.

*Note: Currently, the Web Dashboard is primarily for **viewing** and **analyzing** your data. For heavy transaction management (bulk imports, editing past records), we recommend using the Desktop Application.*

---

That's the grand tour of Investa! It's a powerful tool designed to give you deep insights into your investments. Take your time, ensure your data is accurate, and explore all the features to make the most of your portfolio analysis. Happy investing!
