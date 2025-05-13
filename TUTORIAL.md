# Welcome to Investa! A Quick Tutorial

Investa is your personal desktop assistant for keeping a close eye on your investment portfolio. It takes your transaction history from a simple CSV file, fetches the latest market data, and then presents you with a clear, detailed picture of how your investments are doing.

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

        Otherwise, install them manually:

        ```bash
        pip install PySide6 pandas numpy matplotlib yfinance scipy mplcursors requests numba
        ```

2. **Prepare Your Transaction Data (The CSV File):**
    This is the most crucial step! Investa needs your transaction history in a specific CSV format. Here are the columns it expects, in order:

    1. `Date (MMM DD, YYYY)`: e.g., *Jan 01, 2023*
    2. `Transaction Type`: *Buy, Sell, Dividend, Split, Deposit, Withdrawal, Fees*
    3. `Stock / ETF Symbol`: e.g., *AAPL, VTI*. Use **`$CASH`** for general cash movements (like deposits into your brokerage account) or cash dividends/fees not tied to a specific stock.
    4. `Quantity of Units`: Number of shares or units.
    5. `Amount per unit`: Price per share/unit.
    6. `Total Amount`: (Optional for Buy/Sell if Quantity and Amount per unit are provided). For dividends, this is the total dividend amount.
    7. `Fees`: Any commissions or fees for the transaction.
    8. `Investment Account`: The name of your brokerage account (e.g., *My Brokerage, Roth IRA*). This helps you filter later.
    9. `Split Ratio (new shares per old share)`: Only needed for 'Split' transactions (e.g., *2* for a 2-for-1 split).
    10. `Note`: Any personal notes about the transaction (optional).

### Understanding `$CASH` Transactions

The special symbol **`$CASH`** plays a vital role in accurately tracking your portfolio's value and performance. Here's how it works and when to use it:

* **What it Represents:** `$CASH` is used to denote monetary movements or balances within your investment accounts that are not directly tied to a specific stock or ETF. Think of it as the cash component of your account.
* **When to Use `$CASH`:**
  * **Deposits:** When you add funds to your brokerage account (e.g., `Transaction Type: Deposit`, `Stock / ETF Symbol: $CASH`).
  * **Withdrawals:** When you take funds out of your brokerage account (e.g., `Transaction Type: Withdrawal`, `Stock / ETF Symbol: $CASH`).
  * **Cash Dividends (Not Reinvested into the Same Stock):** If a dividend is paid out as cash to your account and not automatically reinvested into the stock that paid it, you can record it as a `Dividend` for `$CASH`. For example, `Transaction Type: Dividend`, `Stock / ETF Symbol: $CASH`, `Total Amount: [dividend amount]`. (Alternatively, if a stock pays a dividend and you want to track it against that stock, use the stock's symbol and `Transaction Type: Dividend`).
  * **Fees (Not Tied to a Specific Trade):** For general account maintenance fees, record them as `Transaction Type: Fees`, `Stock / ETF Symbol: $CASH`, `Total Amount: [fee amount]`. For trade-specific fees, include them in the `Fees` column of the Buy/Sell transaction for the specific stock.
  * **Interest Received:** If your account earns interest on its cash balance, you can record this as `Transaction Type: Deposit` (or `Dividend`) for `$CASH`.
* **How It's Treated:**
  * `$CASH` is treated like any other holding in terms of contributing to your portfolio's market value.
  * Its "price per unit" is always considered **1.00** in the currency of the investment account it belongs to. So, a `Quantity of Units` of 100 for `$CASH` means $100 (or 100 units of the account's currency).
  * **`$CASH` Buy/Sell Transactions:** These are crucial for modeling internal cash movements within an account:
    * A **`$CASH` `Buy`** transaction typically signifies cash *increasing* in your account's cash balance from an internal source. For example, when you sell a stock, the proceeds can be represented as a `$CASH` `Buy` (or a `$CASH` `Deposit` if you prefer to view proceeds as an inflow to cash).
    * A **`$CASH` `Sell`** transaction signifies cash *decreasing* from your account's cash balance to be used for an internal purpose, most commonly to fund the purchase of a stock or ETF.
* **Impact on Calculations (Especially TWR Daily Gain):**
  * Accurate `$CASH` transactions are crucial for correctly calculating your portfolio's total value, cost basis (for cash itself, it's usually just its face value), and especially for performance metrics like Time-Weighted Return (TWR), as they represent external cash flows (contributions and withdrawals).
  * **External Flows for TWR:**
    * `$CASH` `Deposit` (e.g., transferring new money into your brokerage account) and `$CASH` `Withdrawal` (e.g., taking money out of your brokerage account) are treated as **external cash flows**.
    * These external flows directly impact the capital base upon which TWR is calculated. A deposit increases the investment base; a withdrawal decreases it. TWR aims to measure performance *excluding* the effect of these contributions or withdrawals.
  * **Internal Conversions for TWR:**
    * When you `Buy` a stock: If you model this with an accompanying `$CASH` `Sell` transaction, this represents an **internal asset conversion**. Cash in your account decreases, and your holding in the new stock increases by the same amount (at cost). This is *not* an external cash flow for TWR purposes.
    * When you `Sell` a stock: If you model the proceeds with an accompanying `$CASH` `Buy` (or `$CASH` `Deposit`), this is also an internal asset conversion. Your stock holding decreases, and your cash balance increases. This is *not* an external cash flow for TWR.
  * **Daily Gain and TWR:**
    * The TWR formula is designed to isolate the performance of your investments from the timing and size of when you add or remove funds.
    * For any given day (or sub-period used in TWR calculation), the return is essentially `(End Market Value - Start Market Value - Net External Cash Flow) / (Start Market Value + Weighted Cash Flows during the period)`.
    * `$CASH` `Buy`/`Sell` transactions that represent these internal asset reallocations do *not* count as "Net External Cash Flow." Their immediate effect on the day of the transaction is a shift in value between the `$CASH` asset and the stock/ETF involved.
    * For example, if you start a day with $1000 cash and no stocks, and then buy $500 of SPY:
            1. Optional: `$CASH` `Deposit` $1000 (if this is new money for the day - this IS an external flow).
            2. `$CASH` `Sell` $500 (internal conversion - cash balance reduces).
            3. `SPY` `Buy` $500 (internal conversion - SPY holding increases).
      * Immediately after these internal conversions, the total portfolio value (cash + SPY) remains unchanged by the act of buying SPY itself (assuming SPY is valued at its purchase price initially).
      * The daily gain for TWR will then depend on how the *remaining cash* and the *market value of SPY* change from this point until the end of the day. If SPY's price increases or decreases, that contributes to the investment performance measured by TWR. The `$CASH` asset itself (with a fixed price of 1.00) does not generate capital gains or losses, though it can receive interest (recorded as a separate transaction).

By diligently recording your `$CASH` movements, including internal `Buy`/`Sell` transactions for cash when appropriate, Investa can provide a more complete and accurate picture of your investment activities, cash flow, and true investment performance.

* **Tip:** For detailed examples and specific requirements for each transaction type, once the app is running, check out the **Help > CSV Format Help...** menu.

## Part 2: Your First Launch and Loading Data

1. **Run Investa:**
    Open your terminal, make sure your virtual environment is active (if you're using one), and navigate to the Investa project directory. Then run:

    ```bash
    python main_gui.py
    ```

2. **Select Your Transactions File:**
    * The first time you run Investa, or if it can't find the last file you used, it will prompt you. You can also go to **File > Open Transactions File...** or click the "Select CSV" button.
    * Navigate to and select the CSV file you prepared in Part 1.
    * If you're starting fresh, you can create a new, empty transactions file via **File > New Transactions File...**.

3. **Refresh and See the Magic!**
    * Click the big "Refresh All" button (or press F5).
    * Investa will now:
        * Read your transactions.
        * Go online (using Yahoo Finance) to fetch the latest market prices for your stocks, ETFs, and any currency exchange rates needed.
        * Calculate all your portfolio metrics.
        * Display everything in the dashboard!

## Part 3: Exploring the Main Dashboard

Once the data is loaded, you'll see the main dashboard. Let's break it down:

* **Portfolio Summary (Top Section):**
    This gives you the big picture:
  * `Market Value`: Total current value of your holdings.
  * `Cost Basis`: How much you originally paid for your current holdings.
  * `Unrealized P/L`: Profit or loss on paper for things you still own.
  * `Realized P/L`: Profit or loss from sales you've made.
  * `Dividends`: Total dividends received.
  * `Total Return`: Your overall gain or loss.

* **Controls (Below Summary):**
  * `Display Currency`: Change the currency for all monetary values shown.
  * `Show Closed Positions`: Toggle to include or exclude assets you've completely sold off.
  * `Account Filter`: Choose to see data for "All Accounts" or select a specific investment account you defined in your CSV. Click "Update Accounts" after changing this.

* **Holdings Table (Main Area):**
    This is a detailed list of everything you own (or owned, if "Show Closed Positions" is on).
  * **Columns:** You'll see things like Symbol, Quantity, Current Price, Market Value, Cost Basis, Gains, etc.
  * **Sorting:** Click on any column header to sort the table by that column. Click again to reverse the sort.
  * **Customize Columns:** Right-click on any column header to choose which columns you want to see or hide.
  * **Context Menu:** Right-click on a specific holding (row) in the table for quick actions like:
    * Viewing its transaction history.
    * Charting its price.

## Part 4: Analyzing Your Performance with Charts

Investa offers several charts to help you visualize your investment journey:

* **Historical Performance Graphs (Bottom Left):**
  * `Plot Type`:
    * **TWR (Time-Weighted Return):** Shows how your portfolio performed percentage-wise, independent of when you added or withdrew money. Great for comparing against benchmarks.
    * **Absolute Value:** Shows the actual monetary value of your portfolio over time.
  * `Benchmark`: Compare your TWR against common market indexes like SPY (S&P 500) or QQQ (Nasdaq 100). You can add more in `config.py`.
  * `Date Range`: Choose the start and end dates for the chart.
  * `Interval`: View data Daily, Weekly, or Monthly.
  * Click "Update Graphs" after changing these settings.

* **Periodic Returns Bar Charts (Bottom Middle):**
    See your portfolio's (and selected benchmark's) percentage returns for specific periods:
  * Annual Returns
  * Monthly Returns
  * Weekly Returns

* **Portfolio Allocation Pie Charts (Bottom Right):**
    Get a visual breakdown of your portfolio:
  * **By Account:** Shows how your assets are distributed across your different investment accounts.
  * **By Holding:** Shows the weight of each individual stock/ETF in your portfolio.

## Part 5: Dividend History

The "Dividend History" tab provides insights into your dividend income.

<!-- It's good to add a screenshot here if you have one -->
<!-- Example: !Dividend History Tab -->

1. **Accessing the Tab**: Click on the "Dividend History" tab in the main tab widget.
2. **Account Filtering**: The dividend data displayed is automatically filtered based on the accounts you have selected in the main "Controls" bar (using the "Accounts" button). If "All Accounts" are selected (or no specific accounts are chosen), dividends from all accounts will be shown.
3. **Aggregation Controls**:
    * **Aggregate by**: Choose to view your dividend totals "Annual", "Quarterly", or "Monthly". This selection affects the bar chart and the summary table.
    * **Periods to Show**: Specify how many of the selected periods (e.g., last 10 years, last 12 quarters) you want to see in the chart and summary table.
4. **Dividend Bar Chart**:
    * Visualizes the total dividend amounts for each aggregated period (e.g., total dividends per year).
    * The Y-axis shows the total dividend amount in your selected display currency.
    * The X-axis shows the periods.
5. **Dividend Summary Table**:
    * Located below the bar chart, this table shows the same aggregated data in tabular form: "Period" and "Total Dividends".
6. **Dividend Transaction History Table**:
    * This table, typically to the right of or below the summary table, lists all individual dividend transactions that contribute to the selected scope (based on account filters).
    * It includes columns like "Date", "Symbol", "Account", "LocalCurrency", "DividendAmountLocal" (amount in the asset's local currency), "FXRateUsed", and "DividendAmountDisplayCurrency" (amount converted to your chosen display currency).
    * You can sort this table by clicking on the column headers.

## Part 6: Managing Your Transactions

Investa isn't just for viewing; you can also manage your transaction data:

* **Viewing Transaction History:**
  * In the Holdings Table, right-click a symbol and select "View Transactions for [Symbol]".
  * A dialog will show all buys, sells, dividends, etc., for that specific holding.

* **Adding a New Transaction:**
  * Go to **Transactions > Add New Transaction...**.
  * Fill in the details in the dialog that appears. This will add a new row to your source CSV file.

* **Editing or Deleting Transactions:**
  * When viewing the transaction history for a symbol (or via **Transactions > View/Edit All Transactions...**), you can select a transaction.
  * Buttons will be available to "Edit Selected" or "Delete Selected".
  * **Caution:** Editing or deleting directly modifies your source CSV file, so be careful! It's always a good idea to have a backup of your CSV (Investa creates backups automatically in the `csv_backups` folder within your application data directory).

## Part 7: Handy Extras

* **Manual Price Overrides:**
    If Yahoo Finance doesn't have data for a symbol, or if it's incorrect, you can set a manual price. Go to **Settings > Manual Prices...**.
* **Fundamental Data Viewer:**
    Want to quickly look up some key stats for a stock? Go to **Tools > Fundamental Data Viewer...**, enter a symbol, and Investa will try to fetch data like P/E ratio, market cap, etc.
* **Account Currencies:**
    If your different investment accounts operate in different local currencies (e.g., one in USD, another in THB), you can specify this under **Settings > Account Currencies...**. This helps with accurate cost basis and gain calculations before converting to your main display currency.
* **Data Caching:**
    Investa is smart! It caches market data it fetches. This means if you refresh again soon, it'll load much faster and won't hit the Yahoo Finance servers as often. Cache settings are in `config.py`.
* **Configuration Persistence & User Files:**
    The app remembers your settings (like the path to your last used CSV, display currency, selected accounts, graph preferences, and column visibility). These are automatically saved when you close the application and stored in a file named `gui_config.json`.

    This file, along with `manual_prices.json` (if you use manual price overrides), is stored in a user-specific application data directory. Cache files and transaction data backups are also stored here. The typical locations are:

  * **macOS**: `~/Library/Application Support/StockAlchemist/Investa/`
  * **Windows**: `C:\Users\<YourUserName>\AppData\Local\StockAlchemist\Investa\` (or `Roaming` instead of `Local`)
  * **Linux**: `~/.local/share/StockAlchemist/Investa/` (or `~/.config/StockAlchemist/Investa/`)

    Inside this directory, you'll find:
  * `gui_config.json`: Your main application settings.
  * `manual_prices.json`: Your manual price overrides.
  * `csv_backups/`: A subfolder containing timestamped backups of your transactions CSV, created when you edit or delete transactions.
  * Cache files for market data.

    You generally don't need to interact with these files directly, but knowing their location can be useful for backups or troubleshooting.

## Part 8: Tips & Troubleshooting

* **CSV is Key:** The accuracy of Investa depends entirely on the accuracy and completeness of your transactions CSV. Double-check your entries!
* **"Refresh All" is Your Friend:** Whenever you change your CSV file externally, or want the very latest prices, hit "Refresh All".
* **Check the Status Bar:** The bottom of the window often shows messages about what Investa is doing (e.g., "Fetching prices...", "Calculations complete.").
* **CSV Format Help:** Seriously, if you're unsure about the CSV, the in-app help (**Help > CSV Format Help...**) is very useful.

---

That's the grand tour of Investa! It's a powerful tool, so take your time exploring its features. The more accurate your transaction data, the more insightful your portfolio analysis will be. Happy investing!
