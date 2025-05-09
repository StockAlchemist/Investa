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

## Part 5: Managing Your Transactions

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
  * **Caution:** Editing or deleting directly modifies your source CSV file, so be careful! It's always a good idea to have a backup of your CSV.

## Part 6: Handy Extras

* **Manual Price Overrides:**
    If Yahoo Finance doesn't have data for a symbol, or if it's incorrect, you can set a manual price. Go to **Settings > Manual Prices...**.
* **Fundamental Data Viewer:**
    Want to quickly look up some key stats for a stock? Go to **Tools > Fundamental Data Viewer...**, enter a symbol, and Investa will try to fetch data like P/E ratio, market cap, etc.
* **Account Currencies:**
    If your different investment accounts operate in different local currencies (e.g., one in USD, another in THB), you can specify this under **Settings > Account Currencies...**. This helps with accurate cost basis and gain calculations before converting to your main display currency.
* **Data Caching:**
    Investa is smart! It caches market data it fetches. This means if you refresh again soon, it'll load much faster and won't hit the Yahoo Finance servers as often. Cache settings are in `config.py`.
* **Configuration Persistence:**
    The app remembers your last used CSV file, your display currency, which columns you like to see, and your graph settings. These are saved in a `gui_config.json` file in your user's application support directory.

## Part 7: Tips & Troubleshooting

* **CSV is Key:** The accuracy of Investa depends entirely on the accuracy and completeness of your transactions CSV. Double-check your entries!
* **"Refresh All" is Your Friend:** Whenever you change your CSV file externally, or want the very latest prices, hit "Refresh All".
* **Check the Status Bar:** The bottom of the window often shows messages about what Investa is doing (e.g., "Fetching prices...", "Calculations complete.").
* **CSV Format Help:** Seriously, if you're unsure about the CSV, the in-app help (**Help > CSV Format Help...**) is very useful.

---

That's the grand tour of Investa! It's a powerful tool, so take your time exploring its features. The more accurate your transaction data, the more insightful your portfolio analysis will be. Happy investing!
