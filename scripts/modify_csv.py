import csv
import os


def process_transactions_v2(input_csv_path, output_csv_path):
    """
    Processes a transaction CSV file to add corresponding $CASH transactions
    for specified accounts and transaction types.

    Version 2:
    - For Buy: Adds $CASH Deposit and $CASH Sell.
    - For Sell: Adds $CASH Buy and $CASH Withdrawal.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path where the output CSV file will be saved.
    """
    target_accounts = {"ING Direct", "Sharebuilder", "SET", "Penson", "TD Ameritrade"}
    processed_rows = []

    try:
        with open(input_csv_path, "r", newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            header = next(reader)
            processed_rows.append(header)

            # Define column indices based on a typical header structure
            # "Date (MMM DD, YYYY)",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note
            # Ensure these indices match your CSV header
            try:
                date_col = header.index("Date (MMM DD, YYYY)")
                type_col = header.index("Transaction Type")
                symbol_col = header.index("Stock / ETF Symbol")
                # quantity_col = header.index("Quantity of Units")
                # amount_per_unit_col = header.index("Amount per unit")
                total_amount_col = header.index("Total Amount")
                # fees_col = header.index("Fees")
                account_col = header.index("Investment Account")
                split_ratio_col = header.index("Split Ratio (new shares per old share)")
                # note_col = header.index("Note")
            except ValueError as e:
                print(f"Error: CSV header is missing expected columns: {e}")
                print("Please ensure your CSV header matches the expected format.")
                return

            for row in reader:
                if not row or len(row) < len(header):  # Skip empty or malformed rows
                    if row:  # if malformed but not empty, append it as is
                        processed_rows.append(row)
                    print(
                        f"Warning: Skipped processing or appended malformed row as-is: {row}"
                    )
                    continue

                transaction_date = row[date_col]
                transaction_type = row[type_col]
                original_stock_symbol = row[symbol_col]
                # Clean up total_amount: remove commas if present
                try:
                    transaction_total_amount_str = row[total_amount_col].replace(
                        ",", ""
                    )
                    # Ensure it's a valid float string, otherwise skip or handle
                    float(transaction_total_amount_str)  # Test conversion
                except ValueError:
                    # If Total Amount is not a valid number, we can't process it for $CASH transactions
                    processed_rows.append(
                        row
                    )  # Add original row and skip $CASH generation
                    print(
                        f"Warning: Invalid 'Total Amount' for row: {row}. Skipping $CASH generation."
                    )
                    continue

                investment_account = row[account_col]
                original_split_ratio = (
                    row[split_ratio_col] if len(row) > split_ratio_col else ""
                )

                # Placeholder for other columns to maintain row length
                # Assuming the last column is 'Note' and before it is 'Split Ratio'
                # The script will try to fill up to len(header) columns

                new_rows_for_this_transaction = []

                if investment_account in target_accounts:
                    if transaction_type == "Buy":
                        # 1. Add $CASH Deposit
                        deposit_cash_row = [
                            transaction_date,
                            "Deposit",
                            "$CASH",
                            transaction_total_amount_str,
                            "1.00",
                            transaction_total_amount_str,
                            "0.00",
                            investment_account,
                            original_split_ratio,
                            f"Auto-generated: Cash deposit for {original_stock_symbol} buy",
                        ]
                        new_rows_for_this_transaction.append(deposit_cash_row)

                        # 2. Add $CASH Sell
                        sell_cash_row = [
                            transaction_date,
                            "Sell",
                            "$CASH",
                            transaction_total_amount_str,
                            "1.00",
                            transaction_total_amount_str,
                            "0.00",
                            investment_account,
                            original_split_ratio,
                            f"Auto-generated: Cash settlement for {original_stock_symbol} buy",
                        ]
                        new_rows_for_this_transaction.append(sell_cash_row)

                    elif (
                        transaction_type == "Sell"
                    ):  # Excludes "Short Sell", "Buy to Cover"
                        # 1. Add $CASH Buy (cash received from stock sale)
                        buy_cash_row = [
                            transaction_date,
                            "Buy",
                            "$CASH",
                            transaction_total_amount_str,
                            "1.00",
                            transaction_total_amount_str,
                            "0.00",
                            investment_account,
                            original_split_ratio,
                            f"Auto-generated: Cash received from {original_stock_symbol} sell",
                        ]
                        new_rows_for_this_transaction.append(buy_cash_row)

                        # 2. Add $CASH Withdrawal
                        withdrawal_cash_row = [
                            transaction_date,
                            "Withdrawal",
                            "$CASH",
                            transaction_total_amount_str,
                            "1.00",
                            transaction_total_amount_str,
                            "0.00",
                            investment_account,
                            original_split_ratio,
                            f"Auto-generated: Cash withdrawal from {original_stock_symbol} sell proceeds",
                        ]
                        new_rows_for_this_transaction.append(withdrawal_cash_row)

                # Add newly generated rows (if any)
                for new_row in new_rows_for_this_transaction:
                    while len(new_row) < len(header):
                        new_row.append(
                            ""
                        )  # Pad with empty strings to match header length
                    processed_rows.append(new_row)

                # Add the original row
                processed_rows.append(row)

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback

        traceback.print_exc()
        return

    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)  # Quote all fields
            writer.writerows(processed_rows)
        print(f"Successfully processed transactions and saved to {output_csv_path}")
    except Exception as e:
        print(f"Error writing to output file {output_csv_path}: {e}")


if __name__ == "__main__":
    base_path = "/Users/kmatan/Library/CloudStorage/OneDrive-MahidolUniversity/finance/Stocks/Evaluations/python/Investa/"
    input_file = os.path.join(base_path, "my_transactions.csv")
    output_file_v2 = os.path.join(base_path, "my_transactions_processed_v2.csv")

    process_transactions_v2(input_file, output_file_v2)
