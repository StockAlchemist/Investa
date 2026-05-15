"""Rebuild ING Direct, Sharebuilder, and Penson accounts for dheematan."""
import sqlite3, os

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'users', 'dheematan', 'portfolio.db')
conn = sqlite3.connect(DB)
c = conn.cursor()

# Phase 1: Purge legacy accounts
print("=== Phase 1: Purging legacy accounts ===")
for acct in ['Sharebuilder', 'ING Direct', 'Penson']:
    c.execute("DELETE FROM transactions WHERE Account=? AND user_id=7", (acct,))
    print(f"  Deleted {c.rowcount} rows from {acct}")

def ins(date, typ, sym, qty, pps, total, comm, acct, note, to_acct=None):
    c.execute("""INSERT INTO transactions (Date,Type,Symbol,Quantity,"Price/Share","Total Amount",
        Commission,Account,"Split Ratio",Note,"Local Currency","To Account",user_id)
        VALUES (?,?,?,?,?,?,?,?,1.0,?,'USD',?,7)""",
        (date, typ, sym, qty, pps, total, comm, acct, note, to_acct))

# Phase 2: ING Direct
print("\n=== Phase 2: ING Direct ===")
# 2a. Deposits
ins('2002-06-29','deposit','$CASH',4724.1,1.0,4724.1,0,'ING Direct','Initial deposit for IDIOX/IDLOX/IDMOX')
ins('2004-08-13','deposit','$CASH',2821.08,1.0,2821.08,0,'ING Direct','Deposit for IDBOX')
ins('2005-12-16','deposit','$CASH',113.9,1.0,113.9,0,'ING Direct','Deposit for fund additions')
ins('2005-12-30','deposit','$CASH',115.53,1.0,115.53,0,'ING Direct','Deposit for IDBOX addition')

# 2b. Original purchases (known cost basis from 2006 Schedule D)
ins('2002-06-29','buy','IDIOX',180,6.0275,-1084.96,0,'ING Direct','Cost basis from 2006 Schedule D')
ins('2002-06-29','buy','IDLOX',153,11.854,-1813.7,0,'ING Direct','Cost basis from 2006 Schedule D')
ins('2002-06-29','buy','IDMOX',162,11.268,-1825.44,0,'ING Direct','Cost basis from 2006 Schedule D')
ins('2004-08-13','buy','IDBOX',271,10.41,-2821.08,0,'ING Direct','Cost basis from 2006 Schedule D')
ins('2005-12-16','buy','IDIOX',2,7.235,-14.47,0,'ING Direct','Cost basis from 2006 Schedule D')
ins('2005-12-16','buy','IDLOX',1,15.19,-15.19,0,'ING Direct','Cost basis from 2006 Schedule D')
ins('2005-12-16','buy','IDMOX',8,10.53,-84.24,0,'ING Direct','Cost basis from 2006 Schedule D')
ins('2005-12-30','buy','IDBOX',11,10.503,-115.53,0,'ING Direct','Cost basis from 2006 Schedule D')

# Unknown cost basis lots (will be sold in 2009)
ins('2002-06-29','buy','IDIOX',440.446,0,0,0,'ING Direct','Unknown cost basis - transferred to SB 10/2008')
ins('2002-06-29','buy','IDLOX',141.926,0,0,0,'ING Direct','Unknown cost basis - transferred to SB 10/2008')
ins('2002-06-29','buy','IDMOX',297.775,0,0,0,'ING Direct','Unknown cost basis - transferred to SB 10/2008')

# 2c. 2006 Sales
ins('2006-01-03','sell','IDIOX',-2,11.855,23.71,0,'ING Direct','Amended 2006 Schedule D')
ins('2006-01-03','sell','IDBOX',-271,10.288,2788.17,0,'ING Direct','Amended 2006 Schedule D')
ins('2006-01-30','sell','IDBOX',-11,10.38,114.18,0,'ING Direct','Amended 2006 Schedule D')
ins('2006-01-30','sell','IDIOX',-180,9.877,1777.87,0,'ING Direct','Amended 2006 Schedule D')
ins('2006-01-30','sell','IDLOX',-153,16.559,2533.67,0,'ING Direct','Amended 2006 Schedule D')
ins('2006-01-30','sell','IDLOX',-1,21.22,21.22,0,'ING Direct','Amended 2006 Schedule D')
ins('2006-01-30','sell','IDMOX',-162,18.006,2917.01,0,'ING Direct','Amended 2006 Schedule D')
ins('2006-01-30','sell','IDMOX',-8,16.828,134.62,0,'ING Direct','Amended 2006 Schedule D')

# Withdrawal of 2006 sale proceeds
ins('2006-01-30','withdrawal','$CASH',0,0,-10310.45,0,'ING Direct','Withdrawal of fund sale proceeds')

# 2d. Dividends (2008-2009, from statements)
ins('2008-11-05','dividend','IDLOX',0,0,23.57,0,'ING Direct','ING DIRECT-2008')
ins('2008-12-19','dividend','IDIOX',0,0,106.24,0,'ING Direct','ING DIRECT-2008')
ins('2008-12-19','dividend','IDMOX',0,0,25.85,0,'ING Direct','ING DIRECT-2008')
ins('2008-12-19','dividend','IDLOX',0,0,5.17,0,'ING Direct','ING DIRECT-2008')
ins('2008-12-31','dividend','$CASH',0,0,0.04,0,'ING Direct','ING DIRECT-2008')
ins('2009-01-05','dividend','IDLOX',0,0,0.57,0,'ING Direct','ING DIRECT-2009')
ins('2009-01-05','dividend','IDMOX',0,0,0.89,0,'ING Direct','ING DIRECT-2009')
ins('2009-02-02','dividend','$CASH',0,0,0.07,0,'ING Direct','ING DIRECT-2009')

# 2e. 2009 cash withdrawal and fund liquidation
ins('2009-02-09','withdrawal','$CASH',0,0,-162.4,0,'ING Direct','ING DIRECT-2009')
ins('2009-04-30','sell','IDLOX',-141.926,10.33,1466.1,0,'ING Direct','ING DIRECT-2009')
ins('2009-04-30','sell','IDMOX',-297.775,9.25,2754.42,0,'ING Direct','ING DIRECT-2009')
ins('2009-04-30','sell','IDIOX',-440.446,5.87,2585.42,0,'ING Direct','ING DIRECT-2009')
ins('2009-05-01','withdrawal','$CASH',0,0,-6805.94,0,'ING Direct','ING Direct final withdrawal')

print(f"  Inserted ING Direct transactions")

# Phase 3: Sharebuilder
print("\n=== Phase 3: Sharebuilder ===")
# 3a. Deposits & Withdrawals (from statement)
ins('2005-12-31','deposit','$CASH',275.9,1.0,275.9,0,'Sharebuilder','Opening balance')
ins('2006-02-07','deposit','$CASH',15400,1.0,15400,0,'Sharebuilder','ACH DEPOSIT - PLAN')
ins('2006-03-31','deposit','$CASH',5000,1.0,5000,0,'Sharebuilder','ACH DEPOSIT - ONE-TIME')
ins('2006-10-19','deposit','$CASH',10000,1.0,10000,0,'Sharebuilder','ACH DEPOSIT - ONE-TIME')
ins('2006-03-01','withdrawal','$CASH',0,0,-13400,0,'Sharebuilder','ELECTRONIC FUNDS WITHDRAWAL')
ins('2006-06-08','withdrawal','$CASH',0,0,-7300,0,'Sharebuilder','ELECTRONIC FUNDS WITHDRAWAL')
ins('2006-08-30','withdrawal','$CASH',0,0,-9000,0,'Sharebuilder','ELECTRONIC FUNDS WITHDRAWAL')
ins('2006-10-20','withdrawal','$CASH',0,0,-50,0,'Sharebuilder','ELECTRONIC FUNDS WITHDRAWAL')
ins('2006-12-18','fee','$CASH',0,0,-285.97,0,'Sharebuilder','FEE TRANSFER OUT (ACATS)')
ins('2006-12-21','withdrawal','$CASH',0,0,-404.01,0,'Sharebuilder','CHECK ISSUED')
ins('2007-01-04','withdrawal','$CASH',0,0,-33.13,0,'Sharebuilder','CHECK ISSUED')
ins('2007-01-09','withdrawal','$CASH',0,0,-5.94,0,'Sharebuilder','CHECK ISSUED')
ins('2007-02-06','withdrawal','$CASH',0,0,-74.74,0,'Sharebuilder','CHECK ISSUED')

# 3b. Stock Buys
ins('2005-04-05','buy','AAPL',452.12,61.03,-27590.75,0,'Sharebuilder','Statement + Tax Return')
ins('2005-11-11','buy','SPY',94.2,111.96,-10546.83,0,'Sharebuilder','Statement + Tax Return')
ins('2005-11-15','buy','BBW',191,26.18,-5000,0,'Sharebuilder','Statement + Tax Return')
ins('2005-11-25','buy','QQQQ',109.42,37.05,-4053.35,0,'Sharebuilder','Statement + Tax Return')
ins('2006-01-31','buy','DIA',80,104.58,-8366.09,0,'Sharebuilder','Statement + Tax Return (combined lots)')
ins('2006-02-07','buy','VDE',89,78.65,-7000,0,'Sharebuilder','Statement + Tax Return')
ins('2006-02-07','buy','ADRE',267.72,37.39,-10008.58,0,'Sharebuilder','Statement + Tax Return')
ins('2006-02-24','buy','LQD',124,110.49,-13701.48,0,'Sharebuilder','Statement + Tax Return (combined lots)')
ins('2006-04-04','buy','VGK',84.93,58.88,-5000.24,0,'Sharebuilder','Statement + Tax Return')

# 3c. Stock Sells (short-term, from Amended 2006 D-1)
ins('2006-01-19','sell','BBW',-191,28.35,5414.73,0,'Sharebuilder','Amended 2006 Schedule D-1')
ins('2006-02-24','sell','LQD',-124,4.528,561.35,0,'Sharebuilder','Amended 2006 D-1 short-term portion')
ins('2006-02-24','sell','LQD',-124,103.17,12793.11,0,'Sharebuilder','Amended 2006 D-1 long-term portion')
ins('2006-06-05','sell','VDE',-89,81.54,7256.76,0,'Sharebuilder','Amended 2006 Schedule D-1')
ins('2006-08-25','sell','DIA',-80,2.454,196.3,0,'Sharebuilder','Amended 2006 D-1 short-term portion')
ins('2006-08-25','sell','DIA',-80,109.735,8778.81,0,'Sharebuilder','Amended 2006 D-1 long-term portion')

# 3c2. Fractional sells on 12/15
ins('2006-12-15','sell','ADRE',-0.72,37.79,27.21,0,'Sharebuilder','Fractional share liquidation pre-ACATS')
ins('2006-12-15','sell','VGK',-0.93,69.92,65.03,0,'Sharebuilder','Fractional share liquidation pre-ACATS')
ins('2006-12-15','sell','AAPL',-0.12,90.5,10.86,0,'Sharebuilder','Fractional share liquidation pre-ACATS')
ins('2006-12-15','sell','QQQQ',-0.42,44.55,18.71,0,'Sharebuilder','Fractional share liquidation pre-ACATS')
ins('2006-12-15','sell','SPY',-0.2,146.35,29.27,0,'Sharebuilder','Fractional share liquidation pre-ACATS')

# 3d. Dividends (from statement, key ones)
for d,amt in [('2006-01-12',25),('2006-03-13',50),('2006-05-11',75),
              ('2006-06-12',25),('2006-07-11',25),('2006-08-11',50),
              ('2006-08-14',11.86),('2006-08-21',0.61),('2006-08-31',1.34),
              ('2006-09-11',75),('2006-09-21',0.73),('2006-10-11',25),
              ('2006-10-23',5.93),('2006-10-31',112.69),('2006-11-21',4.51),
              ('2006-12-11',25),('2006-12-21',1.04),('2006-12-29',33.11)]:
    ins(d,'dividend','$CASH',0,0,amt,0,'Sharebuilder','Sharebuilder 2006 statement')

# DRV buys (dividend reinvestment fractional shares)
for d,sym,qty in [('2006-04-28','DIA',0.1146),('2006-04-28','SPY',0.0434),
                  ('2006-04-28','QQQQ',0.0067),('2006-04-28','ADRE',0.1281),
                  ('2006-07-31','DIA',0.1166),('2006-07-31','SPY',0.4054),
                  ('2006-07-31','QQQQ',0.0757),('2006-07-31','ADRE',1.5521),
                  ('2006-10-31','SPY',0.3944),('2006-10-31','QQQQ',0.0603),
                  ('2006-10-31','ADRE',1.6451)]:
    ins(d,'buy',sym,qty,0,0,0,'Sharebuilder','DRV reinvestment')

# 3e. Transfer to TD Ameritrade (ACATS)
for sym,qty in [('AAPL',452),('VGK',84),('SPY',94),('QQQQ',109),('ADRE',267)]:
    ins('2006-12-19','transfer',sym,qty,0,0,0,'Sharebuilder',
        'ACATS Transfer: Sharebuilder -> TD Ameritrade','TD Ameritrade')

print(f"  Inserted Sharebuilder transactions")

# Phase 4: Penson
print("\n=== Phase 4: Penson ===")
# Initial deposit
ins('2006-03-23','deposit','$CASH',1807.49,1.0,1807.49,0,'Penson','Initial deposit')

# 2006 trades (buy+sell pairs from Amended 2006 D-1)
penson_2006 = [
    ('2006-03-23','2006-03-24','AAPL',30,1807.49,1800.85),
    ('2006-05-17','2006-08-01','AAPL',25,1619.24,1681.2),
    ('2006-05-19','2006-08-18','SPY',40,5057.39,5190.85),
    ('2006-05-23','2006-08-18','VWO',79,5224.89,5266.93),
    ('2006-08-21','2006-08-23','AAPL',95,6328.09,6405.51),
    ('2006-08-28','2006-08-28','AAPL',95,6460.14,6409.31),
    ('2006-08-21','2006-09-06','SPY',45,5858.39,5888.22),
    ('2006-09-06','2006-09-08','AAPL',90,6310.19,6518.2),
    ('2006-09-11','2006-09-12','AAPL',80,5826.99,5766.43),
    ('2006-09-15','2006-09-27','AAPL',165,12269.09,12602.61),
    ('2006-10-03','2006-10-05','AAPL',170,12650.99,12653.96),
    ('2006-10-12','2006-10-13','AAPL',170,12538.79,12780.6),
    ('2006-10-18','2006-10-19','AAPL',173,12877.65,13656.66),
    ('2006-10-24','2006-10-24','AAPL',165,13455.44,13359.93),
    ('2006-10-27','2006-11-09','AAPL',170,13664.19,14098.06),
    ('2006-11-13','2006-11-15','AAPL',165,13826.69,14099.94),
    ('2006-11-22','2006-11-24','AAPL',163,14400.78,14586.68),
    ('2006-11-29','2006-12-28','AAPL',300,27664.49,24257.24),
]

# Deposit for initial batch of non-AAPL buys
ins('2006-05-17','deposit','$CASH',11901.52,1.0,11901.52,0,'Penson','Deposit for SPY/VWO/AAPL buys')

for buy_d, sell_d, sym, qty, cost, proceeds in penson_2006:
    pps_buy = cost / qty
    ins(buy_d,'buy',sym,qty,pps_buy,-cost,0,'Penson','Amended 2006 Schedule D-1')
    pps_sell = proceeds / qty
    ins(sell_d,'sell',sym,-qty,pps_sell,proceeds,0,'Penson','Amended 2006 Schedule D-1')

# 2007 trades (from Amended 2007 Schedule D-1)
penson_2007 = [
    ('2006-12-28','2007-01-10','AAPL',250,21327.99,23953.76),
    ('2007-01-16','2007-01-18','AAPL',280,26824.19,25197.61),
    ('2007-04-04','2007-03-23','AAPL',20,1901.79,1870.88),
    ('2007-04-20','2007-04-30','AAPL',40,3646.19,4003.34),
    ('2007-04-20','2007-04-30','BHP',80,3994.99,3901.75),
    ('2007-04-20','2007-04-30','CVX',50,3880.49,3916.95),
    ('2007-02-27','2007-04-19','XLE',170,9871.49,10570.83),
]

for buy_d, sell_d, sym, qty, cost, proceeds in penson_2007:
    pps_buy = cost / qty
    ins(buy_d,'buy',sym,qty,pps_buy,-cost,0,'Penson','Amended 2007 Schedule D')
    pps_sell = proceeds / qty
    ins(sell_d,'sell',sym,-qty,pps_sell,proceeds,0,'Penson','Amended 2007 Schedule D')

# Penson dividends/interest/fees (from Penson Tax Form 2007)
ins('2007-03-28','dividend','XLE',0,0,31.78,0,'Penson','Penson Tax Form 2007')
ins('2007-05-10','dividend','$CASH',0,0,3.86,0,'Penson','Interest - Penson Tax Form 2007')
ins('2007-01-31','fee','$CASH',0,0,-63.52,0,'Penson','Fee - Penson Tax Form 2007')

# Calculate Penson closing balance and withdraw
# Deposits: 1807.49 + 11901.52 = 13709.01
# Net from trades: sum of all proceeds - sum of all costs
total_proceeds = sum(p[5] for p in penson_2006 + penson_2007)
total_costs = sum(p[4] for p in penson_2006 + penson_2007)
net_trades = total_proceeds - total_costs
# Plus dividends/interest minus fees
net_other = 31.78 + 3.86 - 63.52
closing_balance = 13709.01 + net_trades + net_other
ins('2007-06-01','withdrawal','$CASH',0,0,-round(closing_balance,2),0,'Penson',
    f'Final withdrawal (calculated: {closing_balance:.2f})')

print(f"  Inserted Penson transactions")
print(f"  Penson closing balance: ${closing_balance:.2f}")

conn.commit()

# Verify counts
for acct in ['ING Direct','Sharebuilder','Penson']:
    c.execute("SELECT COUNT(*) FROM transactions WHERE Account=?", (acct,))
    print(f"  {acct}: {c.fetchone()[0]} records")

conn.close()
print("\n=== Rebuild complete ===")
