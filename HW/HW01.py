import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
from fredapi import Fred

FRED_API_KEY = '0a7b2ead6894bc702c9badee383397f7'
fred = Fred(api_key=FRED_API_KEY)

# 1. Obtain GDP data (U.S. real GDP, quarterly)
## USA: GDPC1
## JAPAN: JPNRGDPEXP
## INDONESIA: NGDPRSAXDCIDQ
gdp_raw = fred.get_series('NGDPRSAXDCIDQ')
gdp_q = gdp_raw.resample('QE').mean()
gdp_log = np.log(gdp_q)

# 2. Apply HP filter with multiple lambda
lambdas = [10, 100, 1600]
trends = {}
cycles = {}

for lam in lambdas:
    cycle, trend = hpfilter(gdp_log, lamb=lam)
    trends[lam] = trend
    cycles[lam] = cycle

# 3. Trend Comparison Plot
plt.figure(figsize=(12, 6))
plt.plot(gdp_log, label='Log Real GDP', color='black', linewidth=2)
for lam in lambdas:
    plt.plot(trends[lam], label=f'Trend (λ={lam})')
plt.title('Log GDP and HP Filter Trends')
plt.xlabel('Date')
plt.ylabel('Log GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 4. Comparison plots of circulating components
plt.figure(figsize=(12, 6))
for lam in lambdas:
    plt.plot(cycles[lam], label=f'Cycle (λ={lam})')
plt.title('HP Filter Cyclical Components (Different λ)')
plt.xlabel('Date')
plt.ylabel('Deviation from Trend')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()