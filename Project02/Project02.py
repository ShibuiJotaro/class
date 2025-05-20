import pandas as pd
import numpy as np

# データ取得
pwt90 = pd.read_stata('https://www.rug.nl/ggdc/docs/pwt90.dta')

# OECD主要国リスト
oecd_countries = [
    'Japan', 'United States', 'Germany', 'France', 'United Kingdom',
    'Italy', 'Canada', 'Australia', 'Netherlands', 'Sweden', 'Belgium',
    'Austria', 'Denmark', 'Finland', 'Norway', 'Switzerland', 'Spain',
    'Portugal', 'Ireland', 'Greece', 'New Zealand'
]

# 分析対象期間
start_year = 1970
end_year = 2010

# データ抽出・前処理
data = pwt90[
    pwt90['country'].isin(oecd_countries) &
    pwt90['year'].between(start_year, end_year)
]

relevant_cols = ['country', 'year', 'rgdpna', 'rkna', 'emp', 'avh', 'pop', 'labsh', 'rtfpna']
data = data[relevant_cols].dropna()

# 追加変数計算
data['alpha'] = 1 - data['labsh']
data['y_n'] = data['rgdpna'] / data['emp']  # 一人当たりGDP
data['k_n'] = data['rkna'] / data['emp']    # 一人当たり資本
data['hours'] = data['emp'] * data['avh']

# 各国ごとに成長率などを計算
def calc_growth_table(df):
    start = df[df['year'] == start_year].iloc[0]
    end = df[df['year'] == end_year].iloc[0]
    years = end_year - start_year

    g_y = (np.log(end['y_n']) - np.log(start['y_n'])) / years * 100
    g_k = (np.log(end['k_n']) - np.log(start['k_n'])) / years * 100
    alpha = (start['alpha'] + end['alpha']) / 2
    tfp_growth = g_y - alpha * g_k
    capital_deepening = alpha * g_k
    tfp_share = tfp_growth / g_y if g_y != 0 else np.nan
    capital_share = capital_deepening / g_y if g_y != 0 else np.nan

    return pd.Series({
        'Growth Rate': round(g_y, 2),
        'TFP Growth': round(tfp_growth, 2),
        'Capital Deepening': round(capital_deepening, 2),
        'TFP Share': round(tfp_share, 2),
        'Capital Share': round(capital_share, 2),
    })

results = data.groupby('country').apply(calc_growth_table)
results = results.sort_values('Growth Rate', ascending=False)
print(results)

# 必要であればCSV出力
# results.to_csv("growth_accounting_table.csv")
