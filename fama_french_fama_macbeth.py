import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import io
import requests
from scipy.stats import ttest_1samp

tickers = [
    'AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','JPM','V','UNH',
    'XOM','PG','MA','HD','CVX','ABBV','KO','PEP','COST','AVGO',
    'MRK','BAC','ADBE','CSCO','CRM','WMT','MCD','TMO','ACN','LIN',
    'LLY','NFLX','AMD','INTC','QCOM','TXN','ORCL','IBM','CAT','GE',
    'NKE','DIS','PM','LOW','SPGI'
]

train_years = 5
test_years = 5
n_rolls = 2

min_train_obs = 36
min_test_obs = 24
min_cs_stocks = 10

today = pd.Timestamp.today().normalize()
start_date = "2000-01-01"
end_date = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

print("Downloading stock prices...")
price = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False,
    threads=False
)['Close']

ret = price.resample('ME').last().pct_change().dropna(how='all')
ret = ret.dropna(axis=1, thresh=int(len(ret) * 0.8))

print("Downloading FF3 factors...")
url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors.CSV"
raw = requests.get(
    url,
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=30
).content.decode("utf-8", errors="ignore")

ff = pd.read_csv(io.StringIO(raw), skiprows=3)
ff.columns = ['Date', 'MKT', 'SMB', 'HML', 'RF']
ff['Date'] = ff['Date'].astype(str).str.strip()
ff = ff[ff['Date'].str.len() == 6]
ff['Date'] = pd.to_datetime(ff['Date'], format='%Y%m') + pd.offsets.MonthEnd(0)
ff = ff.set_index('Date').apply(pd.to_numeric, errors='coerce') / 100

df = ret.join(ff[['MKT', 'SMB', 'HML', 'RF']], how='inner').dropna(subset=['MKT', 'SMB', 'HML', 'RF'])
stock_cols = [c for c in ret.columns if c in df.columns]

print("Merged data shape:", df.shape)
print("Stocks available after cleaning:", len(stock_cols))
print("Latest month in merged data:", df.index.max().strftime("%Y-%m-%d"))


def run_one_roll_grouped(df, stock_cols, train_start_idx,
                         train_months=60, test_months=60,
                         min_train_obs=36, min_test_obs=24,
                         min_cs_stocks=10):

    train_end_idx = train_start_idx + train_months - 1
    test_start_idx = train_end_idx + 1
    test_end_idx = test_start_idx + test_months - 1

    if test_end_idx >= len(df.index):
        return None

    train_idx = df.index[train_start_idx:train_end_idx + 1]
    test_idx = df.index[test_start_idx:test_end_idx + 1]

    train = df.loc[train_idx].copy()
    test = df.loc[test_idx].copy()

    beta_rows = []

    for s in stock_cols:
        if train[s].notna().sum() < min_train_obs:
            continue
        if test[s].notna().sum() < min_test_obs:
            continue

        reg = pd.concat([
            (train[s] - train['RF']).rename('excess_ret'),
            sm.add_constant(train[['MKT', 'SMB', 'HML']])
        ], axis=1).dropna()

        if len(reg) < min_train_obs:
            continue

        y = reg['excess_ret']
        X = reg[['const', 'MKT', 'SMB', 'HML']]

        try:
            model = sm.OLS(y, X).fit()
            beta_rows.append({
                'stock': s,
                'beta_MKT': model.params['MKT'],
                'beta_SMB': model.params['SMB'],
                'beta_HML': model.params['HML'],
                'alpha_ts': model.params['const'],
                'R2_ts': model.rsquared
            })
        except Exception:
            continue

    if len(beta_rows) < 9:
        return None

    betas = pd.DataFrame(beta_rows).set_index('stock').sort_values('beta_MKT')

    n = len(betas)
    g1 = n // 3
    g2 = n // 3
    g3 = n - g1 - g2

    low_stocks = betas.index[:g1].tolist()
    mid_stocks = betas.index[g1:g1+g2].tolist()
    high_stocks = betas.index[g1+g2:].tolist()

    group_map = {}
    for s in low_stocks:
        group_map[s] = 'Low'
    for s in mid_stocks:
        group_map[s] = 'Mid'
    for s in high_stocks:
        group_map[s] = 'High'

    betas['group'] = betas.index.map(group_map)

    group_ret = pd.DataFrame(index=test.index)
    group_excess = pd.DataFrame(index=test.index)

    for gname, glist in [('Low', low_stocks), ('Mid', mid_stocks), ('High', high_stocks)]:
        group_ret[gname] = test[glist].mean(axis=1)
        group_excess[gname] = group_ret[gname] - test['RF']

    group_ret['H_minus_L'] = group_ret['High'] - group_ret['Low']
    group_excess['H_minus_L'] = group_excess['High'] - group_excess['Low']

    ttest_rows = []
    for col in ['Low', 'Mid', 'High', 'H_minus_L']:
        s = group_excess[col].dropna()

        if len(s) >= 2:
            t_stat, p_val = ttest_1samp(s, popmean=0.0, nan_policy='omit')
        else:
            t_stat, p_val = np.nan, np.nan

        ttest_rows.append({
            'group': col,
            'mean_monthly_return': group_ret[col].mean(),
            'mean_monthly_excess_return': s.mean(),
            'monthly_vol': group_ret[col].std(),
            't_stat_excess_mean_eq_0': t_stat,
            'p_value': p_val,
            'n_months': len(s)
        })

    ttest_table = pd.DataFrame(ttest_rows).set_index('group')

    fm_rows = []

    for dt in test.index:
        y_month = (test.loc[dt, betas.index] - test.loc[dt, 'RF']).dropna()

        if len(y_month) < min_cs_stocks:
            continue

        X_month = sm.add_constant(
            betas.loc[y_month.index, ['beta_MKT', 'beta_SMB', 'beta_HML']]
        )

        fm_data = pd.concat([
            y_month.rename('excess_ret'),
            X_month
        ], axis=1).dropna()

        if len(fm_data) < min_cs_stocks:
            continue

        try:
            fm_model = sm.OLS(
                fm_data['excess_ret'],
                fm_data[['const', 'beta_MKT', 'beta_SMB', 'beta_HML']]
            ).fit()

            fm_rows.append({
                'Date': dt,
                'lambda_0': fm_model.params['const'],
                'lambda_MKT': fm_model.params['beta_MKT'],
                'lambda_SMB': fm_model.params['beta_SMB'],
                'lambda_HML': fm_model.params['beta_HML'],
                'R2_fm': fm_model.rsquared,
                'n_stocks_cs': len(fm_data)
            })
        except Exception:
            continue

    if len(fm_rows) > 0:
        fm_lambdas = pd.DataFrame(fm_rows)

        avg_FM_R2 = fm_lambdas['R2_fm'].mean()
        median_FM_R2 = fm_lambdas['R2_fm'].median()

        lambda_test_rows = []
        for col in ['lambda_0', 'lambda_MKT', 'lambda_SMB', 'lambda_HML']:
            s = fm_lambdas[col].dropna()

            if len(s) >= 2:
                t_stat, p_val = ttest_1samp(s, popmean=0.0, nan_policy='omit')
            else:
                t_stat, p_val = np.nan, np.nan

            lambda_test_rows.append({
                'factor': col,
                'mean_lambda': s.mean(),
                't_stat_mean_eq_0': t_stat,
                'p_value': p_val,
                'n_months': len(s)
            })

        fm_ttest_table = pd.DataFrame(lambda_test_rows).set_index('factor')
    else:
        fm_lambdas = pd.DataFrame(columns=[
            'Date', 'lambda_0', 'lambda_MKT', 'lambda_SMB', 'lambda_HML', 'R2_fm', 'n_stocks_cs'
        ])
        avg_FM_R2 = np.nan
        median_FM_R2 = np.nan
        fm_ttest_table = pd.DataFrame(columns=[
            'mean_lambda', 't_stat_mean_eq_0', 'p_value', 'n_months'
        ])

    cum_return = (1 + group_ret).cumprod() - 1
    final_cum_return = cum_return.iloc[-1]

    r2_summary = pd.DataFrame({
        'avg_R2_ts': betas.groupby('group')['R2_ts'].mean(),
        'median_R2_ts': betas.groupby('group')['R2_ts'].median(),
        'min_R2_ts': betas.groupby('group')['R2_ts'].min(),
        'max_R2_ts': betas.groupby('group')['R2_ts'].max(),
        'n_stocks': betas.groupby('group').size()
    })

    overall_r2 = {
        'overall_avg_R2_ts': betas['R2_ts'].mean(),
        'overall_median_R2_ts': betas['R2_ts'].median(),
        'overall_min_R2_ts': betas['R2_ts'].min(),
        'overall_max_R2_ts': betas['R2_ts'].max()
    }

    group_summary = ttest_table.copy()
    group_summary['final_cumulative_return'] = final_cum_return.reindex(group_summary.index)

    beta_group_summary = betas.groupby('group')[['beta_MKT', 'beta_SMB', 'beta_HML', 'alpha_ts', 'R2_ts']].mean()
    beta_group_summary['n_stocks'] = betas.groupby('group').size()

    roll_info = {
        'train_start': train.index.min(),
        'train_end': train.index.max(),
        'test_start': test.index.min(),
        'test_end': test.index.max(),
        'n_stocks_total': len(betas),
        'n_low': len(low_stocks),
        'n_mid': len(mid_stocks),
        'n_high': len(high_stocks),
        'overall_avg_R2_ts': overall_r2['overall_avg_R2_ts']
    }

    return {
        'roll_info': roll_info,
        'betas': betas,
        'low_stocks': low_stocks,
        'mid_stocks': mid_stocks,
        'high_stocks': high_stocks,
        'group_return_monthly': group_ret,
        'group_excess_monthly': group_excess,
        'group_cum_return': cum_return,
        'group_summary': group_summary,
        'beta_group_summary': beta_group_summary,
        'ttest_table': ttest_table,
        'r2_summary': r2_summary,
        'overall_r2': overall_r2,
        'fm_lambdas': fm_lambdas,
        'fm_ttest_table': fm_ttest_table,
        'avg_FM_R2': avg_FM_R2,
        'median_FM_R2': median_FM_R2
    }

train_months = train_years * 12
test_months = test_years * 12
window_months = train_months + test_months

last_possible_start = len(df) - window_months
if last_possible_start < 0:
    raise ValueError("資料不足以做 5 年 train + 5 年 test")

start_positions = [last_possible_start - 12, last_possible_start]
start_positions = [s for s in start_positions if s >= 0]

all_results = []
for i, start_pos in enumerate(start_positions, start=1):
    out = run_one_roll_grouped(
        df=df,
        stock_cols=stock_cols,
        train_start_idx=start_pos,
        train_months=train_months,
        test_months=test_months,
        min_train_obs=min_train_obs,
        min_test_obs=min_test_obs,
        min_cs_stocks=min_cs_stocks
    )

    if out is None:
        print(f"Roll {i}: failed")
        continue

    out['roll_id'] = i
    all_results.append(out)

summary_rows = []
for r in all_results:
    info = r['roll_info']
    gs = r['group_summary']
    or2 = r['overall_r2']

    row = {
        'roll_id': r['roll_id'],
        'train_start': info['train_start'],
        'train_end': info['train_end'],
        'test_start': info['test_start'],
        'test_end': info['test_end'],
        'n_stocks_total': info['n_stocks_total'],
        'n_low': info['n_low'],
        'n_mid': info['n_mid'],
        'n_high': info['n_high'],

        'overall_avg_R2_ts': or2['overall_avg_R2_ts'],
        'overall_median_R2_ts': or2['overall_median_R2_ts'],

        'Low_avg_R2_ts': r['r2_summary'].loc['Low', 'avg_R2_ts'],
        'Mid_avg_R2_ts': r['r2_summary'].loc['Mid', 'avg_R2_ts'],
        'High_avg_R2_ts': r['r2_summary'].loc['High', 'avg_R2_ts'],

        'Low_mean_excess': gs.loc['Low', 'mean_monthly_excess_return'],
        'Mid_mean_excess': gs.loc['Mid', 'mean_monthly_excess_return'],
        'High_mean_excess': gs.loc['High', 'mean_monthly_excess_return'],
        'HML_mean_excess': gs.loc['H_minus_L', 'mean_monthly_excess_return'],

        'Low_t': gs.loc['Low', 't_stat_excess_mean_eq_0'],
        'Mid_t': gs.loc['Mid', 't_stat_excess_mean_eq_0'],
        'High_t': gs.loc['High', 't_stat_excess_mean_eq_0'],
        'HML_t': gs.loc['H_minus_L', 't_stat_excess_mean_eq_0'],

        'Low_p': gs.loc['Low', 'p_value'],
        'Mid_p': gs.loc['Mid', 'p_value'],
        'High_p': gs.loc['High', 'p_value'],
        'HML_p': gs.loc['H_minus_L', 'p_value'],

        'Low_final_cumret': gs.loc['Low', 'final_cumulative_return'],
        'Mid_final_cumret': gs.loc['Mid', 'final_cumulative_return'],
        'High_final_cumret': gs.loc['High', 'final_cumulative_return'],
        'HML_final_cumret': gs.loc['H_minus_L', 'final_cumulative_return'],

        'avg_FM_R2_test': r['avg_FM_R2'],
        'median_FM_R2_test': r['median_FM_R2'],

        'lambda0_mean': r['fm_ttest_table'].loc['lambda_0', 'mean_lambda'] if 'lambda_0' in r['fm_ttest_table'].index else np.nan,
        'lambda0_t': r['fm_ttest_table'].loc['lambda_0', 't_stat_mean_eq_0'] if 'lambda_0' in r['fm_ttest_table'].index else np.nan,
        'lambda0_p': r['fm_ttest_table'].loc['lambda_0', 'p_value'] if 'lambda_0' in r['fm_ttest_table'].index else np.nan,

        'lambda_MKT_mean': r['fm_ttest_table'].loc['lambda_MKT', 'mean_lambda'] if 'lambda_MKT' in r['fm_ttest_table'].index else np.nan,
        'lambda_MKT_t': r['fm_ttest_table'].loc['lambda_MKT', 't_stat_mean_eq_0'] if 'lambda_MKT' in r['fm_ttest_table'].index else np.nan,
        'lambda_MKT_p': r['fm_ttest_table'].loc['lambda_MKT', 'p_value'] if 'lambda_MKT' in r['fm_ttest_table'].index else np.nan,

        'lambda_SMB_mean': r['fm_ttest_table'].loc['lambda_SMB', 'mean_lambda'] if 'lambda_SMB' in r['fm_ttest_table'].index else np.nan,
        'lambda_SMB_t': r['fm_ttest_table'].loc['lambda_SMB', 't_stat_mean_eq_0'] if 'lambda_SMB' in r['fm_ttest_table'].index else np.nan,
        'lambda_SMB_p': r['fm_ttest_table'].loc['lambda_SMB', 'p_value'] if 'lambda_SMB' in r['fm_ttest_table'].index else np.nan,

        'lambda_HML_mean': r['fm_ttest_table'].loc['lambda_HML', 'mean_lambda'] if 'lambda_HML' in r['fm_ttest_table'].index else np.nan,
        'lambda_HML_t': r['fm_ttest_table'].loc['lambda_HML', 't_stat_mean_eq_0'] if 'lambda_HML' in r['fm_ttest_table'].index else np.nan,
        'lambda_HML_p': r['fm_ttest_table'].loc['lambda_HML', 'p_value'] if 'lambda_HML' in r['fm_ttest_table'].index else np.nan,
    }

    summary_rows.append(row)

summary_report = pd.DataFrame(summary_rows)

if len(summary_report) > 0:
    for c in ['train_start', 'train_end', 'test_start', 'test_end']:
        summary_report[c] = pd.to_datetime(summary_report[c]).dt.strftime('%Y-%m-%d')


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 240)

print("\n" + "=" * 120)
print("SUMMARY REPORT")
print("=" * 120)
print(summary_report.round(4))

for r in all_results:
    print("\n" + "=" * 120)
    print(f"ROLL {r['roll_id']}")
    print("=" * 120)

    info = r['roll_info']
    print(f"Train period : {info['train_start'].strftime('%Y-%m-%d')} ~ {info['train_end'].strftime('%Y-%m-%d')}")
    print(f"Test period  : {info['test_start'].strftime('%Y-%m-%d')} ~ {info['test_end'].strftime('%Y-%m-%d')}")
    print(f"Total stocks : {info['n_stocks_total']}")
    print(f"Low/Mid/High : {info['n_low']} / {info['n_mid']} / {info['n_high']}")
    print(f"Overall avg R^2_ts (train) : {r['overall_r2']['overall_avg_R2_ts']:.4f}")
    print(f"Average FM R^2 (test)      : {r['avg_FM_R2']:.4f}")
    print(f"Median FM R^2 (test)       : {r['median_FM_R2']:.4f}")

    print("\n[R2 Summary - Train Time Series]")
    print(r['r2_summary'].round(4))

    print("\n[Beta Group Summary]")
    print(r['beta_group_summary'].round(4))

    print("\n[T-test Table - Portfolio Excess Return]")
    print(r['ttest_table'].round(4))

    print("\n[Test FM Lambda T-test Table]")
    print(r['fm_ttest_table'].round(4))

    print("\n[Beta + Group Table]")
    print(r['betas'].round(4))

    print("\n[First 10 Test FM Lambdas]")
    print(r['fm_lambdas'].head(10).round(4))

    print("\n[Low Beta Stocks]")
    print(r['low_stocks'])

    print("\n[Mid Beta Stocks]")
    print(r['mid_stocks'])

    print("\n[High Beta Stocks]")
    print(r['high_stocks'])


if len(all_results) > 0:
    print("\n" + "=" * 120)
    print("AVERAGE OF 2 ROLLS: TEST FM RESULTS")
    print("=" * 120)

    avg_fm_r2_across_rolls = np.mean([r['avg_FM_R2'] for r in all_results])
    print(f"Average of roll-level avg_FM_R2 = {avg_fm_r2_across_rolls:.4f}")

    fm_all = pd.concat(
        [r['fm_lambdas'].assign(roll_id=r['roll_id']) for r in all_results],
        ignore_index=True
    )

    if len(fm_all) > 0:
        print(f"Overall mean monthly FM R^2 across all test months = {fm_all['R2_fm'].mean():.4f}")

        avg_rows = []
        for col in ['lambda_0', 'lambda_MKT', 'lambda_SMB', 'lambda_HML']:
            s = fm_all[col].dropna()
            if len(s) >= 2:
                t_stat, p_val = ttest_1samp(s, popmean=0.0, nan_policy='omit')
            else:
                t_stat, p_val = np.nan, np.nan

            avg_rows.append({
                'factor': col,
                'mean_lambda_across_2rolls': s.mean(),
                't_stat': t_stat,
                'p_value': p_val,
                'n_months': len(s)
            })

        avg_fm_ttest = pd.DataFrame(avg_rows).set_index('factor')

        print("\n[Average FM Lambda T-test Across 2 Rolls]")
        print(avg_fm_ttest.round(4))


if len(all_results) > 0:
    with pd.ExcelWriter("beta_group_2roll_full_final_report.xlsx", engine="openpyxl") as writer:
        summary_report.to_excel(writer, sheet_name="summary_report", index=False)

        for r in all_results:
            rid = r['roll_id']
            r['betas'].to_excel(writer, sheet_name=f"roll{rid}_betas")
            r['group_summary'].to_excel(writer, sheet_name=f"roll{rid}_group_summary")
            r['beta_group_summary'].to_excel(writer, sheet_name=f"roll{rid}_beta_group_summary")
            r['ttest_table'].to_excel(writer, sheet_name=f"roll{rid}_ttest")
            r['r2_summary'].to_excel(writer, sheet_name=f"roll{rid}_r2_summary")
            r['fm_lambdas'].to_excel(writer, sheet_name=f"roll{rid}_fm_lambdas", index=False)
            r['fm_ttest_table'].to_excel(writer, sheet_name=f"roll{rid}_fm_ttest")
            r['group_return_monthly'].to_excel(writer, sheet_name=f"roll{rid}_monthly_ret")
            r['group_excess_monthly'].to_excel(writer, sheet_name=f"roll{rid}_monthly_excess")
            r['group_cum_return'].to_excel(writer, sheet_name=f"roll{rid}_cum_ret")

    print("\nExcel saved: beta_group_2roll_full_final_report.xlsx")
else:
    print("\nNo valid results.")