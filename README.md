# Fama-French Three-Factor & Fama-MacBeth Research

本專案使用 Python 建立一套美股因子研究流程，分析個股對市場、規模與價值因子的曝險，並透過樣本外測試與 Fama-MacBeth 橫斷面迴歸檢驗因子風險溢酬。

## Data

- 研究樣本：45 檔美國大型股票
- 股票資料：Yahoo Finance（透過 `yfinance` 下載調整後股價）
- 因子資料：Kenneth French Data Library
- 因子：MKT、SMB、HML 與無風險利率 RF
- 資料頻率：月資料

## Methodology

1. 下載股票價格並轉換為月報酬。
2. 下載 Fama-French 三因子與無風險利率資料。
3. 對齊日期、處理缺失值，建立完整的研究資料集。
4. 使用訓練期資料進行時間序列 OLS，估計每檔股票的 MKT、SMB 與 HML Beta。
5. 依市場 Beta 將股票分成 Low、Mid、High 三組，在測試期比較報酬表現。
6. 在測試期逐月執行 Fama-MacBeth 橫斷面迴歸，估計各因子的風險溢酬。
7. 使用 t 值、p 值與 R-squared 檢查統計顯著性與模型解釋力。
8. 採用五年訓練期、五年測試期與兩次滾動視窗進行樣本外驗證。

## Output

程式會在終端顯示各次滾動回測的 Beta 分組、報酬、統計檢定與 Fama-MacBeth 結果，並輸出：

```text
beta_group_2roll_full_final_report.xlsx
```

Excel 檔包含各次回測的 Beta、分組報酬、累積報酬、R-squared、Fama-MacBeth lambda 與 t 檢定結果。

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python fama_french_fama_macbeth.py
```

## Notes

- 股票與因子資料皆由公開來源於執行時下載。
- 回測結果可能隨資料更新時間而改變。
- 本專案為量化研究與程式實作展示，不構成投資建議。

## Author

Wei-Chun Chou（周煒鈞）
