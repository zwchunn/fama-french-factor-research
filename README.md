# Fama-French Three-Factor & Fama-MacBeth Research

本檔案使用 Python 研究三因子流程，分析個股對市場、規模與價值因子的曝險，並透過樣本外測試與 Fama-MacBeth 橫斷面迴歸檢驗因子風險溢酬。

1. 下載股票價格並轉換為月報酬。
2. 下載 Fama-French 三因子與無風險利率資料。
3. 對齊日期、處理缺失值，建立完整的研究資料集。
4. 使用訓練期資料進行時間序列 OLS，估計每檔股票的 MKT、SMB 與 HML Beta。
5. 依市場 Beta 將股票分成 Low、Mid、High 三組，在測試期比較報酬表現。
6. 在測試期逐月執行 Fama-MacBeth 橫斷面迴歸，估計各因子的風險溢酬。
7. 使用 t 值、p 值與 R-squared 檢查統計顯著性與模型解釋力。
8. 採用五年訓練期、五年測試期與兩次滾動視窗進行樣本外驗證。
