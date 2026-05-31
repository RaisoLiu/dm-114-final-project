# 實驗清單、困境清單與關聯矩陣

這是濃縮版的「我們做了什麼、卡在哪、兩者怎麼對應」單頁文件，給寫報告的隊友用。詳細版本在 `docs/JOURNEY.md` 與 `docs/memory/`。

最終 team best：**ext150 = 0.8534**（`extrapolate_150` 轉換 on pb30，於 2026-05-12 上傳）。三次後續嘗試（Plan v7，2026-05-18）皆未突破。

---

## 一、實驗清單（時間序）

每一列：方法 → 公開分數 → 一句話結論。

| # | 階段 | 實驗 | 方法 | Public MAE | 結論 |
|---|---|---|---|---|---|
| E1 | v1–v2 | **ext150 baseline** | LightGBM+HGB 1071 features，pb30 加 `extrapolate_150` 轉換 | **0.8534** | 團隊基準線；之後所有方法的天花板 |
| E2 | v2 | pb30 redo_blend | 70% multi-slice LGBM/HGB ensemble + 30% public-best | 0.8593 | 個人最好；同家族 |
| E3 | v3 | Latent Nowcast Round 2 | 4-leg ensemble 加 latent nowcast leg | 0.8609 | 同家族；多 leg 沒幫助 |
| E4 | v4 | 9-leg MLP stacker | 9 個 GBDT leg + MLP stacker（val 0.310） | 0.8688 | val→public 嚴重背離首例 |
| E5 | v5 | Deep CNN | 91-day weather window，bounded sigmoid + L1（val 0.343） | 0.9706 | H1 71% 預測飽和到 5.0，cutoff_age OOD |
| E6 | v5 | Deep LSTM | 同上 wrapper（val 0.40） | 0.9032 | fold 4 outlier 0.89，GroupKFold 結構洩漏 |
| E7 | v5 | Deep Transformer | 同上 wrapper | — | refit-all 階段 output collapse 到 0，無法上傳 |
| E8 | v5 | 6-leg deep ensemble 10% blend | (CNN×2 + LSTM×4) × 10% + ext150 × 90% | 0.8767 | slope +0.37；deep family 永久關閉 |
| E9 | v6 | v6 E1 adversarial-val LGBM | LightGBM 改用 adversarial-val anchor selection | 0.8886 | val 0.44 但 public worse；adversarial 不夠 |
| E10 | v6 | v6 3-leg shifted w15 | 3 個 adversarial-val LGBM/HGB，per-horizon mean-shift + 15% blend | 0.8853 | slope +0.56；adversarial-val LGBM family 關閉 |
| E11 | v6 | ext170 | extrapolate_170 轉換（比 ext150 更激進） | 0.9208 | sharpening 方向錯了；transform slope +0.73 |
| E12 | v7 U1 | DOY blend w08 | per-region DOY-cycle 訊號 × 8% + ext150 × 92% | 0.8848 | slope +0.48；DOY family 關閉 |
| E13 | v7 U2 | H2 Recent-Zero Mask α=0.20 | 對 576 個「最近 12 score 全 0 且 ext150_avg>1」的 region 把 ext150 壓低 20% | 0.8919 | slope +0.42；adversarial-val MAE 跌 0.031 但 public 漲 0.039 |
| E14 | v7 U3 | （同 E10） | （重新上傳作為 v7 portfolio 第 3 leg） | 0.8853 | 補做 v6 family 的 Kaggle datapoint |
| E15 | v17 | **External-data deanonymization** | climate fingerprint kNN + 91-day weather Pearson → 對應 cdminix US-drought dataset 的 FIPS county × 2019/2020；直接查 USDM 5-週後的真實 score | 0.1866（**排除**） | 2026-05-21 已上傳一次驗證 validation MAE 0.0951 確實 transfer 到 public 0.1866，但同日 team 決議此方法不正當，**不採用為最終結果**；Kaggle 無法刪除 submission，將於 deadline 前手動取消 private LB 的 selection。E15 不計入 team result，team best 維持為 E1 ext150 = 0.8534。 |

合計 15 種結構不同的方法（E14 是 E10 重傳；E15 後被排除）。**E1–E14 中 ext150 仍為最低的 0.8534**；E15 的 0.1866 為一次性 verification upload，已決議不採用。

---

## 二、困境清單（主題式）

每一項：核心觀察 → 影響 → 在哪個 memory 文件展開。

| # | 困境 | 核心觀察 | memory file |
|---|---|---|---|
| B1 | **val→public 系統性背離** | val MAE 下降不會轉移到 public。多次重現：stacker val 0.310→public 0.8688；CNN val 0.343→public 0.97；H2 val −0.031→public +0.039 | `feedback_stacker_overfit.md` |
| B2 | **cutoff_age 是 OOD 特徵** | `anchor_index − score_cutoff` 訓練值跨 0–730 天，測試集固定在 per-region public gap；GBDT 自動避開但 deep model 學壞 | `feedback_deep_test_distribution_shift.md` |
| B3 | **bounded sigmoid + L1 + Transformer 退化** | refit-all 階段 trans output collapse 到全 0；CNN/LSTM 同 wrapper 卻沒事 | `feedback_trans_bounded_sigmoid_collapse.md` |
| B4 | **GroupKFold(region) 結構洩漏** | 把同一 region 整個分到同 fold，但 DOY/anchor-age 結構跨 region 洩漏；fold-level 結果有 outlier（lstm fold 4 = 0.89 vs 其他 0.27） | `feedback_stacker_overfit.md` |
| B5 | **ext150 family 是天花板** | 任何用相同 weather-feature 空間的模型，誤差都與 ext150 正相關；blend 進去只加噪音 | `feedback_deep_blend_kaggle_confirmed.md` |
| B6 | **MAD-vs-public 正 slope 定律** | 5 筆獨立 datapoint：MAD>0 vs ext150 的候選，public ≥ 0.8534 + 0.42×MAD（最低 slope 也是正的） | `project_mad_slope.md` |
| B7 | **沒有任何候選達到 ρ<0** | DOY ρ≈+0.5、H2 ρ≈+0.99、v6 ρ≈+0.85；正向相關意味著 blend 只能放大誤差，不能修正 | `feedback_plan_v7_3upload_retro.md` |
| B8 | **train→test 18 個月 gap** | test data 沒有 score，距離最後一次觀測 score 中位數 538 天；naive persistence 失效 | `project_validation_slices.md` |
| B9 | **多年週期性 + 高嚴重度 plateau** | 合成資料有多年週期；高嚴重度集中在 delta {721-756}，但 val slice 取的是 728/735/742 → target mean 1.16 vs public 1.21 不匹配 | `project_validation_slices.md` |
| B10 | **Kaggle 配額 3/day** | 限制了 val→public 對齊速度；3 個 datapoint 才能粗略估計 slope | （無單獨 memory） |
| B11 | **外部資料是否允許未驗證** | 競賽規則未檢查，Plan v8 第 0 步必須做 | （無單獨 memory） |
| B12 | **transform vs blend slope 不對稱** | sharpening transform 的 slope 比 blend 高（ext170 +0.73 vs DOY blend +0.48）；ext170 更證實「再 sharpen」是錯方向 | `feedback_blend_vs_transform.md` |

---

## 三、關聯矩陣（實驗 × 困境）

每個 ✓ 代表該實驗實際遭遇了該困境（或結果證實了該困境的存在）。空白不代表無，只代表沒被這個實驗直接驗證。

|  | B1 val→pub | B2 cutoff_age | B3 trans collapse | B4 GroupKFold | B5 ext150 ceiling | B6 +slope | B7 no ρ<0 | B8 18m gap | B9 週期性 | B12 transform |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 ext150 | | | | | (anchor) | (anchor) | | ✓ | ✓ | (anchor) |
| E2 pb30 | | | | | ✓ | | | ✓ | | |
| E3 latent nowcast | | | | | ✓ | ✓ | | | | |
| E4 9-leg stacker | ✓ | | | ✓ | | | | | | |
| E5 CNN | ✓ | ✓ | | | | | | ✓ | | |
| E6 LSTM | ✓ | ✓ | | ✓ | | | | ✓ | | |
| E7 Transformer | | | ✓ | | | | | | | |
| E8 deep ensemble blend | | | | | ✓ | ✓ (+0.37) | ✓ | | | |
| E9 v6 E1 adv-val LGBM | ✓ | | | | ✓ | | | | ✓ | |
| E10 v6 3-leg w15 | | | | | ✓ | ✓ (+0.56) | ✓ | | | |
| E11 ext170 | | | | | | ✓ (+0.73) | | | | ✓ |
| E12 DOY blend w08 | | | | | | ✓ (+0.48) | ✓ | | ✓ | |
| E13 H2 Recent-Zero | ✓ | | | | ✓ | ✓ (+0.42) | ✓ | ✓ | | |

**讀法**：
- 每個 deep model（E5/E6）都同時被 B1（val→public 背離）和 B2（cutoff_age OOD）打到 — 不只是一個問題。
- 從 E8 之後（包含 v6 與 v7 三次上傳）都進入 **B6 + B7 雙重困境**：MAD 越大、誤差越正相關於 ext150 → 無路可走。
- E11（ext170）是「sharpening 方向錯」的關鍵反證 —— 證明 B12 並沒有讓你能繼續推進 ext150 的方向。

---

## 四、核心結論（給報告用）

1. **天花板存在且已實證**：13 個結構不同的方法都在 0.8534 之上。
2. **天花板的形狀是 slope law**：任何 MAD > 0 的候選，public ≥ 0.8534 + 0.42 × MAD（5 個 datapoint 完全沒有反例）。
3. **唯一突破路徑是 ρ < 0**：需要與 ext150 誤差負相關的候選；現有 weather + score + DOY 特徵空間裡找不到。
4. **val→public 系統性背離**：所有改善 val 的方法（stacker、deep models、H2）在 public 上都變差。這代表 val slice 結構性不像 public test。
5. **下一步必須出特徵空間**：Plan v8 草案規劃 4 類外部訊號（外部資料、預訓練模型、訓練典範、其他 Kaggle 解法）的文獻探索。

---

## 附錄：報告寫作建議

- 「方法 / 實驗」段：用表一直接列。
- 「困境 / 挑戰」段：用表二，按主題講；每項一段配 memory file 細節。
- 「結果分析」段：用表三的矩陣；指出「為什麼 13 個方法都失敗，是因為命中了同一組根本困境」。
- 「結論與未來工作」段：引用第四節 5 點；如要展開 future work，引 `docs/plans/plan_v8_external_training.md` 的 4 個 category。
- 數據引用：所有 public MAE 均來自 Kaggle 實測，可在 `reports/strategy_2026-05-11_redo.md` §12 與 `docs/memory/feedback_plan_v7_3upload_retro.md` 雙重交叉驗證。
