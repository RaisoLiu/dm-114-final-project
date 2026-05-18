好，這題要用「競賽紀律」做，不要只當作一個普通回歸 homework。核心判斷是：這是 **91 天多變量氣象序列 → 未來 5 週乾旱嚴重度** 的 time-series tabular forecasting。評分是 **MAE，越低越好**；label `score` 是 0 到 5，且只有每週一天有值，其他 6 天是 NaN。test 每個 region 給 91 天氣象資料，沒有 `score`，submission 是每個 region 輸出 `pred_week1` 到 `pred_week5`。這些規則都在你上傳的課程投影片第 5–11 頁。

我先查了你給的 Kaggle link，但目前從外部會被導到 Kaggle login，所以我看不到即時 leaderboard 排名；下面策略主要根據課程投影片與競賽設計來規劃。([Kaggle][1])

第一件事：你們的目標不是「做一個模型」，而是做出一套 **可重現、抗 private leaderboard、可寫進報告的實驗系統**。這門課的分數配置很明確：Final Project 35 分，其中 Kaggle & Report 30 分、Progress Check 5 分；而 Kaggle & Report 裡面 Kaggle 佔 60%、Report 佔 40%。換算後，Kaggle public 只有 5.4 分，private 是 12.6 分，report 是 12 分。這代表 report 幾乎跟 private leaderboard 一樣重要，不要最後兩天才寫。

你們應該採取的主線是這樣：先建立一個嚴格的 validation，再用它驅動所有模型決策。不要 random split。這種資料是時間序列，而且每個 region 有連續 5,480 天 train data；如果 random split，模型會偷看到鄰近時間點與同 region 的未來資訊，public 分數可能好看，但 private 會爛。正確做法是模擬 test：對每個 region，用過去 91 天氣象資料建立一筆樣本，target 是接下來 5 個 weekly score。validation 應該用「時間切割」，例如每個 region 最後一段時間做 validation，前面時間做 training；也可以做 3–5 個 rolling folds，讓每一 fold 都是「用過去預測未來」。

模型先不要急著上 deep learning。這題最強、最穩的第一主力通常會是 **feature engineering + gradient boosting regression**。把 91 天資料壓成 tabular features，然後對 5 個 week 分別訓練 5 個模型，也就是 week1 model、week2 model、week3 model、week4 model、week5 model。這通常比一個模型同時輸出 5 個值更穩，因為 week1 和 week5 的預測難度、資訊衰減、target 分布都不一樣。

你們的 baseline 要至少有四個。第一個是 global mean/median baseline：所有 training label 的中位數或平均數。第二個是 region-level baseline：每個 region 過去 score 的平均或中位數。第三個是 seasonal baseline：依照 month、week-of-year、region 的歷史中位數。第四個是 model baseline：Random Forest、XGBoost、LightGBM、CatBoost 或 sklearn HistGradientBoosting 任一種。報告裡這些 baseline 很有用，因為投影片要求 experiments 可以包含 self-defined baselines、ablation、parameter analysis、case study。

feature engineering 是勝負點。每個氣象欄位都做 7、14、28、56、91 天視窗統計：mean、std、min、max、median、q10、q90、last、first、slope。乾旱不是單日事件，它有累積性，所以 precipitation 特別重要：做 `prec_sum_7/14/28/56/91`、無雨天數、最長連續無雨天數、最近 28 天降雨量減去過去 91 天平均、最大單日降雨。溫度也要做壓力特徵：`tmp_max_mean`、`tmp_mean_28 - tmp_mean_91`、`tmp - dp_tmp`、`tmp - wb_tmp`。濕度、surface pressure、wind 也做一樣的 rolling stats。最後加上 date features：month、day-of-year、sin/cos seasonality。`region_id` 一定要進模型，可以 one-hot、label encoding，或用 CatBoost categorical。

很重要：不要把 `score` 的 NaN 當成 0。投影片明確說 score 是 weekly recorded，其他 6 天是 NaN；那些 NaN 不是「沒有災害」，只是「沒有標籤」。訓練 target 只能用非 NaN 的 score row。這是第一個常見爆炸點。

第二個爆炸點是 label leakage。你們在 training 建立 91 天 input window 時，不能把未來 35 天的氣象資料、未來 score、validation future 的統計量混進 feature。所有 feature 都必須只來自該筆樣本的 91 天 input window，以及明確允許使用的歷史資料。因為 test.csv 只有 91 天 input，沒有 score column，所以模型主線應該先做「只用氣象資料 + region/date」的版本。歷史 score lag 可以做成實驗，但要非常小心：如果 test inference 時拿不到同樣形式的 score lag，就不要讓它進正式 submission。

第三個爆炸點是 public leaderboard overfitting。投影片說 daily limit 是每隊每天最多 3 次 submission，reset 是 UTC+0，也就是台灣早上 8 點；private leaderboard competition 結束後才會看到，且預設 public leaderboard 最好的兩個 submission 會用於 private leaderboard。這表示你們不要亂丟純測試 LB 的檔案。每次 submission 前要問：這個檔案是 validation 真的變好，還是只是猜 public？如果只是猜 public，不要上傳。

模型策略我會這樣排優先級。第一版用 tabular GBDT，5 個 horizon 各自訓練。objective 儘量用 MAE/L1；如果模型不支援，就用 squared error 先做 baseline，再看 validation。預測結果一律 clip 到 `[0, 5]`，因為真實 score 範圍就是 0 到 5，超出範圍只會增加 MAE。不要一開始就 round 成整數，雖然 label 是整數，但投影片允許 float prediction；對 MAE 來說，float ensemble 常常更好。rounding、smoothing、ordinal calibration 都要用 validation 決定，不要憑感覺。

第二版做 ensemble。你們可以訓練 LightGBM-like model、CatBoost-like model、RandomForest/ExtraTrees-like model，各自用不同 seed、不同 feature subset，最後依 validation MAE 加權平均。MAE 下 ensemble 很有效，因為它降低 variance。若有時間，再加 sequence model，例如 1D CNN、GRU、Transformer encoder，直接吃 91×features 的序列；但這應該是加分支線，不是主線。除非你們 tabular baseline 已經很強，否則 deep learning 只會吃時間。

你們的實作 pipeline 應該長這樣：`make_windows.py` 負責從 train 產生 supervised samples；`features.py` 負責 rolling/statistical features；`train.py` 負責 time-based CV 和模型訓練；`predict.py` 負責讀 test.csv、對每個 region 產生 5 週 prediction；`make_submission.py` 負責按照 sample_submission.csv 的格式輸出。README 要清楚寫如何重現：環境、安裝、資料放哪裡、訓練指令、推論指令、會產生哪個 submission 檔。這不是形式主義，投影片明確說 TA 會測 code 能不能跑，code、report、Kaggle results 必須一致且可重現；若不一致或不可執行，會是 0 分。

你們第一週的目標應該是：完成 EDA、validation、四個 baseline、第一個 Kaggle submission。不要一開始就調模型。EDA 至少要回答這些問題：每個 region 的日期是否連續？每個 region 是否剛好 5,480 天？score 出現在哪個 weekday 或每隔幾天？score 分布是否嚴重偏 0/1？不同 region 的 score 分布差多少？test 的日期是否接在 train 後面？test region 是否完全都出現在 train？meteorological columns 是否有 missing/outlier？`wind_range` 是否等於 `wind_max - wind_min`、`tmp_range` 是否等於 `tmp_max - tmp_min`？這些檢查會直接決定 feature 和 validation 的正確性。

進度檢查在 5 月 21 日，每組 5 分鐘 presentation，不用提前寄 slides，但要帶自己的 laptop；評分看你們展示的 progress 完整度。5 分鐘版本建議只講五張投影片：problem & metric、data observations、validation design、baseline results、current best model & next steps。不要講空泛方法，TA 要看到你們真的跑過東西。

報告要從第一天開始寫。投影片要求英文、5–8 頁，不含 references，格式用 Overleaf IEEE template 或 IEEE Word A4 template，不能改基本格式。內容要有 abstract，且 abstract 要包含 group ID 和 GitHub link；Project Summary 最多 1 頁；Proposed Method 1–3 頁；Experiments 2–4 頁。你們應該把 validation design、feature groups、model comparison、ablation study 寫得很清楚。最有說服力的 ablation 是：baseline → +rolling weather stats → +precipitation dry-spell features → +seasonality/date → +region encoding → +ensemble → +post-processing。

另外，references 千萬不要亂寫。投影片說可以使用 AI tools，但 plagiarism 會 0 分；report 裡若有 hallucinated references，也會 0 分。你們寧可只引用少量真實、確定存在的 time-series forecasting、gradient boosting、drought prediction 相關文獻，也不要塞一堆看起來高級但不存在的 citation。

Kaggle 行政細節也不要出錯。team name 要改成 `Team {Group ID}`，例如 `Team 17`；投影片最後還強調 Kaggle Display Name 必須是 `Team {Group ID}`，否則成績會記 0。deadline 是 6 月 10 日 11:55 PM，late submission 不收；code 要放 public GitHub repo 並在 deadline 前 commit；report 檔名要是 `DM_project_Group_{GroupID}.pdf`，並且透過 E3 交。

我會給你們一個明確的執行順序：

先在 24 小時內做出可跑的 baseline submission。目標不是高分，是確認資料 alignment、submission format、Kaggle pipeline 都正確。

接著在 48–72 小時內建立 time-based CV。沒有可靠 CV，就沒有可靠競賽策略。你們所有後續實驗都要記錄 CV MAE、public MAE、feature set、model、seed、submission filename。

第三步開始堆 feature。先做 precipitation、temperature、humidity 的多視窗統計，再做 region/date encoding，再做 dry spell 和 trend features。

第四步訓練 5 個 horizon-specific models。week1 到 week5 分開看 validation MAE，因為 week5 通常會比 week1 難。報告中也可以畫出 per-horizon MAE，這很加分。

第五步做 ensemble 和 post-processing。ensemble 後 clip 到 `[0,5]`；rounding 只在 validation 證明有效時使用。若預測出現不合理劇烈跳動，可以測試 week-to-week smoothing，但一樣只能看 validation 決定。

最後一週不要再大幅改 pipeline。只做穩定性檢查、report、README、final submission。這門課的風險不是只輸在模型，而是輸在格式、reproducibility、team name、report inconsistency、fake citation。這些全部可以避免。

[1]: https://www.kaggle.com/competitions/data-mining-2026-final-project/leaderboard "Kaggle: Your Home for Data Science"
