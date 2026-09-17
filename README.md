<p align="center">
  <img src="data/logo/logo.png" alt="Euphonia" width="520">
</p>

# Euphonia

Windows 本機聲音角色與截圖朗讀工具。使用 Qwen/Qwen3-TTS-12Hz-0.6B-Base 進行 zero-shot voice cloning，RapidOCR 辨識圖片中的中英文。

## Audiobook Mode · 有聲小說模式

主視窗新增「有聲小說設定」：啟用模式、分別框選角色名稱與台詞區域，並建立角色名稱 → 既有 Voice 映射。設定會永久儲存；切回遊戲後按 **F11** 開始／停止背景監控，主視窗與系統匣顯示 ON/OFF。只有匹配角色且穩定、未重複的台詞才會依序播放，使用既有常駐 TTS 模型。監控開啟時，對話框旁的「OCR ＋ 播放」只辨識對話區域，直接使用主畫面目前選取的 Voice 優先播放，不需要角色名稱匹配或穩定等待；另一個浮動面板會顯示本次情緒、信心值及實際採用的 Sampling。F12 仍為手動框選截圖，使用時會先停止監控。

模型選單的 **Qwen3-TTS-streaming（標點預生成）** 使用同一套 GPU 加速 Qwen 權重。在有聲小說逐字顯示尚未停止時，半形／全形的逗號、句點、驚嘆號、問號及分號會立即形成 TTS 工作；OCR 仍持續追蹤後續文字，穩定時間到達後再提交剩餘尾段。所有引號及換行不會觸發分段。每個片段只生成與播放一次，依原文順序進入播放佇列。播放目前片段時會在 GPU 預先生成後續片段，並依設定縮短模型產生的邊界靜音、保留尾端停頓，減少片段之間等待推論造成的斷點。

「有聲小說設定」可調整 Streaming 靜音閾值（預設相對峰值 -38 dB）與片段間隔（預設 75 ms）。閾值越接近 -10 dB，越多微弱邊界聲音會視為靜音並裁掉；越接近 -80 dB 越保守。片段間隔決定有效聲音後保留的靜音長度，0 ms 可取得最緊密銜接。

片段間隔也會計入 OCR 仍在逐字增加的情況：上一段的語音與尾端間隔播放完成後，若尚未出現下一個分段標點，會直接提交目前累積的安全尾段，不再額外等待完整的文字穩定時間。英文保留正在增長的最後一個單字，避免從單字中央截斷；中文則提交目前可見內容。

Streaming 的每個獨立片段會在裁切後進行響度一致化：只量測有效語音區域並對齊 -18 dB RMS，峰值限制在 -1.5 dB，語音起迄另使用 8 ms 淡入／淡出。這可降低更換情緒 Sampling 時下一段突然變大，並避免非零振幅切換造成 click／爆音。

小說模式在送交 TTS 前會移除半形／全形單雙引號、彎引號及常見 CJK 引號，避免模型把引號讀成停頓或符號。

預設 OCR 間隔 300 ms、文字穩定 700 ms，可在設定調整。重新開啟軟體後監控保持 OFF。

情緒模板支援本機情緒路由：英文使用 INT8 ONNX DistilRoBERTa 分為 anger、disgust、fear、joy、neutral、sadness、surprise；中日韓文字改用 INT8 XLM-EMO 分為 anger、fear、joy、sadness。實際 OCR 後會先篩選相同 class，再以完整情緒分數向量的距離選擇最接近的參考音檔。分類模型在 CPU 執行，候選 TTS 角色特徵在預載階段建立；可執行 `scripts/setup_emotion.py` 安裝分類器。

## 切換語音模型

左側「語音模型」保留 Qwen3-TTS 0.6B、Qwen3-TTS-streaming、F5-TTS v1 與 ZipVoice-Distill。角色音檔與逐字稿共用，不必重新建立角色。選擇會記住；F5 與 ZipVoice 支援中文、英文。工作進行中暫停切換，切換後會釋放前一模型。

四個後端使用隔離的 Python 環境與常駐子程序。開啟工具或切換模型後，會依目前角色自動載入並暖機；看到「模型與角色已就緒」後，生成會重用模型與角色快取。停止會立即停播，等待當次推論結束並丟棄結果，保留模型供下一次生成。視窗縮到系統匣時保留模型；切換模型或完全離開時釋放。

自動截圖朗讀完成後，狀態列顯示 OCR、TTS 與合計時間，明細儲存在 `data/last-pipeline.json`。此處合計從框選完成、收到圖片起算至要求播放；不含手動框選時間及音訊裝置延遲。

安裝模型後端可執行 `powershell -ExecutionPolicy Bypass -File setup_models.ps1`；加上 `-Model faster`、`qwen_streaming`、`f5` 或 `zipvoice` 可只安裝一個。模型較大，首次安裝需要下載時間與額外磁碟空間。

- 加速版 Qwen 共用 `data/model/`，使用 CUDA Graphs。
- F5-TTS v1 與 ZipVoice-Distill 使用 16 步與 4 步 PyTorch CUDA 推論，目前尚未使用 TensorRT。
- 預設勾選「快速 OCR」：限制圖片最大邊長 960，使用 4 個運算執行緒，減少等待。若細小文字辨識不完整，可取消勾選切回原設定。兩者仍在本機辨識。
- 推論診斷：`data/logs/`。加速版與 Streaming 共用 `.venv-faster`；F5 與 ZipVoice 分別使用 `.venv-f5` 和 `.venv-zipvoice`，不要混裝套件。

執行 `.venv/Scripts/python.exe scripts/compare_models.py faster`（或 `qwen_streaming`、`f5`、`zipvoice`）可使用已建立的角色測試短、中、長英文，保存 `data/comparison-*.json` 和 WAV。再執行 `scripts/build_comparison_page.py` 可產生 `data/model-comparison.html` 離線試聽頁。時間以整段完成計算，不是第一段聲音的延遲。介面實際生成與播放測試：`.venv/Scripts/python.exe scripts/test_models_ui.py faster qwen_streaming f5 zipvoice`。

## 啟動

雙擊 `start.bat`。首次安裝請先執行 `setup.bat`（需要 Python 3.12 與 uv）。安裝程式使用專案內的 `.venv`；相容的 NVIDIA 顯示卡可使用 CUDA 加速，沒有 CUDA 時則使用較慢的 CPU。

1. 按「建立角色」，輸入名稱、選擇 3～30 秒的 WAV／FLAC／MP3／OGG，填寫與語音一致的逐字稿，儲存。建立後可按「編輯所選角色」修改名稱、主要採樣、逐字稿及情緒模板；Voice ID 保持不變，編輯前資料存入該角色的 `backups/`。
2. 選擇角色及朗讀語言。
3. 按 **F12** 或「框選截圖」，拖曳選取文字區域；Esc 取消。工具開啟期間，F12 在其他視窗或工具最小化時也可觸發；多螢幕可先在左側選擇螢幕。仍保留工具視窗內的 `Ctrl+Shift+S`。OCR／生成中按 F12 會取消目前工作，待其結束後開啟截圖，不再丟棄要求；為保留常駐模型，仍須等目前推論／載入結束。框選中或開啟對話框時不重複開啟框選，長按不會重複觸發。

F12 透過獨立訊息執行緒上的 Windows 鍵盤 hook 偵測；單獨按 F12 由 Euphonia 處理，避免同時觸發前景程式的 F12 功能。Ctrl／Shift／Alt／Win 搭配 F12 的組合不會被攔截。

另有 8ms 按鍵狀態備援，當 hook 被 Windows 移除時仍可接收按鍵，並於放開後重新建立 hook。備援接管的該次按鍵可能同時傳給前景程式。`data/logs/hotkey.log` 記錄啟動、截圖要求、忙碌狀態、框選開關及 hook 恢復，不記錄其他按鍵、OCR 文字或截圖內容。啟動錯誤另存 `data/logs/app.log`。更新後請从系統匣「結束 Euphonia」再重開 `start.bat`，執行中的舊版不會自動更新。

### 遊戲中背景使用

先選好角色與截圖螢幕，保持「截圖後自動辨識並播放」勾選，再按「背景執行（遊戲模式）」。進入遊戲後按 F12 框選，結束框選後會回到原本視窗，OCR 與播放都在背景完成。Esc 取消框選也會回到原視窗。

框選後會保留置頂的綠色浮動框。遊戲文字更新時，按框上的「截圖朗讀」即可重新擷取同一區域並自動辨識、播放；不必再次按 F12 拉框。截圖前會暫時隱藏框線與工具列，框內可照常操作原視窗。處理期間按鈕顯示「處理中…」，避免重複送出。需要改變範圍時按「重新框選」或 F12；按 Esc 會保留原範圍。「關閉框框」只隱藏浮動框，下次 F12 可重新框選。範圍固定在螢幕座標，不會自動跟隨遊戲視窗移動。

「背景執行（遊戲模式）」只會把目前同一個 Euphonia 縮到系統匣；雙擊系統匣圖示可重新開啟。右上角關閉按鈕及系統匣的「結束 Euphonia」都會結束整個程式、小說監控、快捷鍵與模型子程序。程式使用單一實例鎖，重複執行 `start.bat` 不會建立分身。

一般視窗及無邊框視窗適合這種桌面框選方式；獨佔全螢幕、受保護畫面及某些反作弊環境可能不允許截圖或快捷鍵。若遊戲使用系統管理員權限，請確認工具與遊戲的權限相符。程式不繞過遊戲的反作弊或畫面保護。

更新後請先關閉舊版程式，再重新執行 `start.bat`，執行中的舊版不會自動載入程式碼修改。
4. 截圖後會自動 OCR，再使用目前選取的角色生成語音並播放，不需要再按朗讀按鈕。若要先修改文字，取消勾選「截圖後自動辨識並播放」，修改後再按「生成並朗讀」。也可以直接貼上文字。
5. 生成後可停止播放、重播及匯出 WAV。

## 資料與模型

- 角色音檔及逐字稿：`data/voices/`；刪除原始匯入檔不影響角色。
- 生成語音：`data/outputs/`。每次生成均保留 WAV，可自行清理舊檔。
- 模型：優先使用 `data/model/` 的完整模型；否則首次載入從 Hugging Face 下載至預設快取。這是公開模型，通常不需要 token；程式不儲存 token。如需驗證，從執行環境提供 `HF_TOKEN`。
- 角色只需參考音檔與逐字稿，無需訓練；聲音特徵在本次執行期間快取，下次開啟會重新建立。

## 注意與限制

- OCR 結果可能有錯字、漏字或順序問題，複雜排版建議分區擷取；朗讀前可以編輯。
- 長文以最多 160 字分句生成，整段完成後播放，每次最多 5000 字。
- 停止生成會在目前句子或模型載入結束後生效；背景工作完成前暫時無法關閉視窗。
- 首次下載與模型載入可能需要數分鐘。使用 PyTorch SDPA，不需要在 Windows 編譯 FlashAttention。
- 語音相似度取決於音檔品質及逐字稿的一致性。

## 開發驗證

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\smoke.py
.venv\Scripts\python.exe scripts\smoke.py --tts
.venv\Scripts\python.exe scripts\test_background.py
```

可先執行 `.venv\Scripts\python.exe scripts\download_models.py all` 下載全部保留模型，或以 `qwen`、`f5`、`zipvoice` 只下載指定權重。加速版與 Streaming 共用 Qwen 權重。腳本支援斷點續傳並驗證 SHA-256。

`smoke.py` 使用自行繪製圖片測試 OCR 和介面；`--tts` 另外下載 Qwen 官方示範音檔，實際生成短句，不會建立使用者角色。

`test_background.py` 在 Windows 開啟另一個測試視窗，注入 F12 驗證原生鍵盤 hook、實際框選、背景工作銜接、靜音播放及原視窗焦點還原；OCR 與 TTS 使用替身，實際模型／OCR 另由 `test_sample.py` 驗證。測試結束會關閉測試視窗並釋放 hook，結果在 `data/background-test.json`。

官方模型與 API：https://github.com/QwenLM/Qwen3-TTS#voice-clone
