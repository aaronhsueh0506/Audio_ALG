# RNNoise-ERB TODO

- [ ] **gen_dataset 雙聲道單檔**: 評估把 `noisy/000000.wav` + `clean/000000.wav` 合併成單一 2-channel WAV.
  - 好處: 檔案數減半、對齊自動保證、I/O 一次 open
  - 壞處: debug 時要 split channel 才能單獨聽 clean；給其他工具用稍不直觀
  - 結論: 先用分開兩檔跑訓練，覺得不順手再改
