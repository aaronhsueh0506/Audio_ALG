# Mono AEC、NR 與 RES

應用程式每 hop 接收一路 mic 與一路 far。先取 AEC 的 error spectrum 與 RES gain，將 RES／NR gain 逐 bin 合成並套用一次，最後做一次 ISTFT／OLA。支援 8 kHz 的 256/128、16 kHz 的 256/128 或 512/256、48 kHz 的 1024/512（FFT／hop）。

- [最短 C 接法、公開介面及參數](integration.md)
- [初始化、reset 與錯誤恢復](lifecycle.md)
- [WAV CLI、可編譯的 C wrapper 與 buffer 生命週期](cli-and-example.md)
- [頻域資料流與 gain 合成](frequency-domain.md)
- [六條 pipeline 的程式入口](../../../pipelines/README.md)
