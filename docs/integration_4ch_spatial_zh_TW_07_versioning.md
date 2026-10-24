> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_06_checklist.md) <!-- split-nav -->
## 8. 版本與相容性

### 8.1 `AudioPipeline4ChMemReq`

固定 32 bytes、每個欄位固定 byte offset（由 header 內的 `_Static_assert` 釘死），
全部是定寬整數。形狀與核心層 `FourAecNrResMemReq`
相同，但**版本號各自獨立**；本次 checkout，`layout_version=21`（`AUDIO_PIPELINE_4CH_LAYOUT_VERSION`）。

| Offset | 欄位 | 型別 | 目前值 |
|---:|---|---|---|
| 0 | `descriptor_version` | `uint32_t` | `1` |
| 4 | `layout_version` | `uint32_t` | `21` |
| 8 | `backend_id` | `uint32_t` | `2` = NE10（直接沿用核心層的值；目前唯一的 FFT 後端；永遠不會是 0） |
| 12 | `build_flags_hash` | `uint32_t` | FNV-1a-32，**已把核心層的 `build_flags_hash` 摺進去** |
| 16 | `alignment` | `uint32_t` | `16` |
| 20 | `reserved` | `uint32_t` | `0`（必須為 0，`init_ex()` 會驗證） |
| 24 | `bytes` | `uint64_t` | pool 總需求（含 core + SRP + BF + 本層 scratch；不含 RES／NR／合成等後級 stage） |

`build_flags_hash` 摺入核心層 hash 的意義：**核心層佈局一改版，
你所有已保存的複合 descriptor 就一起失效**，不會出現「用舊核心佈局算出來的 pool
剛好還塞得下」這種安靜的錯誤。

### 8.2 `init_ex()` 的 8 項檢查

`expected == NULL` 時完全等同 `audio_pipeline_4ch_init()`。
`expected != NULL` 時，下列 8 項全部成立才會開始 carve；任一不符即回 `NULL`
（**不輸出任何訊息**）：

1. `descriptor_version` 相符
2. `layout_version` 相符
3. `backend_id` 相符
4. `build_flags_hash` 相符
5. `alignment` 相符
6. `reserved == 0`
7. `expected->bytes >= 目前 build 的需求`
8. `bytes`（實際交進來的 pool 大小）`>= 目前 build 的需求`

### 8.3 序列化

可以逐 byte 拷貝到檔案 / flash / 訊息 buffer 再讀回來，**但僅限相同 endianness**。
不提供 byte-swap 輔助，跨 endianness 交換不在支援範圍。

### 8.4 相容性規則（給整合者的四條）

1. **descriptor 永遠現查現用**。不要存進 NVRAM、不要跨 firmware rebuild 沿用。
2. **`backend_id` 不是 2 的 descriptor 一律不相容**，即使 `bytes` 相同。
3. **升級 library 之後一定要重跑一次 `get_mem_requirements()`**，並確認 pool 預算仍足夠。
4. **升級之後也要重新確認 [§4.6](integration_4ch_spatial_zh_TW_03_config.md#46-你設定的值不一定是實際生效的值) 的三個生效值**：改 grid、改 backend、
   或核心層改版都可能讓生效值變動，而設定值看起來完全沒變。
> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_06_checklist.md) <!-- split-nav -->
