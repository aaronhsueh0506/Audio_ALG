> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_06_checklist.md) <!-- split-nav -->
## 8. 版本與相容性

只有 mcab 有 descriptor。helper（`mcab_sample_beam_*`）是範例碼，pool 以 `mcab_sample_beam_get_mem_size()` 查大小，沒有 descriptor、沒有 layout version；
lane 與 estimator 各由 libaec 自己的 `*_get_mem_size()` 查大小。

### 8.1 `McabMemReq`

固定 32 bytes、每個欄位固定 byte offset（由 header 內的 `_Static_assert` 釘死），
全部是定寬整數。本次 checkout，`layout_version=25`（`MCAB_LAYOUT_VERSION`；4ch flow 只剩一個 descriptor，見 `mcab.h`），
它只描述 **mcab 自己的 pool**（control block、mic lane、aligned far、`enable_post` 時的 fuse 緩衝、MATCHED／FIXED 的對齊環）：
lane、estimator、SRP、BF 都不在其中。`MCAB_DESCRIPTOR_VERSION` 為 1。

| Offset | 欄位 | 型別 | 目前值 |
|---:|---|---|---|
| 0 | `descriptor_version` | `uint32_t` | `1` |
| 4 | `layout_version` | `uint32_t` | `25` |
| 8 | `backend_id` | `uint32_t` | `2` = NE10（`MCAB_BACKEND_NE10`；目前唯一的 FFT 後端；永遠不會是 0 或 1） |
| 12 | `build_flags_hash` | `uint32_t` | FNV-1a-32，只涵蓋 carve 結構（`mcab.c` 的 `mcab_build_flags_hash()` 的 carve token），不含可調數值 |
| 16 | `alignment` | `uint32_t` | `16` |
| 20 | `reserved` | `uint32_t` | `0`（必須為 0，`mcab_init_ex()` 會驗證） |
| 24 | `bytes` | `uint64_t` | mcab pool 總需求（不含 lane、estimator、helper，也不含 RES／NR／合成等後級 stage） |

被取代的舊 layout（21～24 屬於前身的兩層；17 與 20 是更早的 unit layout）永遠不會再使用。`tests/test_mcab.c` 的 `run_static_parity` 對 17～24 逐一檢查：
帶這些版號的 descriptor 一律被拒絕（17～23 另外把快取的 bytes 加大 4096，24 帶目前的 bytes，版號是唯一差異），被拒時 pool 一個 byte 都不會被寫。

### 8.2 `mcab_init_ex()` 的 8 項檢查

`expected == NULL` 時完全等同 `mcab_init()`。
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

這 8 項之外，`mcab_init_ex()` 還會驗證注入的 lane 與 estimator（[§5.7](integration_4ch_spatial_zh_TW_04_errors.md#57-mcab_init_ex-驗證什麼不能驗證什麼)）。

### 8.3 序列化

可以逐 byte 拷貝到檔案 / flash / 訊息 buffer 再讀回來，**但僅限相同 endianness**。
不提供 byte-swap 輔助，跨 endianness 交換不在支援範圍。

### 8.4 相容性規則（給整合者的四條）

1. **descriptor 永遠現查現用**。不要存進 NVRAM、不要跨 firmware rebuild 沿用，也不要自己合成一個。
2. **`backend_id` 不是 2 的 descriptor 一律不相容**，即使 `bytes` 相同。
3. **升級 library 之後一定要重跑一次 `mcab_get_mem_requirements()`**，並確認 pool 預算仍足夠；lane、estimator 與 helper 的大小也要各自重查。
4. **升級之後也要重新確認 [§4.6](integration_4ch_spatial_zh_TW_03_config.md#46-你設定的值不一定是實際生效的值) 的三個生效值**：改 grid、改 backend、
   或 mcab 改版都可能讓生效值變動，而設定值看起來完全沒變。
> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_06_checklist.md) <!-- split-nav -->
