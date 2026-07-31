# GTCRN 論文分析與 DeepFilterNet 比較

> **論文**: GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources (ICASSP 2024)
>
> **作者**: Xiaobin Rong et al. (Nanjing University, Horizon Robotics)

---

## 1. 核心數據一覽

| 指標 | GTCRN | DeepFilterNet | RNNoise |
|------|-------|---------------|---------|
| **參數量** | 23.7K (48.2K 含 ERB) | 1.80M | 0.06M |
| **MACs/s** | 39.6M (33.0M 優化後) | 350M | 40M |
| **PESQ** (VCTK) | **2.87** | 2.81 | 2.29 |
| **SISNR** (VCTK) | **18.83** | 16.63 | - |
| **STOI** (VCTK) | 0.940 | **0.942** | - |

**結論**: GTCRN 用 **1/36 的參數** 和 **1/9 的計算量** 達到了比 DeepFilterNet 更好的 PESQ 和 SISNR。

---

## 2. GTCRN 架構總覽

```
Input Spectrum (B, F, T, 2)
       │
       ▼
┌─────────────────┐
│   Band Merging  │  257 → 129 bins (ERB scale)
└────────┬────────┘
         ▼
┌─────────────────┐
│      SFE        │  Subband Feature Extraction
└────────┬────────┘
         ▼
┌─────────────────────────────────────────┐
│              Encoder                     │
│  Conv → Conv → GT-Conv → GT-Conv → GT-Conv │
└────────┬────────────────────────────────┘
         ▼
┌─────────────────┐
│   G-DPRNN x2    │  Grouped Dual-Path RNN
└────────┬────────┘
         ▼
┌─────────────────────────────────────────┐
│              Decoder                     │
│  GT-Conv → GT-Conv → GT-Conv → DeConv → DeConv │
└────────┬────────────────────────────────┘
         ▼
┌─────────────────┐
│  Band Splitting │  129 → 257 bins
└────────┬────────┘
         ▼
┌─────────────────┐
│  Complex Mask   │  CRM 應用
└────────┬────────┘
         ▼
Output Spectrum (B, F, T, 2)
```

---

## 3. 關鍵技術詳解

### 3.1 ERB Band Merging/Splitting

**目的**: 減少頻率維度的冗餘

**原理**:
- 人耳對高頻的解析度較低（符合 ERB scale）
- 語音諧波主要出現在低頻，高頻資訊冗餘度高

**實作**:
```python
# 低頻 65 bins (0-2kHz): 保持原解析度
# 高頻 192 bins (2-8kHz): 合併成 64 ERB bands
# 總計: 65 + 64 = 129 bins

x_low = x[..., :65]                    # 保留
x_high = self.erb_fc(x[..., 65:])      # 線性映射 192→64
output = torch.cat([x_low, x_high], dim=-1)
```

**效果**: 頻率維度從 257 減少到 129，後續所有計算量約減少 **50%**

---

### 3.2 GT-Conv Block (Grouped Temporal Convolution)

**設計來源**: ShuffleNetV2

**架構**:
```
Input (B, C, T, F)
       │
       ├──────────────────┐
       │                  │
       ▼                  │
  Channel Split           │
  (C → C/2, C/2)          │
       │                  │
       ▼                  │
┌─────────────┐           │
│    SFE      │           │
├─────────────┤           │
│  P-Conv2D   │  1×1 conv │
│  BN + PReLU │           │
├─────────────┤           │
│ DD-Conv2D   │  Dilated  │
│  BN + PReLU │  Depthwise│
├─────────────┤           │
│  P-Conv2D   │  1×1 conv │
│     BN      │           │
├─────────────┤           │
│    TRA      │  Attention│
└──────┬──────┘           │
       │                  │
       ▼                  ▼
     Concat ◄────────────┘
       │
       ▼
  Channel Shuffle
       │
       ▼
Output (B, C, T, F)
```

**關鍵設計**:
1. **Channel Split**: 只處理一半 channel，另一半 bypass → 計算減半
2. **Depthwise Separable Conv**: P-Conv + D-Conv 取代標準 Conv
3. **Dilated Conv**: dilation = [1, 2, 5] 擴大感受野不增加參數
4. **Channel Shuffle**: 確保兩個分支的資訊交換

---

### 3.3 SFE - Subband Feature Extraction

**目的**: 讓 1×1 convolution 也能捕捉頻率間的關係

**原理**:
```
原本: 每個頻率 bin 獨立處理
SFE 後: 每個位置包含相鄰 k 個 bins 的資訊
```

**實作**:
```python
class SFE(nn.Module):
    def __init__(self, kernel_size=3):
        self.unfold = nn.Unfold(kernel_size=(1, 3))

    def forward(self, x):
        # x: (B, C, T, F)
        # unfold 在頻率維度展開
        xs = self.unfold(x)
        # reshape: (B, C, T, F) → (B, C*3, T, F)
        return xs.reshape(B, C*3, T, F)
```

**效果**:
- 將頻率維度的 subband 關係整合到 channel 維度
- 後續的 pointwise conv 可以利用相鄰頻帶資訊
- **幾乎零計算成本** (只是 tensor reshape)

---

### 3.4 TRA - Temporal Recurrent Attention

**目的**: 對時間特徵進行動態重新校準

**與一般 Time Attention 的差異**:
| 方法 | 建模方式 | 特點 |
|------|----------|------|
| TA (Time Attention) | FC layers | 無法建模時間序列依賴 |
| **TRA** | GRU + FC | 能建模時間序列的動態變化 |

**架構**:
```
Input V: (B, C, T, F)
         │
         ▼
┌─────────────────────┐
│  Square + Avg Pool  │  在頻率維度平均
│  Z(c,t) = mean(V²)  │
└────────┬────────────┘
         │  (B, C, T)
         ▼
┌─────────────────────┐
│       GRU           │  channels: C → 2C
└────────┬────────────┘
         ▼
┌─────────────────────┐
│       FC            │  channels: 2C → C
└────────┬────────────┘
         ▼
┌─────────────────────┐
│     Sigmoid         │  生成 attention mask
└────────┬────────────┘
         │  A: (B, C, T, 1)
         ▼
    Output = V ⊗ A     (element-wise multiply)
```

**實作**:
```python
class TRA(nn.Module):
    def __init__(self, channels):
        self.att_gru = nn.GRU(channels, channels*2, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, T, F)
        zt = torch.mean(x.pow(2), dim=-1)  # (B, C, T) 時間能量
        at = self.att_gru(zt.transpose(1,2))[0]
        at = self.att_fc(at).transpose(1,2)
        at = self.att_act(at)
        At = at[..., None]  # (B, C, T, 1)
        return x * At
```

---

### 3.5 G-DPRNN (Grouped Dual-Path RNN)

**設計來源**: DPRNN + Grouped RNN

**Dual-Path 概念**:
```
Intra-frame RNN: 建模單一幀內的頻譜模式 (頻率方向)
Inter-frame RNN: 建模時間依賴 (時間方向)
```

**Grouped RNN 策略**:
```python
# 標準 GRU
gru = nn.GRU(input_size=16, hidden_size=16)
# 參數量 ≈ 3 × 16 × 16 × 2 = 1536

# Grouped GRU (groups=2)
gru1 = nn.GRU(input_size=8, hidden_size=8)
gru2 = nn.GRU(input_size=8, hidden_size=8)
# 參數量 ≈ 3 × 8 × 8 × 2 × 2 = 768 (減少 50%)
```

**因果性保證**:
- Intra-frame: Bidirectional GRU (可以看完整頻譜)
- Inter-frame: **Unidirectional** GRU (只看過去，保證因果)

---

### 3.6 Loss Function

**多域聯合損失**:
```python
L = α·L_SISNR + (1-β)·L_mag + β·(L_real + L_imag)
# α = 0.01, β = 0.3
```

| 損失項 | 域 | 公式 |
|--------|-----|------|
| L_SISNR | 波形域 | -log₁₀(‖sₜ‖²/‖s̃-sₜ‖²) |
| L_mag | 頻譜域 | MSE(\|S̃\|^0.3, \|S\|^0.3) |
| L_real | 頻譜域 | MSE(S̃ᵣ/\|S̃\|^0.7, Sᵣ/\|S\|^0.7) |
| L_imag | 頻譜域 | MSE(S̃ᵢ/\|S̃\|^0.7, Sᵢ/\|S\|^0.7) |

**特點**: 使用 compressed magnitude (0.3 次方) 來平衡大小值的貢獻

---

## 4. GTCRN vs DeepFilterNet 詳細比較

### 4.1 設計哲學差異

| 面向 | GTCRN | DeepFilterNet |
|------|-------|---------------|
| **核心目標** | 極致輕量化 | 全頻帶高品質 |
| **增強方法** | Complex Ratio Mask | Deep Filtering + Mask 混合 |
| **頻率處理** | ERB 合併壓縮 | ERB + 細粒度處理 |
| **週期成分** | 無特殊處理 | Deep filtering 增強 |
| **目標平台** | Edge device (耳機、助聽器) | 有更多資源的裝置 |

### 4.2 架構差異

**DeepFilterNet 架構**:
```
┌─────────────────────────────────────┐
│         Spectral Envelope           │
│    (Mask-based, ERB bands)          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       Periodic Components           │
│    (Deep Filtering, fine-grained)   │
│    學習 FIR filter coefficients      │
└──────────────┬──────────────────────┘
               │
               ▼
           Enhanced
```

**GTCRN 架構**:
```
┌─────────────────────────────────────┐
│      Complex Ratio Mask             │
│   (直接預測實部虛部 mask)            │
└──────────────┬──────────────────────┘
               │
               ▼
           Enhanced
```

### 4.3 Deep Filtering vs Complex Masking

**Deep Filtering (DeepFilterNet)**:
```python
# 對每個頻帶學習一組 FIR filter coefficients
# y[n] = Σ h[k] · x[n-k]
# 可以精細處理語音的週期性成分 (pitch harmonics)
```
- 優點: 對週期成分增強效果好
- 缺點: 計算量較大

**Complex Ratio Mask (GTCRN)**:
```python
# 直接預測複數 mask
s_real = spec_real * mask_real - spec_imag * mask_imag
s_imag = spec_imag * mask_real + spec_real * mask_imag
```
- 優點: 計算簡單
- 缺點: 對週期成分沒有特殊處理

### 4.4 計算量分析

| 組件 | GTCRN | DeepFilterNet |
|------|-------|---------------|
| 輸入處理 | ERB merge (簡單) | ERB + 多分支 |
| 主幹網路 | Grouped Conv/RNN | UNet-like DNN |
| 輸出處理 | CRM (乘法) | Deep filtering (卷積) |
| **總計** | **33-40 MMACs/s** | **350 MMACs/s** |

---

## 5. Ablation Study 結果

| SFE | TA | TRA | Params | MACs | SISNR | PESQ | STOI |
|:---:|:--:|:---:|:------:|:----:|:-----:|:----:|:----:|
| ✗ | ✗ | ✗ | 13.35K | 33.91M | 9.87 | 1.87 | 0.834 |
| ✗ | ✓ | ✗ | 14.84K | 34.00M | 10.00 | 1.89 | 0.838 |
| ✗ | ✗ | ✓ | 21.65K | 34.47M | 10.25 | 1.91 | 0.840 |
| ✓ | ✗ | ✗ | 15.37K | 39.07M | 10.10 | 1.90 | 0.838 |
| ✓ | ✓ | ✗ | 16.86K | 39.16M | 10.29 | 1.92 | 0.841 |
| ✓ | ✗ | ✓ | 23.67K | 39.63M | **10.39** | **1.94** | **0.844** |

**結論**:
- TRA 比 TA 效果更好 (+0.25 SISNR)，幾乎不增加計算量
- SFE + TRA 組合達到最佳效果

---

## 6. 實作細節

### 6.1 STFT 參數
```python
window_length = 32 ms (512 samples @ 16kHz)
hop_length = 16 ms (256 samples)
fft_size = 512
window = sqrt(hann)  # 平方根漢寧窗
```

### 6.2 網路配置
```python
# ERB
erb_subband_1 = 65   # 低頻保留 bins
erb_subband_2 = 64   # 高頻壓縮後 bins

# Encoder/Decoder
channels = 16
kernel_size = (1, 5) for Conv, (3, 3) for GT-Conv
dilations = [1, 2, 5]  # 時間方向

# G-DPRNN
hidden_size = 16
groups = 2
```

### 6.3 訓練配置
```python
optimizer = Adam(lr=0.001)
lr_scheduler = ReduceLROnPlateau(patience=5, factor=0.5)
batch_size = 4 (VCTK) / 16 (DNS3)
chunk_length = 8 seconds
```

---

## 7. 優缺點總結

### GTCRN 優點
1. **極致輕量**: 23.7K 參數，適合 edge device
2. **低延遲**: 因果設計，RTF 0.07
3. **高效能**: 在相似計算量下大幅超越 RNNoise
4. **設計巧妙**: 多種策略組合達到最佳效率

### GTCRN 缺點
1. **去混響能力有限**: 純 masking 難以處理長時間依賴
2. **無週期成分增強**: 不如 DeepFilterNet 對語音諧波的處理
3. **頻率解析度降低**: ERB 合併犧牲了部分高頻細節

### 適用場景
| 場景 | 推薦模型 |
|------|----------|
| 耳機、助聽器、IoT | **GTCRN** |
| 手機、平板 | GTCRN 或 DeepFilterNet2 |
| PC、伺服器 | DeepFilterNet3 或更大模型 |

---

## 8. 參考資料

- [GTCRN Paper (IEEE Xplore)](https://ieeexplore.ieee.org/document/10448310/)
- [GTCRN GitHub](https://github.com/Xiaobin-Rong/gtcrn)
- [DeepFilterNet Paper](https://arxiv.org/abs/2110.05588)
- [DeepFilterNet GitHub](https://github.com/Rikorose/DeepFilterNet)
- [ShuffleNetV2 Paper](https://arxiv.org/abs/1807.11164)
- [DPRNN Paper](https://arxiv.org/abs/1910.06379)
