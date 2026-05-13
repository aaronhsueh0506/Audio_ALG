#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static void write_u16(FILE* f, uint16_t v) {
    uint8_t b[2] = {(uint8_t)(v & 0xFF), (uint8_t)(v >> 8)};
    fwrite(b, 1, 2, f);
}

// Distribute `total` among `bins` buckets, each in [minv, cap].
// Requires: bins * minv <= total <= bins * cap
static void distribute(int* out, int bins, int total, int minv, int cap) {
    for (int i = 0; i < bins; i++) out[i] = minv;
    int remaining = total - bins * minv;
    int room = cap - minv;
    for (int i = 0; i < bins; i++) {
        int left = bins - i - 1;
        int lo = remaining - left * room; if (lo < 0) lo = 0;
        int hi = remaining < room ? remaining : room;
        int extra = lo + rand() % (hi - lo + 1);
        out[i] += extra;
        remaining -= extra;
    }
}

// =====================================================
// gen() — canonical zero-RLE testcase generator
//
// 支援 4 種 case：
//   Case 1 (start=0, end=N): 0xZZNN N...  (一般 run)
//   Case 2 (start=N, end=N): 0x00NN N...  (z=0 first header)
//   Case 3 (start=0, end=0): ...0xZZ00    (trailing zeros，可含 split)
//   Case 4 (start=N, end=0): 0x00NN ... 0xZZ00
//
// 規則：
//   - 非 trailing 的 split (z=zmax, nz=0) 後面必須有 nz>0 的 final。
//   - 只有第一個 header 可以 z=0 (0x00NN)，不允許 0x0000。
//   - Trailing zeros: S_trail 個 split (z=zmax,nz=0) + 1 個 final (z=r_trail,nz=0)，
//     其中 r_trail in [1, zmax-1] (確保不是 zmax，避免混淆 split)。
//
// 回傳：0=成功，-1=參數不合法，-2=self-check 失敗
// =====================================================
int gen(int raw_size, int encode_size, int bit, FILE* raw_fp, FILE* enc_fp) {
    if (encode_size > 640) return -1;
    if (raw_size == 0 && encode_size == 0) return 0;

    int hw   = (bit == 16) ? 1 : 2;
    int zmax = (bit == 16) ? 255 : 65535;

    // ---- 決定 zero_start / zero_end ----
    // zero_start: 第一個 header z=0, nz=nz_first (raw 以 nonzero 開頭)
    // zero_end  : trailing S_trail splits + 1 final (nz=0, z=r_trail) (raw 以 zero 結尾)
    int nz_first = 0;
    int S_trail = 0, r_trail = 0;

    // zero_start: costs hw + nz_first enc words, nz_first raw elements
    // adj_raw=0, adj_enc=0 is valid (0 normal runs), so allow consuming all space
    if (rand() % 2 && raw_size >= 1 && encode_size >= hw + 1) {
        int max_nz = raw_size;
        int cap    = encode_size - hw;
        if (cap < max_nz) max_nz = cap;
        if (max_nz > zmax) max_nz = zmax;
        if (max_nz >= 1)
            nz_first = 1 + rand() % max_nz;
    }

    // zero_end: (S_trail+1)*hw enc words, (S_trail*zmax + r_trail) raw elements
    // r_trail in [1, zmax-1] so trailing final is never confused with split
    {
        int tmp_raw = raw_size - nz_first;
        int tmp_enc = encode_size - (nz_first > 0 ? hw + nz_first : 0);
        if (rand() % 2 && tmp_raw >= 1 && tmp_enc >= hw) {
            int max_S  = tmp_enc / hw - 1;       // (S_trail+1)*hw <= tmp_enc
            if (max_S < 0) max_S = 0;
            if (max_S > 5) max_S = 5;
            S_trail = rand() % (max_S + 1);
            int avail = tmp_raw - S_trail * zmax; // raw left for r_trail
            int max_r = (avail < zmax - 1) ? avail : zmax - 1;
            if (max_r < 1) {                      // S_trail too large, fall back to 0 splits
                S_trail = 0;
                avail   = tmp_raw;
                max_r   = (avail < zmax - 1) ? avail : zmax - 1;
            }
            if (max_r >= 1)
                r_trail = 1 + rand() % max_r;
        }
    }

    int z_trail = S_trail * zmax + r_trail;
    int adj_raw = raw_size  - nz_first - z_trail;
    int adj_enc = encode_size
                  - (nz_first > 0 ? hw + nz_first       : 0)
                  - (r_trail  > 0 ? (S_trail + 1) * hw  : 0);

    // ---- 可行性分析（先試 adj，若失敗 fallback 到原始）----
    //
    // R 個 normal runs，每個 run i 有 si 個 split header + 1 個 final header。
    //   H      = R + sum(si)          = (adj_enc - nz_total) / hw
    //   S      = sum(si) = H - R
    //   z_i    = si * zmax + ri,  ri in [1, zmax-1]  (保證 z_i >= 1)
    //   sum_ri = R*zmax - d,  d = H*zmax - total_zeros
    //
    // R 合法範圍：
    //   R >= ceil(d/zmax), >= 1, >= ceil(nz/zmax), >= ceil(d/(zmax-1))
    //   R <= min(d, nz_total, H)
    // ---- 可行性分析：5-pass + adj=0,0 sentinel ----
    //
    // n_cand == -1：sentinel，adj=0,0，0 個 normal run，直接合法
    // n_cand >  0 ：正常候選清單
    // n_cand == 0 ：真正不可行
    //
    // Pass 0: 隨機 z_start + z_trail（已算好 adj）
    // Pass 1: 放棄 z_start/z_trail，回到原始 (raw_size, encode_size)，純 normal runs
    // Pass 2: 強制最大 z_start，無 trail（覆蓋全 nonzero 的案例）
    // Pass 3: 強制 trail-only，無 z_start，無 normal（覆蓋全 zero with M = (S+1)*hw）
    // Pass 4: 強制 z_start + trail 組合，無 normal（覆蓋小 M 的 mixed 案例）
    int candidates[641], n_cand = 0;
    for (int pass = 0; pass < 5; pass++) {
        if (pass == 1) {
            nz_first = 0; S_trail = 0; r_trail = 0; z_trail = 0;
            adj_raw  = raw_size;
            adj_enc  = encode_size;
        } else if (pass == 2) {
            int nz_cap = encode_size - hw;
            if (nz_cap > raw_size) nz_cap = raw_size;
            if (nz_cap > zmax)     nz_cap = zmax;
            if (nz_cap < 1) continue;
            nz_first = nz_cap;
            S_trail = 0; r_trail = 0; z_trail = 0;
            adj_raw = raw_size  - nz_first;
            adj_enc = encode_size - hw - nz_first;
        } else if (pass == 3) {
            // Trail-only: M 必須等於 (S_trail+1)*hw, r_trail in [1, zmax-1]
            if (raw_size == 0) continue;
            if (encode_size < hw || encode_size % hw != 0) continue;
            int S_t = encode_size / hw - 1;
            if (S_t < 0) continue;
            int r_t = raw_size - S_t * zmax;
            if (r_t < 1 || r_t > zmax - 1) continue;
            nz_first = 0;
            S_trail = S_t; r_trail = r_t;
            z_trail = S_t * zmax + r_t;
            adj_raw = 0; adj_enc = 0;
        } else if (pass == 4) {
            // z_start + trail combo, no normal runs
            // M_z = hw + nz_first (z_start words), M_t = (S_trail+1)*hw (trail words)
            // M_z + M_t = M; nz_first + S_trail*zmax + r_trail = N
            if (raw_size == 0) continue;
            if (encode_size < 2*hw + 1) continue;
            int found = 0;
            for (int M_z = hw + 1; M_z <= encode_size - hw; M_z++) {
                int nz_f = M_z - hw;
                if (nz_f < 1 || nz_f > raw_size || nz_f > zmax) continue;
                int M_t = encode_size - M_z;
                if (M_t % hw != 0) continue;
                int S_t = M_t / hw - 1;
                if (S_t < 0) continue;
                int r_t = raw_size - nz_f - S_t * zmax;
                if (r_t < 1 || r_t > zmax - 1) continue;
                nz_first = nz_f; S_trail = S_t; r_trail = r_t;
                z_trail = S_t * zmax + r_t;
                adj_raw = 0; adj_enc = 0;
                found = 1; break;
            }
            if (!found) continue;
        }
        if (adj_raw == 0 && adj_enc == 0) { n_cand = -1; break; }  // 0 normal runs，合法
        if (adj_raw < 0  || adj_enc < 0)  continue;

        n_cand = 0;
        for (int nz = 1; nz <= adj_raw && nz <= adj_enc; nz++) {
            int hw_words = adj_enc - nz;
            if (hw_words % hw != 0) continue;
            int h = hw_words / hw;
            if (h < 1) continue;
            int tz = adj_raw - nz;
            long long d = (long long)h * zmax - tz;
            if (d < 0) continue;
            long long r_min = (d + zmax - 1) / zmax;
            if (r_min < 1) r_min = 1;
            long long r_min_nz = ((long long)nz + zmax - 1) / zmax;
            if (r_min < r_min_nz) r_min = r_min_nz;
            long long r_min_z1 = (d + (zmax - 2)) / (zmax - 1);
            if (r_min < r_min_z1) r_min = r_min_z1;
            long long r_max = (d < nz) ? d : (long long)nz;
            if (r_max > h) r_max = (long long)h;
            if (r_min > r_max) continue;
            candidates[n_cand++] = nz;
        }
        if (n_cand > 0) break;
    }
    if (n_cand == 0) return -1;

    // ---- 隨機選 nz_total，再選 R ----
    int nz_total = 0, H = 0, R = 0, S = 0, sum_ri = 0, total_zeros = 0;
    if (n_cand > 0) {
        nz_total    = candidates[rand() % n_cand];
        H           = (adj_enc - nz_total) / hw;
        total_zeros = adj_raw - nz_total;

        long long d     = (long long)H * zmax - total_zeros;
        long long r_min = (d + zmax - 1) / zmax; if (r_min < 1) r_min = 1;
        long long r_min_nz = ((long long)nz_total + zmax - 1) / zmax;
        if (r_min < r_min_nz) r_min = r_min_nz;
        long long r_min_z1 = (d + (zmax - 2)) / (zmax - 1);
        if (r_min < r_min_z1) r_min = r_min_z1;
        long long r_max = (d < nz_total) ? d : (long long)nz_total;
        if (r_max > H) r_max = H;
        R = (int)(r_min + rand() % (int)(r_max - r_min + 1));
        S = H - R;
        sum_ri = (int)((long long)R * zmax - d);
    }
    // n_cand == -1: R=0，跳過 run 生成迴圈

    // ---- 分配每個 run 的 (si, ri, nzh) ----
    int* nzh = (int*)calloc(R > 0 ? R : 1, sizeof(int));
    int* si  = (int*)calloc(R > 0 ? R : 1, sizeof(int));
    int* ri  = (int*)calloc(R > 0 ? R : 1, sizeof(int));

    if (R > 0) {
        distribute(nzh, R, nz_total, 1, zmax);
        if (S > 0) distribute(si, R, S, 0, S);
        distribute(ri, R, sum_ri, 1, zmax - 1);
    }

    // ---- 隨機打亂 run 順序 ----
    for (int i = R - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t;
        t = si[i];  si[i]  = si[j];  si[j]  = t;
        t = ri[i];  ri[i]  = ri[j];  ri[j]  = t;
        t = nzh[i]; nzh[i] = nzh[j]; nzh[j] = t;
    }

    // ---- 產生 raw 和 encoded buffer ----
    int raw_len = 0, enc_len = 0;
    uint16_t* raw_buf = (uint16_t*)malloc((raw_size  + 1) * sizeof(uint16_t));
    uint16_t* enc_buf = (uint16_t*)malloc((encode_size + 1) * sizeof(uint16_t));

    // Step 1: zero_start header (z=0, nz=nz_first) — 0x00NN
    if (nz_first > 0) {
        if (bit == 16) {
            enc_buf[enc_len++] = (uint16_t)(nz_first);   // hi=0x00, lo=NN
        } else {
            enc_buf[enc_len++] = (uint16_t)(nz_first);   // lo word: nz
            enc_buf[enc_len++] = 0;                       // hi word: z=0
        }
        for (int j = 0; j < nz_first; j++) {
            uint16_t v = (uint16_t)(rand() % 0xFFFE + 1);
            raw_buf[raw_len++] = v;
            enc_buf[enc_len++] = v;
        }
    }

    // Step 2: normal runs (each z>=1, nz>=1)
    for (int i = 0; i < R; i++) {
        // si[i] split headers: z=zmax, nz=0
        for (int k = 0; k < si[i]; k++) {
            for (int j = 0; j < zmax; j++)
                raw_buf[raw_len++] = 0;
            if (bit == 16) {
                enc_buf[enc_len++] = (uint16_t)(zmax << 8);
            } else {
                uint32_t w = (uint32_t)zmax << 16;
                enc_buf[enc_len++] = (uint16_t)(w & 0xFFFF);
                enc_buf[enc_len++] = (uint16_t)(w >> 16);
            }
        }
        // 1 final header: z=ri[i], nz=nzh[i]
        for (int j = 0; j < ri[i]; j++)
            raw_buf[raw_len++] = 0;
        if (bit == 16) {
            enc_buf[enc_len++] = (uint16_t)((ri[i] << 8) | nzh[i]);
        } else {
            uint32_t w = ((uint32_t)ri[i] << 16) | (uint32_t)nzh[i];
            enc_buf[enc_len++] = (uint16_t)(w & 0xFFFF);
            enc_buf[enc_len++] = (uint16_t)(w >> 16);
        }
        // nzh[i] nonzero payload
        for (int j = 0; j < nzh[i]; j++) {
            uint16_t v = (uint16_t)(rand() % 0xFFFE + 1);
            raw_buf[raw_len++] = v;
            enc_buf[enc_len++] = v;
        }
    }

    // Step 3: trailing zeros — S_trail splits (z=zmax,nz=0) + 1 final (z=r_trail,nz=0)
    if (r_trail > 0) {
        for (int k = 0; k < S_trail; k++) {
            for (int j = 0; j < zmax; j++) raw_buf[raw_len++] = 0;
            if (bit == 16) {
                enc_buf[enc_len++] = (uint16_t)(zmax << 8);
            } else {
                uint32_t w = (uint32_t)zmax << 16;
                enc_buf[enc_len++] = (uint16_t)(w & 0xFFFF);
                enc_buf[enc_len++] = (uint16_t)(w >> 16);
            }
        }
        for (int j = 0; j < r_trail; j++) raw_buf[raw_len++] = 0;
        if (bit == 16) {
            enc_buf[enc_len++] = (uint16_t)(r_trail << 8);  // nz=0
        } else {
            uint32_t w = (uint32_t)r_trail << 16;
            enc_buf[enc_len++] = (uint16_t)(w & 0xFFFF);
            enc_buf[enc_len++] = (uint16_t)(w >> 16);
        }
    }

    // ---- Self-check：尺寸與 decode 比對 ----
    // raw_len 必須等於 raw_size (N)，enc_len 必須等於 encode_size (M)，
    // decode(enc_buf) 必須得到完全相同的 raw_size 個 elements 並與 raw_buf bit-exact
    {
        if (raw_len != raw_size || enc_len != encode_size) {
            free(si); free(ri); free(nzh); free(raw_buf); free(enc_buf);
            return -2;
        }
        uint16_t* dec = (uint16_t*)malloc((raw_size + 1) * sizeof(uint16_t));
        int dec_len = 0, ei = 0;
        while (ei < enc_len) {
            int z, nz;
            if (bit == 16) {
                uint16_t w = enc_buf[ei++];
                z = (w >> 8) & 0xFF;
                nz = w & 0xFF;
            } else {
                uint32_t lo = enc_buf[ei++], hi = enc_buf[ei++];
                uint32_t w  = (hi << 16) | lo;
                z  = (int)((w >> 16) & 0xFFFF);
                nz = (int)(w & 0xFFFF);
            }
            for (int j = 0; j < z;  j++) dec[dec_len++] = 0;
            for (int j = 0; j < nz; j++) dec[dec_len++] = enc_buf[ei++];
        }
        int ok = (dec_len == raw_size);
        for (int i = 0; i < dec_len && ok; i++)
            ok = (dec[i] == raw_buf[i]);
        free(dec);
        if (!ok) {
            free(si); free(ri); free(nzh); free(raw_buf); free(enc_buf);
            return -2;
        }
    }

    // ---- 寫出檔案 ----
    for (int i = 0; i < raw_len; i++) write_u16(raw_fp, raw_buf[i]);
    for (int i = 0; i < enc_len; i++) write_u16(enc_fp, enc_buf[i]);

    free(si); free(ri); free(nzh); free(raw_buf); free(enc_buf);
    return 0;
}

int main(int argc, char* argv[]) {
    int N = -1, M = -1, bit = 16;
    unsigned seed = (unsigned)time(NULL);
    const char* raw_path = "raw.bin";
    const char* enc_path = "encoded.bin";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--n")    && i+1 < argc) N        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--m")    && i+1 < argc) M        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bit")  && i+1 < argc) bit      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed     = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--raw")  && i+1 < argc) raw_path = argv[++i];
        else if (!strcmp(argv[i], "--enc")  && i+1 < argc) enc_path = argv[++i];
        else {
            fprintf(stderr, "未知參數: %s\n", argv[i]);
            return 1;
        }
    }

    if (N < 0 || M < 0) {
        fprintf(stderr,
            "用法: %s --n N --m M [--bit 16|32] [--seed S] [--raw raw.bin] [--enc encoded.bin]\n",
            argv[0]);
        return 1;
    }
    if (bit != 16 && bit != 32) {
        fprintf(stderr, "錯誤: --bit 只能是 16 或 32\n");
        return 1;
    }

    srand(seed);

    FILE* r = fopen(raw_path, "wb");
    FILE* e = fopen(enc_path, "wb");
    if (!r || !e) {
        fprintf(stderr, "錯誤: 無法開啟輸出檔案\n");
        if (r) fclose(r);
        if (e) fclose(e);
        return 1;
    }

    int ret = gen(N, M, bit, r, e);
    fclose(r);
    fclose(e);

    if (ret == 0) {
        printf("gen(%d, %d, %d) 成功 -> %s (%d bytes), %s (%d bytes)\n",
               N, M, bit, raw_path, N*2, enc_path, M*2);
        return 0;
    } else {
        fprintf(stderr, "gen(%d, %d, %d) 失敗: %d\n", N, M, bit, ret);
        return (ret == -1) ? 2 : 3;
    }
}
