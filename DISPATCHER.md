# FlashAttention FP8 forward — диспетчер ядер для sm_120a

Готовые pre-built бинари для **RTX PRO 6000 Blackwell Workstation Edition** (sm_120a, 188 SMs).
Для других карт пересоберите через `runs/_build_*.sh`.

## Как запустить

```bash
# Прямой бенч любого ядра — печатает correctness 8/8 + 23 config бенч:
runs/fa_v121_addrhoist          # peak champion hd=128
runs/fa_v118_localfix           # mid-grid champion hd=128
runs/fa_v122_br64_mt1           # wave-tail bh=4 hd=128
runs/fa_v96b_localfix           # universal baseline hd=128
runs/fa_v89_pinregs_fp8         # peak champion hd=64
runs/fa_v117_partial_top        # sliding-window niche hd=128

# Полный A/B всех топ-ядер ×3:
runs/run_dispatcher_audit.sh

# NCu профиль (нужна privileged GPU access):
runs/run_ncu_v122.sh
```

## Диспетчер по (hd, bh, sl, wnd)

### hd=64 стек

| Условие | Ядро | Peak |
|---|---|---|
| grid ≤ 128 (wave-tail) | `fa_v80b` * | до +40% на bh=4 |
| grid > 128 (peak) | `fa_v89_pinregs_fp8` | **466T best / 413T mean** |

\* v80b бинарь в этом репо отсутствует — пересобрать из старого коммита `fbdb868`.

### hd=128 стек — полная карта победителей

| bh | sl | wnd | Ядро | TFLOPS | Δ vs runner-up |
|---|---|---|---|---|---|
| 4 | 1024 | 0 | **fa_v122_br64_mt1** | 74.9 | +21% v118 |
| 4 | 2048 | 0 | **fa_v122_br64_mt1** | 167.6 | +20% v118 |
| 4 | 4096 | 0 | **fa_v118_localfix** | 284.4 | +9% v117 |
| 8 | 2048 | 0 | **fa_v118_localfix** | 262.2 | +7% v117 |
| 8 | 4096 | 0 | **fa_v118_localfix** | 426.4 | +1.5% v117 |
| 16 | 2048 | 0 | **fa_v118_localfix** | 385.1 | +1.1% v121 |
| 16 | 4096 | 0 | **fa_v96b_localfix** | 454.8 | +0.7% v121 |
| 32 | 2048 | 0 | **fa_v96b_localfix** | 414.7 | +1.0% v121 |
| 32 | 4096 | 0 | **fa_v121_addrhoist** | 549.4 | +5.1% v96b |
| 32 | 8192 | 0 | **fa_v121_addrhoist** | 587.3 | +0.7% v96b |
| 64 | 4096 | 0 | **fa_v96b_localfix** | 552.5 | +1.3% v121 |
| 64 | 8192 | 0 | **fa_v121_addrhoist** | **605.8** | +3.0% v96b |
| 128 | 2048 | 0 | **fa_v96b_localfix** | 508.1 | +1.3% v121 |
| 128 | 4096 | 0 | **fa_v121_addrhoist** | 581.5 | +3.2% v96b |
| 128 | 8192 | 0 | **fa_v121_addrhoist** | 604.7 | +2.4% v96b |
| 256 | 2048 | 0 | **fa_v121_addrhoist** | 536.1 | +2.6% v96b |
| 256 | 4096 | 0 | **fa_v121_addrhoist** | 581.8 | +2.6% v96b |
| 4 | 4096 | 1024 | **fa_v118_localfix** | 199.6 | +5% v122 |
| 4 | 8192 | 1024 | **fa_v117_partial_top** | 288.9 | +1.6% v111 |
| 8 | 8192 | 1024 | **fa_v118_localfix** | 297.3 | +0.7% v121 |
| 16 | 8192 | 1024 | **fa_v121_addrhoist** | 380.8 | +0.7% v96b |
| 32 | 8192 | 1024 | **fa_v121_addrhoist** | 400.4 | +0.2% v96b |
| 64 | 8192 | 1024 | **fa_v121_addrhoist** | 424.5 | +0.1% v96b |

### Сводка покрытия

| Ядро | Doля configs | Тип |
|---|---|---|
| `fa_v121_addrhoist` | 10 (43%) | peak champion (bh≥32 ∨ sl≥4096) |
| `fa_v118_localfix` | 8 (35%) | mid-grid + sliding-window niches |
| `fa_v96b_localfix` | 5 (22%) | узкие peak (bh=16-32×sl≤4096) |
| `fa_v122_br64_mt1` | 2 (9%) | bh=4 sl≤2048 wave-tail |
| `fa_v117_partial_top` | 1 (4%) | bh=4 sl=8192 wnd=1024 |

## Псевдокод диспетчера

```python
def dispatch_fa_fp8_forward(bh, sl, hd, wnd=0):
    if hd == 64:
        grid = bh * ((sl + 127) // 128)
        return "fa_v89_pinregs_fp8" if grid > 128 else "fa_v80b"

    if hd != 128:
        raise ValueError("Supported: hd ∈ {64, 128}")

    # hd=128
    if bh == 4 and sl <= 2048:
        return "fa_v122_br64_mt1"
    if bh == 4 and sl == 8192 and wnd == 1024:
        return "fa_v117_partial_top"
    if bh <= 8 and sl <= 4096:
        return "fa_v118_localfix"
    if bh == 4 and wnd > 0:
        return "fa_v118_localfix"
    if bh == 8 and wnd == 1024:
        return "fa_v118_localfix"
    if bh in (16, 32) and sl <= 4096:
        return "fa_v96b_localfix"
    if bh == 64 and sl == 4096:
        return "fa_v96b_localfix"
    if bh == 128 and sl == 2048:
        return "fa_v96b_localfix"
    return "fa_v121_addrhoist"
```

## Сборка вручную

```bash
# v121 (peak champion):
runs/_build_v121.sh

# v122 (wave-tail):
runs/_build_v122.sh

# v118 (mid-grid):
runs/run_fa_v118_localfix.sh

# v117 (sliding-window niche):
# (нет отдельного _build скрипта — собирается вручную:)
/usr/local/cuda-13.1/bin/nvcc -O3 \
    -gencode arch=compute_120a,code=sm_120a -std=c++17 -Xptxas=-v -lineinfo \
    libs/flash_attention_v117_partial_top_sync_hd128_fp8_forward.cu \
    -o runs/fa_v117_partial_top -lcudart
```

## Файлы

- `libs/flash_attention_v*.cu` — исходники всех ядер
- `runs/fa_v*` — pre-built бинари (sm_120a)
- `libs/*.so` — sm_120a CUDA shared libs (для Go integration)
- `runs/ncu_*.csv` — NCu профили (stall/util cfg=9)
- `runs/_build_*.sh` — сборочные скрипты
- `runs/run_*.sh` — A/B бенч скрипты
- `runs/mbar_repro_v3*` — диагностические репродьюсеры M2-M8 mbarrier hang
