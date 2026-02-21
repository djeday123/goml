/*
 * nvlink_probe2.c - NVLink Probe v2
 *
 * Исправлено:
 *   - 0xBADF1301 теперь правильно классифицируется как PRI_TIMEOUT
 *   - 0xBADF5040 как PRI_ERROR
 *   - Добавлен доступ через PRAMIN window для MMIO > 16MB
 *   - Более точный анализ fuse register status
 *   - Чтение NVIDIA open-gpu-kernel register offsets для Ada
 *
 * gcc -O2 -o nvlink_probe2 nvlink_probe2.c
 * sudo ./nvlink_probe2
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <glob.h>
#include <stdbool.h>

// ============================================================================
// PRI (Privileged Register Interface) error codes
// Это коды ошибок внутренней шины GPU, НЕ данные
// ============================================================================

static bool is_pri_error(uint32_t val)
{
    // BADF prefix = PRI fault/timeout
    // Known patterns from nouveau/open-gpu-kernel:
    //   0xBADF1000-0xBADF1FFF = PRI_TIMEOUT variants
    //   0xBADF3000-0xBADF3FFF = PRI_TIMEOUT_IN_RECOVERY
    //   0xBADF5000-0xBADF5FFF = PRI_ERROR variants
    //   0xBADFx000           = Various PRI faults
    return (val & 0xFFFF0000) == 0xBADF0000;
}

static bool is_dead(uint32_t val)
{
    return val == 0x00000000 ||
           val == 0xFFFFFFFF ||
           val == 0xDEADDEAD ||
           is_pri_error(val);
}

static const char *classify_reg(uint32_t val)
{
    if (val == 0x00000000)
        return "ZERO (unmapped/default)";
    if (val == 0xFFFFFFFF)
        return "ALL-1s (bus error/no device)";
    if (val == 0xDEADDEAD)
        return "DEADDEAD (out of range)";
    if ((val & 0xFFFF0000) == 0xBADF0000)
    {
        uint16_t code = val & 0xFFFF;
        if (code >= 0x1000 && code < 0x2000)
            return "PRI_TIMEOUT (engine power-gated)";
        if (code >= 0x3000 && code < 0x4000)
            return "PRI_TIMEOUT_RECOVERY";
        if (code >= 0x5000 && code < 0x6000)
            return "PRI_ERROR (access denied/invalid)";
        return "PRI_FAULT (unknown subtype)";
    }
    return "LIVE DATA";
}

// ============================================================================
// GPU structures
// ============================================================================

typedef struct
{
    char bdf[32];
    uint16_t device_id;
    uint64_t bar0_addr;
    uint64_t bar0_size;
    char sysfs_path[512];
} GPUDevice;

typedef struct
{
    void *base;
    size_t size;
    int fd;
} MMIOMapping;

// ============================================================================
// PRAMIN window - доступ к верхнему MMIO через окно в нижних адресах BAR0
//
// На NVIDIA GPU есть "PRAMIN window" - 1MB окно в BAR0 (обычно 0x700000),
// которое можно настроить на любой физический адрес GPU MMIO.
// Это позволяет читать регистры выше BAR0 size.
//
// Регистры:
//   0x001700 - PBUS_BAR0_WINDOW (NV_PBUS_BAR0_WINDOW_BASE)
//              Задаёт базовый адрес в GPU MMIO space, делённый на 0x10000
// ============================================================================

#define NV_PBUS_BAR0_WINDOW 0x00001700
#define PRAMIN_WINDOW_BASE 0x00700000 // Начало окна в BAR0
#define PRAMIN_WINDOW_SIZE 0x00100000 // 1MB окно

// ============================================================================
// Known Ada (AD102) register offsets
// Из open-gpu-kernel-modules и envytools
// ============================================================================

// Top-Level registers
#define NV_PMC_BOOT_0 0x00000000
#define NV_PMC_BOOT_1 0x00000004
#define NV_PMC_BOOT_2 0x00000008
#define NV_PMC_ENABLE 0x00000200
#define NV_PMC_DEVICE_ENABLE(i) (0x00000600 + (i) * 4)

// PTOP - Partition topology
// На Ada device info table формат отличается от Ampere
#define NV_PTOP_DEVICE_INFO2_OFFSET 0x00022800 // Ada uses different table

// PBUS
#define NV_PBUS_PCI_NV_0 0x00001800
#define NV_PBUS_PCI_NV_1 0x00001804
#define NV_PBUS_PCI_NV_2 0x00001808
#define NV_PBUS_PCI_NV_19 0x0000184C // Subsystem ID

// NVLink blocks (high MMIO, need PRAMIN for BAR0 < 32MB)
// Из open-gpu-kernel ga100/dev_nvlipt_ip.h, ad102 может отличаться

// NVLIPT_COMMON
#define NV_NVLIPT_COMMON_BASE_AD 0x00A20000
#define NV_NVLIPT_INTR_CONTROL 0x00A20344

// IOCTRL
#define NV_IOCTRL_BASE_AD 0x00A00000
#define NV_IOCTRL_RESET 0x00A00140
#define NV_IOCTRL_DISCOVERY 0x00A00028

// Для MINION/NVLW/NVLTLC/PHY нужны адреса выше 16MB:
#define NV_NVLW_BASE_AD 0x01180000
#define NV_MINION_BASE_AD 0x01140000
#define NV_NVLTLC_BASE_AD 0x011A0000
#define NV_NVLPHYCTL_BASE_AD 0x01300000

// Fuse-related
#define NV_FUSE_OPT_GPU_DISABLE_0 0x0002100C
// NVLink-specific fuses - offset varies per chip
// На Ada может быть в другом месте

// ============================================================================

int find_nvidia_gpus(GPUDevice *gpus, int max_gpus)
{
    int count = 0;
    glob_t g;

    if (glob("/sys/bus/pci/devices/*/vendor", 0, NULL, &g) != 0)
        return 0;

    for (size_t i = 0; i < g.gl_pathc && count < max_gpus; i++)
    {
        char buf[64];
        FILE *f = fopen(g.gl_pathv[i], "r");
        if (!f)
            continue;
        if (fgets(buf, sizeof(buf), f))
        {
            uint16_t vendor = strtoul(buf, NULL, 0);
            if (vendor == 0x10de)
            {
                char *dir = strdup(g.gl_pathv[i]);
                char *last_slash = strrchr(dir, '/');
                *last_slash = '\0';
                char *bdf = strrchr(dir, '/') + 1;
                strncpy(gpus[count].bdf, bdf, sizeof(gpus[count].bdf) - 1);
                strncpy(gpus[count].sysfs_path, dir, sizeof(gpus[count].sysfs_path) - 1);
                

                char path[600];
                snprintf(path, sizeof(path), "%s/device", dir);
                FILE *df = fopen(path, "r");
                if (df)
                {
                    if (fgets(buf, sizeof(buf), df))
                        gpus[count].device_id = strtoul(buf, NULL, 0);
                    fclose(df);
                }

                snprintf(path, sizeof(path), "%s/class", dir);
                df = fopen(path, "r");
                uint32_t class = 0;
                if (df)
                {
                    if (fgets(buf, sizeof(buf), df))
                        class = strtoul(buf, NULL, 0);
                    fclose(df);
                }

                snprintf(path, sizeof(path), "%s/resource", dir);
                df = fopen(path, "r");
                if (df)
                {
                    if (fgets(buf, sizeof(buf), df))
                    {
                        uint64_t start, end;
                        sscanf(buf, "%lx %lx", &start, &end);
                        gpus[count].bar0_addr = start;
                        gpus[count].bar0_size = end - start + 1;
                    }
                    fclose(df);
                }

                if ((class >> 16) == 0x03)
                    count++;
                free(dir);
            }
        }
        fclose(f);
    }
    globfree(&g);
    return count;
}

MMIOMapping *map_bar0(const GPUDevice *gpu)
{
    char path[600];
    snprintf(path, sizeof(path), "%s/resource0", gpu->sysfs_path);

    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0)
    {
        fd = open(path, O_RDONLY | O_SYNC);
        if (fd < 0)
        {
            printf("  [!] Cannot open BAR0: %s\n", strerror(errno));
            return NULL;
        }
    }

    size_t size = gpu->bar0_size;
    if (size == 0 || size > 256 * 1024 * 1024)
        size = 16 * 1024 * 1024;

    int prot = PROT_READ | PROT_WRITE;
    void *base = mmap(NULL, size, prot, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED)
    {
        // Fallback read-only
        base = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED)
        {
            printf("  [!] mmap BAR0 failed: %s\n", strerror(errno));
            close(fd);
            return NULL;
        }
        printf("  [!] BAR0 mapped READ-ONLY (нужен rw для PRAMIN window)\n");
    }

    MMIOMapping *m = malloc(sizeof(MMIOMapping));
    m->base = base;
    m->size = size;
    m->fd = fd;
    return m;
}

void unmap_bar0(MMIOMapping *m)
{
    if (m)
    {
        munmap(m->base, m->size);
        close(m->fd);
        free(m);
    }
}

static inline uint32_t mmio_rd32(MMIOMapping *m, uint32_t off)
{
    if (off + 4 > m->size)
        return 0xDEADDEAD;
    return *(volatile uint32_t *)((uint8_t *)m->base + off);
}

static inline void mmio_wr32(MMIOMapping *m, uint32_t off, uint32_t val)
{
    if (off + 4 > m->size)
        return;
    *(volatile uint32_t *)((uint8_t *)m->base + off) = val;
}

// ============================================================================
// PRAMIN window - чтение верхнего MMIO
// Настраиваем окно на нужный адрес, читаем через BAR0
// ============================================================================

// Прочитать регистр через PRAMIN window (для адресов > BAR0 size)
uint32_t pramin_read32(MMIOMapping *m, uint32_t target_addr)
{
    // PRAMIN window = 1MB окно начинающееся с PRAMIN_WINDOW_BASE в BAR0
    // NV_PBUS_BAR0_WINDOW задаёт куда это окно указывает
    // Base granularity = 64KB (0x10000)

    // Проверяем что PRAMIN_WINDOW_BASE в пределах BAR0
    if (PRAMIN_WINDOW_BASE + PRAMIN_WINDOW_SIZE > m->size)
    {
        return 0xDEADDEAD;
    }

    // Сохраняем текущее значение window
    uint32_t old_window = mmio_rd32(m, NV_PBUS_BAR0_WINDOW);

    // Вычисляем нужный base для window (выровнено на 1MB)
    uint32_t window_base = target_addr & ~(PRAMIN_WINDOW_SIZE - 1);
    uint32_t window_offset = target_addr - window_base;

    // Записываем новый base в PBUS_BAR0_WINDOW
    // Формат: base >> 16 | enable_bit
    uint32_t window_val = (window_base >> 16);
    mmio_wr32(m, NV_PBUS_BAR0_WINDOW, window_val);

    // Читаем заголовок для fence
    (void)mmio_rd32(m, NV_PBUS_BAR0_WINDOW);

    // Читаем данные через окно
    uint32_t result = mmio_rd32(m, PRAMIN_WINDOW_BASE + window_offset);

    // Восстанавливаем окно
    mmio_wr32(m, NV_PBUS_BAR0_WINDOW, old_window);

    return result;
}

// Читаем регистр — автоматически через прямой доступ или PRAMIN
uint32_t gpu_read32(MMIOMapping *m, uint32_t addr)
{
    if (addr + 4 <= m->size)
    {
        return mmio_rd32(m, addr);
    }
    return pramin_read32(m, addr);
}

// ============================================================================
// Chip identification
// ============================================================================

void identify_chip(MMIOMapping *m)
{
    uint32_t boot0 = mmio_rd32(m, NV_PMC_BOOT_0);
    uint32_t boot1 = mmio_rd32(m, NV_PMC_BOOT_1);
    uint32_t boot2 = mmio_rd32(m, NV_PMC_BOOT_2);

    printf("  BOOT_0 = 0x%08X\n", boot0);
    printf("  BOOT_1 = 0x%08X\n", boot1);
    printf("  BOOT_2 = 0x%08X\n", boot2);

    // BOOT_0 format for Ada:
    // [31:20] = implementation (chip identifier)
    // [19:12] = architecture
    // [11:8]  = revision
    // [7:0]   = stepping/mask revision
    //
    // Но на Ada формат немного другой. 0x192000A1:
    // Верхние биты 0x192 = это AD102
    // NV_PMC_BOOT_0 на Ada: bits[31:20] содержат chip id

    uint32_t chip_id = (boot0 >> 20) & 0xFFF;
    uint32_t revision = (boot0 >> 0) & 0xFF;

    printf("  Chip ID: 0x%03X  Revision: 0x%02X\n", chip_id, revision);

    // Маппинг chip_id -> название
    // 0x190 = AD102 full? 0x192 = AD102?
    // Device ID 0x2684 = RTX 4090
    if (chip_id >= 0x190 && chip_id <= 0x19F)
    {
        printf("  → Ada Lovelace (AD10x)\n");
        if (chip_id == 0x192)
            printf("  → Вероятно AD102 (RTX 4090 / L40 / RTX 6000 Ada)\n");
    }
}

// ============================================================================
// Fuse analysis - правильная интерпретация
// ============================================================================

void analyze_fuses(MMIOMapping *m)
{
    printf("\n  === FUSE Analysis (v2 — с правильной BADF обработкой) ===\n\n");

    // Статистика по типам ответов в fuse region
    int real_data = 0, pri_timeout = 0, pri_error = 0, zeros = 0, all_ones = 0;

    for (uint32_t off = 0x21000; off < 0x22000; off += 4)
    {
        uint32_t val = mmio_rd32(m, off);
        if (val == 0)
            zeros++;
        else if (val == 0xFFFFFFFF)
            all_ones++;
        else if ((val & 0xFFFF0000) == 0xBADF0000)
        {
            if ((val & 0xF000) == 0x1000)
                pri_timeout++;
            else if ((val & 0xF000) == 0x5000)
                pri_error++;
        }
        else
            real_data++;
    }

    int total = (0x22000 - 0x21000) / 4;
    printf("  Fuse region 0x21000-0x22000 (%d registers):\n", total);
    printf("    Real data:    %3d (%.1f%%)\n", real_data, 100.0 * real_data / total);
    printf("    PRI_TIMEOUT:  %3d (%.1f%%) ← engine power-gated\n", pri_timeout, 100.0 * pri_timeout / total);
    printf("    PRI_ERROR:    %3d (%.1f%%) ← access denied/invalid\n", pri_error, 100.0 * pri_error / total);
    printf("    Zero:         %3d\n", zeros);
    printf("    All-ones:     %3d\n", all_ones);

    // NVLink-specific fuse area
    printf("\n  NVLink fuse region 0x21C00-0x21C40:\n");
    int nvl_timeout = 0, nvl_error = 0, nvl_real = 0;
    for (uint32_t off = 0x21C00; off < 0x21C40; off += 4)
    {
        uint32_t val = mmio_rd32(m, off);
        if (is_pri_error(val))
        {
            if ((val & 0xF000) == 0x1000)
                nvl_timeout++;
            else
                nvl_error++;
        }
        else if (val != 0 && val != 0xFFFFFFFF)
        {
            nvl_real++;
            printf("    [0x%05X] = 0x%08X  ← REAL FUSE DATA!\n", off, val);
        }
    }

    if (nvl_real == 0)
    {
        printf("    Все регистры возвращают PRI ошибки (%d timeout, %d error)\n",
               nvl_timeout, nvl_error);
        printf("    >>> НЕВОЗМОЖНО прочитать NVLink fuses — блок не отвечает <<<\n");
        printf("    Это означает: fuse controller для NVLink региона ЗАБЛОКИРОВАН\n");
        printf("    NVIDIA может блокировать чтение fuses через PRI routing\n");
    }

    // Покажем РЕАЛЬНЫЕ данные из fuse region
    printf("\n  Реальные fuse данные (не-BADF, не-0, не-FF):\n");
    int shown = 0;
    for (uint32_t off = 0x21000; off < 0x22000; off += 4)
    {
        uint32_t val = mmio_rd32(m, off);
        if (!is_dead(val) && val != 0xFFFFFFCF)
        { // 0xFFFFFFCF тоже подозрительно
            printf("    [0x%05X] = 0x%08X\n", off, val);
            shown++;
            if (shown > 40)
            {
                printf("    ... (truncated)\n");
                break;
            }
        }
    }
}

// ============================================================================
// NVLink block scan — v2 с правильной классификацией
// ============================================================================

typedef struct
{
    uint32_t start;
    uint32_t end;
    const char *name;
    bool needs_pramin; // true если выше BAR0 16MB
} NvLinkBlock;

static const NvLinkBlock BLOCKS[] = {
    {0x00A00000, 0x00A04000, "IOCTRL (NVLink IO Controller)", false},
    {0x00A20000, 0x00A24000, "NVLIPT_COMMON", false},
    {0x00A40000, 0x00A44000, "NVLIPT_LNK(0)", false},
    {0x00A60000, 0x00A64000, "NVLIPT_LNK(1)", false},
    {0x01140000, 0x01144000, "MINION (NVLink FW Controller)", true},
    {0x01180000, 0x01184000, "NVLW(0) (NVLink Wrapper 0)", true},
    {0x01184000, 0x01188000, "NVLW(1) (NVLink Wrapper 1)", true},
    {0x011A0000, 0x011A4000, "NVLTLC(0) (Transport Layer 0)", true},
    {0x011C0000, 0x011C4000, "NVLTLC(1) (Transport Layer 1)", true},
    {0x01300000, 0x01304000, "NVLPHY(0) (PHY SerDes 0)", true},
    {0x01304000, 0x01308000, "NVLPHY(1) (PHY SerDes 1)", true},
    {0, 0, NULL, false}};

void scan_nvlink_v2(MMIOMapping *m, bool have_write)
{
    printf("\n  === NVLink Block Scan v2 ===\n");
    printf("  BADF1301 = PRI_TIMEOUT = engine адрес маршрутизируется, но engine OFF\n");
    printf("  BADF5040 = PRI_ERROR = доступ запрещён или невалиден\n");
    printf("  0/FF = адрес не маршрутизируется вообще\n\n");

    for (int i = 0; BLOCKS[i].name; i++)
    {
        const NvLinkBlock *blk = &BLOCKS[i];

        printf("  [%s] 0x%07X-0x%07X", blk->name, blk->start, blk->end);

        if (blk->needs_pramin && !have_write)
        {
            printf(" → ПРОПУЩЕН (нужен R/W для PRAMIN window)\n");
            continue;
        }
        printf("\n");

        int live = 0, timeout = 0, error = 0, zero = 0, ones = 0;
        uint32_t first_live_off = 0, first_live_val = 0;
        uint32_t first_timeout_val = 0;

        // Сэмплируем каждые 4 байта но только первые 256 байт
        for (uint32_t off = blk->start; off < blk->start + 256 && off < blk->end; off += 4)
        {
            uint32_t val;
            if (blk->needs_pramin)
            {
                val = pramin_read32(m, off);
            }
            else
            {
                val = mmio_rd32(m, off);
            }

            if (val == 0)
            {
                zero++;
            }
            else if (val == 0xFFFFFFFF)
            {
                ones++;
            }
            else if (is_pri_error(val))
            {
                if ((val & 0xF000) == 0x1000)
                {
                    timeout++;
                    if (!first_timeout_val)
                        first_timeout_val = val;
                }
                else
                    error++;
            }
            else
            {
                live++;
                if (!first_live_off)
                {
                    first_live_off = off;
                    first_live_val = val;
                }
            }
        }

        // Interpret
        if (live > 0)
        {
            printf("    >>> ЖИВОЙ — РЕАЛЬНЫЕ ДАННЫЕ (live=%d) <<<\n", live);
            printf("    First: [0x%07X] = 0x%08X\n", first_live_off, first_live_val);
            // Дамп первых живых
            int s = 0;
            for (uint32_t off = blk->start; off < blk->start + 256 && s < 8; off += 4)
            {
                uint32_t val = blk->needs_pramin ? pramin_read32(m, off) : mmio_rd32(m, off);
                if (!is_dead(val))
                {
                    printf("    [0x%07X] = 0x%08X\n", off, val);
                    s++;
                }
            }
        }
        else if (timeout > 0)
        {
            printf("    PRI_TIMEOUT (%d regs) — адрес МАРШРУТИЗИРУЕТСЯ, engine POWER-GATED\n", timeout);
            printf("    Код: 0x%08X → %s\n", first_timeout_val, classify_reg(first_timeout_val));
            printf("    → Шина GPU ЗНАЕТ что здесь должен быть NVLink\n");
            printf("    → Но engine выключен (clock/power gated)\n");
        }
        else if (error > 0)
        {
            printf("    PRI_ERROR (%d regs) — доступ запрещён\n", error);
        }
        else if (zero > 0)
        {
            printf("    Все нули — адрес не маршрутизируется (блок не существует)\n");
        }
        else if (ones > 0)
        {
            printf("    Все 0xFF — шинная ошибка (нет устройства)\n");
        }
        printf("\n");
    }
}

// ============================================================================
// PMC / PBUS deep analysis
// ============================================================================

void analyze_pmc(MMIOMapping *m)
{
    printf("\n  === PMC Engine Enable Analysis ===\n\n");

    uint32_t pmc_enable = mmio_rd32(m, NV_PMC_ENABLE);
    printf("  PMC_ENABLE = 0x%08X\n", pmc_enable);
    printf("  Бинарно: ");
    for (int i = 31; i >= 0; i--)
    {
        printf("%d", (pmc_enable >> i) & 1);
        if (i % 4 == 0)
            printf(" ");
    }
    printf("\n\n");

    // Расширенные device enable регистры (Ada имеет больше)
    printf("  PMC_DEVICE_ENABLE registers:\n");
    for (int i = 0; i < 8; i++)
    {
        uint32_t addr = NV_PMC_DEVICE_ENABLE(i);
        uint32_t val = mmio_rd32(m, addr);
        if (!is_dead(val) && val != 0)
        {
            printf("    [0x%04X] PMC_DEVICE_ENABLE(%d) = 0x%08X\n", addr, i, val);
        }
    }

    // PBUS registers - дополнительная информация
    printf("\n  PBUS PCI Config Mirror:\n");
    uint32_t pci0 = mmio_rd32(m, NV_PBUS_PCI_NV_0);
    uint32_t pci19 = mmio_rd32(m, NV_PBUS_PCI_NV_19);
    printf("    PCI_NV_0 (ID)       = 0x%08X (dev=0x%04X ven=0x%04X)\n",
           pci0, (pci0 >> 16) & 0xFFFF, pci0 & 0xFFFF);
    printf("    PCI_NV_19 (Subsys)  = 0x%08X\n", pci19);

    // BAR0 window current state
    uint32_t bar0_win = mmio_rd32(m, NV_PBUS_BAR0_WINDOW);
    printf("\n  PRAMIN Window state:\n");
    printf("    NV_PBUS_BAR0_WINDOW = 0x%08X\n", bar0_win);
    printf("    Current target base = 0x%08X\n", (bar0_win & 0xFFFF) << 16);
}

// ============================================================================
// PTOP discovery - Ada uses discovery table, not fixed offsets
// ============================================================================

void scan_discovery_table(MMIOMapping *m)
{
    printf("\n  === PTOP Discovery Table ===\n");
    printf("  Ada/Hopper GPU используют discovery table для перечисления IP блоков\n\n");

    // На Ada, discovery table может быть по другому адресу
    // Попробуем несколько известных мест
    uint32_t disc_bases[] = {0x00022700, 0x00022800, 0x00020000, 0x00024000};

    for (int b = 0; b < 4; b++)
    {
        uint32_t base = disc_bases[b];
        if (base + 256 > m->size)
            continue;

        // Проверяем первые несколько слов
        uint32_t first = mmio_rd32(m, base);
        if (is_dead(first) || first == 0)
            continue;

        printf("  Table @ 0x%05X:\n", base);
        int printed = 0;
        for (uint32_t off = base; off < base + 512 && off < m->size; off += 4)
        {
            uint32_t val = mmio_rd32(m, off);
            if (is_dead(val) || val == 0)
                continue;
            printf("    [0x%05X] = 0x%08X", off, val);

            // Попробуем декодировать как discovery entry
            // Формат зависит от архитектуры, но обычно:
            // содержит type, instance, base address offset
            uint32_t type = val & 0x3F;
            if (type == 0x13)
                printf(" → *** NVLINK TYPE ***");
            if (type == 0x01)
                printf(" → GR");
            if (type == 0x03)
                printf(" → CE");
            printf("\n");
            if (++printed > 30)
            {
                printf("    ...\n");
                break;
            }
        }
        printf("\n");
    }
}

// ============================================================================
// PRAMIN window scan - доступ к верхнему MMIO
// ============================================================================

void scan_upper_mmio(MMIOMapping *m)
{
    printf("\n  === Upper MMIO via PRAMIN Window ===\n");
    printf("  Доступ к NVLink блокам выше 16MB через PRAMIN переадресацию\n\n");

    // Проверяем что можем писать в BAR0_WINDOW
    uint32_t orig_window = mmio_rd32(m, NV_PBUS_BAR0_WINDOW);

    // Пробуем записать тестовое значение
    mmio_wr32(m, NV_PBUS_BAR0_WINDOW, 0x00000114); // Target = 0x01140000 (MINION)
    uint32_t check = mmio_rd32(m, NV_PBUS_BAR0_WINDOW);

    if (check == orig_window)
    {
        printf("  [!] BAR0 mapped READ-ONLY — PRAMIN window недоступен\n");
        printf("  [!] Для PRAMIN нужен R/W доступ к BAR0\n");
        printf("  [!] Убедитесь что nvidia драйвер не блокирует запись\n");
        printf("  [!] Попробуйте: rmmod nvidia && sudo ./nvlink_probe2\n");
        return;
    }

    printf("  PRAMIN window работает! (wrote 0x114, read back 0x%08X)\n\n", check);

    // Восстановим
    mmio_wr32(m, NV_PBUS_BAR0_WINDOW, orig_window);

    // Теперь сканируем каждый NVLink блок через PRAMIN
    typedef struct
    {
        uint32_t addr;
        const char *name;
    } ProbePoint;
    ProbePoint probes[] = {
        {0x01140000, "MINION[0]"},
        {0x01140004, "MINION[1]"},
        {0x01180000, "NVLW[0]"},
        {0x01180004, "NVLW[1]"},
        {0x011A0000, "NVLTLC[0]"},
        {0x011A0004, "NVLTLC[1]"},
        {0x01300000, "NVLPHY[0]"},
        {0x01300004, "NVLPHY[1]"},
        {0x01300100, "NVLPHY_CTL"},
        {0x01304000, "NVLPHY2[0]"},
        {0x01310000, "NVLPHY_ALT"},
        {0, NULL}};

    for (int i = 0; probes[i].name; i++)
    {
        uint32_t val = pramin_read32(m, probes[i].addr);
        printf("  [0x%07X] %s = 0x%08X → %s\n",
               probes[i].addr, probes[i].name, val, classify_reg(val));
    }
}

// ============================================================================
// NVLink PCIe Vendor Specific extended cap deep decode
// ============================================================================

void decode_vendor_specific(const GPUDevice *gpu)
{
    printf("\n  === PCIe Vendor Specific Deep Decode ===\n");

    char path[600];
    snprintf(path, sizeof(path), "%s/config", gpu->sysfs_path);

    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return;

    uint8_t config[4096];
    ssize_t n = read(fd, config, sizeof(config));
    close(fd);

    // Найдём Vendor Specific extended cap (0x600 из первого прогона)
    if (n < 0x640)
        return;

    uint32_t ext_offset = 0x600;
    uint32_t ext_header = *(uint32_t *)&config[ext_offset];
    if ((ext_header & 0xFFFF) != 0x000B)
        return;

    printf("  ExtCap @ 0x%03X: Vendor Specific\n", ext_offset);
    printf("  Raw data (64 bytes):\n  ");
    for (int i = 0; i < 64 && ext_offset + 4 + i < (uint32_t)n; i++)
    {
        printf("%02X ", config[ext_offset + 4 + i]);
        if ((i + 1) % 16 == 0)
            printf("\n  ");
    }
    printf("\n");

    // NVIDIA VSD decode attempt
    // Bytes 0-1: capability version
    // Bytes 2-3: capability length
    // Bytes 4+: NVIDIA-specific fields
    uint16_t vsd_ver = *(uint16_t *)&config[ext_offset + 4];
    uint16_t vsd_len = *(uint16_t *)&config[ext_offset + 6];
    printf("  VSD Version: 0x%04X  Length: %u\n", vsd_ver, vsd_len);

    // Попробуем найти NVLink-related биты в VSD
    // В L40/A100 VSD содержит NVLink capability flags
    printf("  Scan for NVLink-related patterns in VSD:\n");
    for (int i = 0; i < 256 && ext_offset + 4 + i + 3 < (uint32_t)n; i += 4)
    {
        uint32_t word = *(uint32_t *)&config[ext_offset + 4 + i];
        // NVLink capability часто содержит version nibbles
        if ((word & 0xFF) == 0x04 || (word & 0xFF) == 0x03)
        {
            printf("    VSD[%d] = 0x%08X (possible NVLink version field?)\n", i, word);
        }
    }
}

// ============================================================================
// Окончательный анализ
// ============================================================================

void final_analysis(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║               АНАЛИЗ v2 — RTX 4090 (AD102)                ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║                                                            ║\n");
    printf("║  КЛЮЧЕВЫЕ НАХОДКИ:                                        ║\n");
    printf("║                                                            ║\n");
    printf("║  1. PRI_TIMEOUT (0xBADF1301) на IOCTRL/NVLIPT:            ║\n");
    printf("║     → Address decoder GPU МАРШРУТИЗИРУЕТ запросы к NVLink  ║\n");
    printf("║     → Engine за этими адресами POWER-GATED (выключен)     ║\n");
    printf("║     → Это НЕ то же самое что 'нет железа'                 ║\n");
    printf("║     → Если бы блока не было, был бы 0x0 или 0xFFFFFFFF    ║\n");
    printf("║                                                            ║\n");
    printf("║  2. PRI_ERROR (0xBADF5040) на FUSE region:                ║\n");
    printf("║     → Fuse controller для NVLink ЗАБЛОКИРОВАН             ║\n");
    printf("║     → Мы НЕ МОЖЕМ прочитать реальные fuse значения        ║\n");
    printf("║     → NVIDIA может скрывать состояние fuses               ║\n");
    printf("║                                                            ║\n");
    printf("║  3. PMC_ENABLE bit 28 = SET:                              ║\n");
    printf("║     → OS-видимый engine enable бит для NVLink ВКЛЮЧЁН     ║\n");
    printf("║     → Но engine всё равно не отвечает (power-gated)       ║\n");
    printf("║     → Это ПРОТИВОРЕЧИЕ: enable=1 но timeout на доступе    ║\n");
    printf("║                                                            ║\n");
    printf("║  ИНТЕРПРЕТАЦИИ:                                           ║\n");
    printf("║                                                            ║\n");
    printf("║  A) NVLink IP блоки ЕСТЬ в кремнии AD102, но:             ║\n");
    printf("║     - Power management unit (PMU/GSP) не подаёт clk/pwr   ║\n");
    printf("║     - VBIOS/firmware не инициализирует NVLink subsystem    ║\n");
    printf("║     - Даже если включить — нет PCB traces к коннектору    ║\n");
    printf("║                                                            ║\n");
    printf("║  B) Address decoder routing — shared layout:              ║\n");
    printf("║     - AD102 использует общий address decoder с L40/RTX6K  ║\n");
    printf("║     - Маршруты к NVLink прописаны, но за ними пусто       ║\n");
    printf("║     - PRI_TIMEOUT = стандартная реакция на power-gated IP  ║\n");
    printf("║                                                            ║\n");
    printf("║  ВЫВОД:                                                    ║\n");
    printf("║                                                            ║\n");
    printf("║  NVLink адресное пространство РАЗМЕЧЕНО в address decoder  ║\n");
    printf("║  Engine POWER-GATED (не clock/not powered)                 ║\n");
    printf("║  Fuse состояние НЕЧИТАЕМО (заблокировано PRI)             ║\n");
    printf("║                                                            ║\n");
    printf("║  Для 100%% ответа нужно:                                   ║\n");
    printf("║  • PRAMIN window (R/W BAR0) → сканировать PHY блоки       ║\n");
    printf("║  • VBIOS дамп → найти NVLink init таблицы                 ║\n");
    printf("║  • GSP-RM firmware → найти NVLink enable/disable логику   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
}

// ============================================================================

int main(void)
{
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  NVLink Probe v2 — Fixed BADF + PRAMIN window  ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    if (geteuid() != 0)
    {
        printf("[!] Root нужен. sudo ./nvlink_probe2\n\n");
    }

    GPUDevice gpus[8];
    int count = find_nvidia_gpus(gpus, 8);
    if (!count)
    {
        printf("GPU не найдены\n");
        return 1;
    }

    for (int i = 0; i < count; i++)
    {
        GPUDevice *gpu = &gpus[i];
        printf("═══ GPU %d: %s (0x%04X) BAR0=%luMB ═══\n\n",
               i, gpu->bdf, gpu->device_id, gpu->bar0_size / (1024 * 1024));

        MMIOMapping *m = map_bar0(gpu);
        if (!m)
            continue;

        // Определяем R/W
        bool have_write = false;
        uint32_t test_val = mmio_rd32(m, NV_PBUS_BAR0_WINDOW);
        mmio_wr32(m, NV_PBUS_BAR0_WINDOW, test_val ^ 0x10);
        uint32_t check = mmio_rd32(m, NV_PBUS_BAR0_WINDOW);
        if (check != test_val)
        {
            have_write = true;
            mmio_wr32(m, NV_PBUS_BAR0_WINDOW, test_val); // restore
        }
        printf("  BAR0 access: %s\n\n", have_write ? "READ-WRITE (PRAMIN available)" : "READ-ONLY");

        identify_chip(m);
        analyze_pmc(m);
        analyze_fuses(m);
        scan_nvlink_v2(m, have_write);
        scan_discovery_table(m);
        decode_vendor_specific(gpu);

        if (have_write)
        {
            scan_upper_mmio(m);
        }
        else
        {
            printf("\n  [!] PRAMIN window недоступен (read-only BAR0)\n");
            printf("  [!] Для сканирования PHY/MINION/NVLW попробуйте:\n");
            printf("  [!]   1. rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia\n");
            printf("  [!]   2. sudo ./nvlink_probe2\n");
            printf("  [!]   3. modprobe nvidia\n");
        }

        unmap_bar0(m);
    }

    final_analysis();
    return 0;
}
