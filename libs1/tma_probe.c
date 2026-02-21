/*
 * tma_probe.c - Проверка наличия TMA (Tensor Memory Accelerator) на RTX 4090
 *
 * TMA = аппаратный DMA-движок для асинхронных тензорных копий
 * Официально: только Hopper (sm_90) и Blackwell (sm_100)
 * Вопрос: есть ли скрытый TMA в AD102 (sm_89)?
 *
 * Проверяем:
 *   1. MMIO регистры TMA engine
 *   2. Discovery table — полная декодировка всех IP блоков
 *   3. Копировальные движки (CE) — TMA может быть расширением CE
 *   4. SM capabilities через MMIO
 *   5. Скрытые engine типы в PTOP
 *
 * gcc -O2 -o tma_probe tma_probe.c
 * sudo ./tma_probe
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <glob.h>
#include <stdbool.h>

// ============================================================================
// PRI error classification (from v2)
// ============================================================================

static bool is_pri_error(uint32_t val)
{
    return (val & 0xFFFF0000) == 0xBADF0000;
}

static bool is_dead(uint32_t val)
{
    return val == 0 || val == 0xFFFFFFFF || val == 0xDEADDEAD || is_pri_error(val);
}

static const char *classify_val(uint32_t val)
{
    if (val == 0)
        return "UNMAPPED";
    if (val == 0xFFFFFFFF)
        return "BUS_ERROR";
    if (val == 0xDEADDEAD)
        return "OUT_OF_RANGE";
    if ((val & 0xFFFF0000) == 0xBADF0000)
    {
        if ((val & 0xF000) == 0x1000)
            return "PRI_TIMEOUT(power-gated)";
        if ((val & 0xF000) == 0x5000)
            return "PRI_ERROR(denied)";
        return "PRI_FAULT";
    }
    return "LIVE";
}

// ============================================================================
// GPU & MMIO
// ============================================================================

typedef struct
{
    char bdf[32];
    uint16_t device_id;
    uint64_t bar0_size;
    char sysfs_path[512];
} GPU;

typedef struct
{
    void *base;
    size_t size;
} MMIO;

#define NV_PBUS_BAR0_WINDOW 0x00001700
#define PRAMIN_BASE 0x00700000
#define PRAMIN_SIZE 0x00100000

static inline uint32_t rd32(MMIO *m, uint32_t off)
{
    if (off + 4 > m->size)
        return 0xDEADDEAD;
    return *(volatile uint32_t *)((uint8_t *)m->base + off);
}

static inline void wr32(MMIO *m, uint32_t off, uint32_t val)
{
    if (off + 4 > m->size)
        return;
    *(volatile uint32_t *)((uint8_t *)m->base + off) = val;
}

uint32_t pramin_rd32(MMIO *m, uint32_t target)
{
    if (PRAMIN_BASE + PRAMIN_SIZE > m->size)
        return 0xDEADDEAD;
    uint32_t old = rd32(m, NV_PBUS_BAR0_WINDOW);
    uint32_t wbase = target & ~(PRAMIN_SIZE - 1);
    uint32_t woff = target - wbase;
    wr32(m, NV_PBUS_BAR0_WINDOW, wbase >> 16);
    (void)rd32(m, NV_PBUS_BAR0_WINDOW);
    uint32_t result = rd32(m, PRAMIN_BASE + woff);
    wr32(m, NV_PBUS_BAR0_WINDOW, old);
    return result;
}

// Auto-select direct or PRAMIN
uint32_t gpu_rd32(MMIO *m, uint32_t addr)
{
    if (addr + 4 <= m->size)
        return rd32(m, addr);
    return pramin_rd32(m, addr);
}

int find_gpu(GPU *gpu)
{
    glob_t g;
    if (glob("/sys/bus/pci/devices/*/vendor", 0, NULL, &g) != 0)
        return 0;
    for (size_t i = 0; i < g.gl_pathc; i++)
    {
        char buf[64];
        FILE *f = fopen(g.gl_pathv[i], "r");
        if (!f)
            continue;
        if (fgets(buf, sizeof(buf), f) && strtoul(buf, NULL, 0) == 0x10de)
        {
            fclose(f);
            char *dir = strdup(g.gl_pathv[i]);
            *strrchr(dir, '/') = 0;
            strncpy(gpu->sysfs_path, dir, sizeof(gpu->sysfs_path) - 1);
            strncpy(gpu->bdf, strrchr(dir, '/') + 1, sizeof(gpu->bdf) - 1);
            char path[600];
            snprintf(path, sizeof(path), "%s/device", dir);
            FILE *df = fopen(path, "r");
            if (df)
            {
                if (fgets(buf, sizeof(buf), df))
                    gpu->device_id = strtoul(buf, NULL, 0);
                fclose(df);
            }
            snprintf(path, sizeof(path), "%s/class", dir);
            df = fopen(path, "r");
            uint32_t cls = 0;
            if (df)
            {
                if (fgets(buf, sizeof(buf), df))
                    cls = strtoul(buf, NULL, 0);
                fclose(df);
            }
            snprintf(path, sizeof(path), "%s/resource", dir);
            df = fopen(path, "r");
            if (df)
            {
                uint64_t s, e;
                if (fgets(buf, sizeof(buf), df))
                {
                    sscanf(buf, "%lx %lx", &s, &e);
                    gpu->bar0_size = e - s + 1;
                }
                fclose(df);
            }
            free(dir);
            if ((cls >> 16) == 0x03)
            {
                globfree(&g);
                return 1;
            }
        }
        else
            fclose(f);
    }
    globfree(&g);
    return 0;
}

MMIO *map_bar0(GPU *gpu)
{
    char path[600];
    snprintf(path, sizeof(path), "%s/resource0", gpu->sysfs_path);
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0)
        fd = open(path, O_RDONLY | O_SYNC);
    if (fd < 0)
    {
        printf("[!] BAR0: %s\n", strerror(errno));
        return NULL;
    }
    size_t sz = gpu->bar0_size;
    if (sz == 0 || sz > 256 * 1024 * 1024)
        sz = 16 * 1024 * 1024;
    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED)
        p = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED)
    {
        close(fd);
        return NULL;
    }
    MMIO *m = malloc(sizeof(MMIO));
    m->base = p;
    m->size = sz;
    close(fd);
    return m;
}

// ============================================================================
// Discovery Table Full Decode
//
// Формат Ada discovery table (из open-gpu-kernel-modules):
// Entry types в верхних битах определяют формат
//
// Из gpu/gpu_device_table.h, каждая запись — тройка слов:
//   Word 0: [31:28]=entry_type [27:16]=chain/flags [15:0]=info
//   Word 1: [31:30]=type [29:20]=addr_hi [19:0]=addr_lo
//   Word 2: misc
//
// IP Type IDs (из nvlinkip_discovery.h и hw manuals):
// ============================================================================

typedef struct
{
    int type_id;
    const char *name;
    const char *description;
} IPType;

// Полный список известных IP type IDs на Ada/Hopper
static const IPType IP_TYPES[] = {
    {0x01, "GR", "Graphics Engine"},
    {0x02, "SEC", "Security Engine"},
    {0x03, "CE", "Copy Engine (DMA)"},
    {0x04, "DISPLAY", "Display Controller"},
    {0x05, "HDACODEC", "HDA Audio Codec"},
    {0x06, "MSPDEC", "Media SPU Decoder"},
    {0x07, "MSPPP", "Media SPU Post-Processor"},
    {0x08, "MSVLD", "Media SPU VLD"},
    {0x09, "SEC2", "Security Engine v2"},
    {0x0A, "PERFMON", "Performance Monitor"},
    {0x0B, "BUS", "Bus Interface"},
    {0x0C, "PMU", "Power Management Unit"},
    {0x0D, "PRI_MASTER", "PRI Master Hub"},
    {0x0E, "NVENC", "Video Encoder (NVENC)"},
    {0x0F, "NVDEC", "Video Decoder (NVDEC)"},
    {0x10, "GSP", "GPU System Processor"},
    {0x11, "FBHUB", "Framebuffer Hub"},
    {0x12, "LTC", "L2 Cache (Last-level Texture Cache)"},
    {0x13, "NVLINK", "NVLink Controller"},
    {0x14, "NVJPEG", "JPEG Decoder"},
    {0x15, "OFA", "Optical Flow Accelerator"},
    {0x16, "RESERVED16", "(Reserved)"},
    {0x17, "HSHUB", "High-Speed Hub"},
    {0x18, "FBFLCN", "FB Falcon"},
    {0x19, "MIG", "Multi-Instance GPU"},
    {0x1A, "SMBPBI", "SMBus PBI"},
    {0x1B, "XBAR", "Crossbar"},
    {0x1C, "NVD", "NVDebug"},
    {0x1D, "PRI_SLAVE", "PRI Slave Hub"},
    {0x1E, "CTXSW", "Context Switch"},
    {0x1F, "TMA", "*** TENSOR MEMORY ACCELERATOR ***"}, // !!! Возможный TMA type
    {0x20, "CTA", "Compute Thread Array"},
    {0x21, "GRCE", "GR Copy Engine"},
    {0x22, "FBPA", "FB Partition Agent"},
    {0x23, "HBMFLCN", "HBM Falcon"},
    {0x24, "FSP", "Firmware Security Processor"},
    {0x25, "PCIE", "PCIe Controller"},
    {0x26, "DPAUX", "DisplayPort AUX"},
    {0x27, "GPCCS", "GPC Context Switch"},
    {0x28, "FECS", "Front-End Context Switch"},
    // Hopper/Ada specific
    {0x29, "SMPC", "SM Performance Counter"},
    {0x2A, "NVLW", "NVLink Wrapper"},
    {0x2B, "MINION", "NVLink Minion FW"},
    {0x2C, "NVLTLC", "NVLink Transport Layer"},
    {0x2D, "NVLPHY", "NVLink PHY SerDes"},
    {0x2E, "IOCTRL", "NVLink IO Controller"},
    // TMA candidates — несколько возможных type IDs
    {0x30, "DMA_ENGINE", "DMA Engine (TMA candidate)"},
    {0x31, "BULK_COPY", "Bulk Copy (TMA candidate)"},
    {0x32, "TMAUNIT", "TMA Unit (speculative)"},
    {0x40, "UNKNOWN40", "Unknown 0x40"},
    {0, NULL, NULL}};

static const char *lookup_ip_type(int type_id)
{
    for (int i = 0; IP_TYPES[i].name; i++)
    {
        if (IP_TYPES[i].type_id == type_id)
            return IP_TYPES[i].name;
    }
    return NULL;
}

static const char *lookup_ip_desc(int type_id)
{
    for (int i = 0; IP_TYPES[i].name; i++)
    {
        if (IP_TYPES[i].type_id == type_id)
            return IP_TYPES[i].description;
    }
    return NULL;
}

// ============================================================================
// Full discovery table decode
// ============================================================================

void decode_discovery_table(MMIO *m)
{
    printf("\n  === Full Discovery Table Decode ===\n\n");

    // Из v2 вывода мы знаем таблица начинается с 0x22800
    uint32_t base = 0x22800;

    printf("  %-4s %-12s %-8s %-30s %s\n",
           "#", "Raw", "TypeID", "IP Block", "Description");
    printf("  ─────────────────────────────────────────────────────────────────\n");

    // Массив для подсчёта всех найденных типов
    int type_counts[256] = {0};
    int entry_num = 0;

    // Декодируем discovery table
    // Формат entry: каждая запись 3 слова (12 bytes), но формат зависит от entry_type
    //
    // Из open-gpu-kernel nv_discovery.h:
    // Word 0 bit [31:28] = entry type:
    //   0x8 = ENGINE, 0x9 = ENGINE_UNICAST, 0xC = TOP_END
    // Word 0 bit [15:0] = для ENGINE: [15:8]=instance, [7:2]=type, [1:0]=flags
    // Word 1 = address info
    // Word 2 = more address info

    for (int i = 0; i < 256; i++)
    {
        uint32_t w0 = gpu_rd32(m, base + i * 4);
        if (is_dead(w0) || w0 == 0)
            continue;

        // Попробуем несколько схем декодирования

        // Схема 1: entry type в [31:28]
        uint32_t entry_type = (w0 >> 28) & 0xF;

        // Для ENGINE entries (type 0x8, 0x9):
        if (entry_type == 0x8 || entry_type == 0x9)
        {
            uint32_t ip_type = (w0 >> 2) & 0x3F;    // bits [7:2]
            uint32_t instance = (w0 >> 8) & 0xFF;   // bits [15:8]
            uint32_t chain_id = (w0 >> 16) & 0xFFF; // bits [27:16]

            type_counts[ip_type]++;

            const char *name = lookup_ip_type(ip_type);
            const char *desc = lookup_ip_desc(ip_type);

            printf("  %3d  0x%08X  0x%02X/%-2d  %-30s %s\n",
                   entry_num, w0, ip_type, instance,
                   name ? name : "(UNKNOWN)",
                   desc ? desc : "");

            // Подсветка интересных типов
            if (ip_type == 0x13)
            {
                printf("       ^^^^^^^^^^^^^^^^^ NVLink КОНТРОЛЛЕР ^^^^^^^^^^^^^^^^^^^^^\n");
            }
            if (ip_type == 0x1F || ip_type == 0x30 || ip_type == 0x31 || ip_type == 0x32)
            {
                printf("       !!!!!!!! ВОЗМОЖНЫЙ TMA / DMA ENGINE !!!!!!!!\n");
            }
            if (ip_type == 0x03)
            {
                printf("       (CE — Copy Engine, может содержать TMA-like функции)\n");
            }
        }
        // Схема 2: для address/config words
        else if (entry_type == 0xC || entry_type == 0x0)
        {
            // Address/config entry — дополнительная информация
            // Можем попробовать декодировать IP type из нижних бит тоже
            uint32_t alt_type = w0 & 0x3F;
            if (alt_type > 0 && alt_type < 0x40)
            {
                const char *name = lookup_ip_type(alt_type);
                if (name)
                {
                    printf("  %3d  0x%08X  (addr)   ref→%-25s\n", entry_num, w0, name);
                }
            }
        }

        entry_num++;
    }

    // Суммарная таблица по типам
    printf("\n  === IP Type Summary ===\n\n");
    for (int t = 0; t < 256; t++)
    {
        if (type_counts[t] > 0)
        {
            const char *name = lookup_ip_type(t);
            const char *desc = lookup_ip_desc(t);
            printf("  Type 0x%02X: count=%d  %s%s%s\n", t, type_counts[t],
                   name ? name : "UNKNOWN",
                   desc ? " — " : "",
                   desc ? desc : "");

            if (t == 0x1F || t == 0x30 || t == 0x31 || t == 0x32)
            {
                printf("  >>>>>>> ВОЗМОЖНЫЙ TMA! <<<<<<<\n");
            }
        }
    }
}

// ============================================================================
// TMA-specific MMIO regions
//
// На Hopper TMA (cp.async.bulk.tensor) реализован как:
// 1. Часть SM — tensor descriptor decoder в SM pipeline
// 2. CTA-level DMA unit — отдельный от CE
// 3. Shared memory controller расширение
//
// Возможные MMIO locations:
// - SM внутренние регистры (GPC/TPC/SM)
// - FBPA (Framebuffer Partition Agent) — bulk DMA
// - CTA scheduler расширения
// ============================================================================

typedef struct
{
    uint32_t addr;
    const char *name;
} ProbePoint;

void scan_tma_mmio(MMIO *m)
{
    printf("\n  === TMA MMIO Scan ===\n");
    printf("  Сканируем регистры которые на Hopper содержат TMA логику\n\n");

    // SM registers — TPC/SM level
    // На Ada: GPC0 base = ~0x500000, TPC inside GPC = +0x4000 per TPC, SM = +0x400
    ProbePoint sm_regs[] = {
        // GPC0 registers
        {0x00500000, "GPC0 base"},
        {0x00500004, "GPC0 +4"},
        {0x00500200, "GPC0 status"},
        // TPC0 in GPC0
        {0x00504000, "GPC0.TPC0 base"},
        {0x00504004, "GPC0.TPC0 +4"},
        {0x00504200, "GPC0.TPC0 status"},
        {0x00504400, "GPC0.TPC0.SM0 base"},
        {0x00504404, "GPC0.TPC0.SM0 +4"},
        {0x00504410, "GPC0.TPC0.SM0 caps"},
        {0x00504500, "GPC0.TPC0.SM0 +0x100"},
        // SM capability registers — may reveal hidden TMA
        {0x00504600, "GPC0.TPC0.SM0 extended_caps?"},
        {0x00504700, "GPC0.TPC0.SM0 dma_caps?"},
        {0x00504800, "GPC0.TPC0.SM0 tma_region?"},
        {0, NULL}};

    printf("  SM/TPC/GPC Level:\n");
    for (int i = 0; sm_regs[i].name; i++)
    {
        uint32_t val = gpu_rd32(m, sm_regs[i].addr);
        if (!is_dead(val))
        {
            printf("    [0x%06X] %-35s = 0x%08X  ← LIVE\n",
                   sm_regs[i].addr, sm_regs[i].name, val);
        }
        else
        {
            printf("    [0x%06X] %-35s = 0x%08X  (%s)\n",
                   sm_regs[i].addr, sm_regs[i].name, val, classify_val(val));
        }
    }

    // CE (Copy Engine) extended registers — TMA may be CE extension
    printf("\n  Copy Engine (CE) Extended:\n");
    // CE обычно в range 0x104000+
    ProbePoint ce_regs[] = {
        {0x00104000, "CE0 base"},
        {0x00104004, "CE0 +4"},
        {0x00104010, "CE0 caps"},
        {0x00104020, "CE0 config"},
        {0x00104100, "CE0 extended?"},
        {0x00104200, "CE0 tma_config?"},
        {0x00104400, "CE0 bulk_config?"},
        {0x00105000, "CE1 base"},
        {0x00105010, "CE1 caps"},
        {0x00106000, "CE2 base"},
        {0x00106010, "CE2 caps"},
        {0, NULL}};

    for (int i = 0; ce_regs[i].name; i++)
    {
        uint32_t val = gpu_rd32(m, ce_regs[i].addr);
        if (!is_dead(val))
        {
            printf("    [0x%06X] %-35s = 0x%08X  ← LIVE\n",
                   ce_regs[i].addr, ce_regs[i].name, val);
        }
    }

    // FBPA — Framebuffer Partition Agent
    // TMA bulk copies route through FBPA/FBHUB
    printf("\n  FBPA / FBHUB (Framebuffer bulk transfer path):\n");
    ProbePoint fb_regs[] = {
        {0x00100000, "FBHUB base"},
        {0x00100004, "FBHUB +4"},
        {0x00100010, "FBHUB caps"},
        {0x00100100, "FBHUB config"},
        {0x00100200, "FBHUB DMA config?"},
        {0x00100300, "FBHUB bulk_copy?"},
        {0x009A0000, "FBPA0 base"},
        {0x009A0004, "FBPA0 +4"},
        {0x009A0100, "FBPA0 caps"},
        {0x009A0200, "FBPA0 extended?"},
        {0, NULL}};

    for (int i = 0; fb_regs[i].name; i++)
    {
        uint32_t val = gpu_rd32(m, fb_regs[i].addr);
        if (!is_dead(val))
        {
            printf("    [0x%06X] %-35s = 0x%08X  ← LIVE\n",
                   fb_regs[i].addr, fb_regs[i].name, val);
        }
    }

    // CTA / Warp Scheduler level — TMA instructions decoded here
    printf("\n  CTA / Warp Scheduler (instruction decode path):\n");
    ProbePoint cta_regs[] = {
        {0x00419000, "GR_PRI_CTA base?"},
        {0x00419004, "GR_PRI_CTA +4"},
        {0x00419100, "CTA_scheduler?"},
        {0x00419200, "CTA_dma?"},
        {0x00419300, "CTA_tma_unit?"},
        {0x00418000, "GR_PRI_GPCS base"},
        {0x00418004, "GR_PRI_GPCS +4"},
        {0x00418100, "GPCS_config"},
        {0x00418200, "GPCS_smcaps?"},
        {0, NULL}};

    for (int i = 0; cta_regs[i].name; i++)
    {
        uint32_t val = gpu_rd32(m, cta_regs[i].addr);
        if (!is_dead(val))
        {
            printf("    [0x%06X] %-35s = 0x%08X  ← LIVE\n",
                   cta_regs[i].addr, cta_regs[i].name, val);
        }
    }
}

// ============================================================================
// Широкое сканирование — ищем ВСЕ живые движки в GPU
// Составляем полную карту: что есть, что power-gated, что отсутствует
// ============================================================================

void full_engine_map(MMIO *m)
{
    printf("\n  === Full GPU Engine Map (0x000000 - 0x1000000) ===\n");
    printf("  Сканируем всё BAR0, классифицируем каждый 4KB блок\n\n");

    // Будем считать в бакетах по 64KB
    int total_buckets = m->size / (64 * 1024);

    typedef struct
    {
        uint32_t base;
        int live;    // реальные данные
        int timeout; // PRI_TIMEOUT (power-gated)
        int error;   // PRI_ERROR (denied)
        int zero;    // unmapped
    } Bucket;

    Bucket *buckets = calloc(total_buckets, sizeof(Bucket));

    for (int b = 0; b < total_buckets; b++)
    {
        buckets[b].base = b * 64 * 1024;

        // Sample 16 points per 64KB bucket
        for (int s = 0; s < 16; s++)
        {
            uint32_t off = buckets[b].base + s * 4096;
            if (off + 4 > m->size)
                break;
            uint32_t val = rd32(m, off);

            if (val == 0 || val == 0xFFFFFFFF)
                buckets[b].zero++;
            else if (is_pri_error(val))
            {
                if ((val & 0xF000) == 0x1000)
                    buckets[b].timeout++;
                else
                    buckets[b].error++;
            }
            else
                buckets[b].live++;
        }
    }

    // Print map
    printf("  Addr     Status    Description\n");
    printf("  ──────────────────────────────────────────────\n");

    // Known block names
    typedef struct
    {
        uint32_t start;
        uint32_t end;
        const char *name;
    } KnownBlock;
    KnownBlock known[] = {
        {0x000000, 0x001000, "PMC (Master Control)"},
        {0x001000, 0x002000, "PBUS"},
        {0x002000, 0x003000, "PFIFO_CONTROL"},
        {0x009000, 0x00A000, "PFIFO"},
        {0x020000, 0x021000, "PTIMER"},
        {0x021000, 0x022000, "FUSE"},
        {0x022000, 0x023000, "PTOP"},
        {0x060000, 0x061000, "PCOPY/CE"},
        {0x088000, 0x089000, "PNVLINK_SYS?"},
        {0x100000, 0x110000, "FBHUB/FBPA"},
        {0x104000, 0x108000, "CE (Copy Engines)"},
        {0x110000, 0x120000, "LTC"},
        {0x140000, 0x160000, "FB/MEM"},
        {0x170000, 0x180000, "PMU/FECS"},
        {0x400000, 0x500000, "GR (Graphics)"},
        {0x500000, 0x600000, "GPC (Graphics Processing Cluster)"},
        {0x700000, 0x800000, "PRAMIN window"},
        {0x800000, 0x900000, "PDISP (Display)"},
        {0x900000, 0xA00000, "PVIC/NVENC/NVDEC"},
        {0xA00000, 0xA20000, "IOCTRL (NVLink)"},
        {0xA20000, 0xA80000, "NVLIPT (NVLink)"},
        {0, 0, NULL}};

    for (int b = 0; b < total_buckets; b++)
    {
        if (buckets[b].live == 0 && buckets[b].timeout == 0 &&
            buckets[b].error == 0 && buckets[b].zero <= 16)
            continue;

        // Determine status
        const char *status;
        if (buckets[b].live > 0)
            status = "LIVE     ";
        else if (buckets[b].timeout > 0)
            status = "PWR-GATED";
        else if (buckets[b].error > 0)
            status = "DENIED   ";
        else
            status = "empty    ";

        // Skip boring empty buckets
        if (buckets[b].live == 0 && buckets[b].timeout == 0 && buckets[b].error == 0)
            continue;

        // Find name
        const char *name = "";
        for (int k = 0; known[k].name; k++)
        {
            if (buckets[b].base >= known[k].start && buckets[b].base < known[k].end)
            {
                name = known[k].name;
                break;
            }
        }

        printf("  0x%06X  %s  L=%2d T=%2d E=%2d  %s",
               buckets[b].base, status,
               buckets[b].live, buckets[b].timeout, buckets[b].error, name);

        // Flag unknown live regions — potential hidden engines!
        if (buckets[b].live > 0 && strlen(name) == 0)
        {
            printf("  <<< UNKNOWN LIVE ENGINE — INVESTIGATE!");
        }
        printf("\n");
    }

    free(buckets);
}

// ============================================================================
// Scan upper MMIO via PRAMIN for TMA-related blocks
// ============================================================================

void scan_upper_tma(MMIO *m)
{
    printf("\n  === Upper MMIO TMA Scan (via PRAMIN) ===\n\n");

    // На Hopper TMA-related блоки могут быть в диапазонах:
    // 0x01400000 - 0x01600000 (расширения после NVLink PHY)
    // 0x02000000+ (дополнительные IP блоки)

    typedef struct
    {
        uint32_t start;
        uint32_t end;
        const char *name;
    } ScanRange;
    ScanRange ranges[] = {
        {0x01400000, 0x01500000, "Post-NVLPHY region"},
        {0x01500000, 0x01600000, "Extended PHY/DMA?"},
        {0x01600000, 0x01700000, "Unknown 0x16x"},
        {0x01800000, 0x01900000, "Unknown 0x18x"},
        {0x02000000, 0x02100000, "Extended engine space 0x20x"},
        {0x02200000, 0x02300000, "Extended engine space 0x22x"},
        {0, 0, NULL}};

    for (int r = 0; ranges[r].name; r++)
    {
        int live = 0, timeout = 0, zeros = 0;
        uint32_t first_live_addr = 0, first_live_val = 0;

        // Sample every 4KB
        for (uint32_t addr = ranges[r].start; addr < ranges[r].end; addr += 4096)
        {
            uint32_t val = pramin_rd32(m, addr);
            if (val == 0 || val == 0xFFFFFFFF)
                zeros++;
            else if (is_pri_error(val))
            {
                if ((val & 0xF000) == 0x1000)
                    timeout++;
            }
            else
            {
                live++;
                if (!first_live_addr)
                {
                    first_live_addr = addr;
                    first_live_val = val;
                }
            }
        }

        printf("  [%s] 0x%07X-0x%07X: ", ranges[r].name, ranges[r].start, ranges[r].end);
        if (live > 0)
        {
            printf("LIVE=%d (first: 0x%07X=0x%08X)\n", live, first_live_addr, first_live_val);
            printf("    >>> ЖИВОЙ БЛОК В ВЕРХНЕМ MMIO — МОЖЕТ БЫТЬ СКРЫТЫЙ ENGINE! <<<\n");
        }
        else if (timeout > 0)
        {
            printf("PWR-GATED (timeout=%d)\n", timeout);
        }
        else
        {
            printf("empty (unmapped)\n");
        }
    }
}

// ============================================================================
// Check SM architectural capabilities — hidden instruction support
// ============================================================================

void check_sm_caps(MMIO *m)
{
    printf("\n  === SM Capability Registers ===\n");
    printf("  Ищем скрытые capabilities в SM (Streaming Multiprocessor)\n\n");

    // GR capability registers
    // NV_PGRAPH_PRI_GPC0_GPM_BASE = 0x00500000-ish
    // NV_PGRAPH_PRI_GPCS_TPCS_SM_ARCH = 0x00419A04 (from envytools)

    ProbePoint sm_caps[] = {
        // Global GR caps
        {0x00409800, "GR_PRI_FE_STATUS"},
        {0x00409804, "GR_PRI_FE_CAPS"},
        {0x00409A00, "GR_PRI_SKED_STATUS"},
        {0x00409A04, "GR_PRI_SKED_CAPS"},
        {0x00409C00, "GR_PRI_CWD_STATUS"},
        {0x00409C04, "GR_PRI_CWD_CAPS"},

        // SM architectural capabilities
        {0x00419A00, "GPCS_TPCS_SM_STATUS"},
        {0x00419A04, "GPCS_TPCS_SM_ARCH"}, // <<< KEY: SM architecture caps
        {0x00419A08, "GPCS_TPCS_SM_ARCH2"},
        {0x00419A0C, "GPCS_TPCS_SM_ARCH3"},
        {0x00419A10, "GPCS_TPCS_SM_CAPS"},
        {0x00419A14, "GPCS_TPCS_SM_CAPS2"},
        {0x00419A18, "GPCS_TPCS_SM_CAPS3"},
        {0x00419A1C, "GPCS_TPCS_SM_CAPS4"},
        {0x00419A20, "GPCS_TPCS_SM_EXTENDED1"},
        {0x00419A24, "GPCS_TPCS_SM_EXTENDED2"},
        {0x00419A28, "GPCS_TPCS_SM_DMA_CAPS?"},
        {0x00419A2C, "GPCS_TPCS_SM_TMA_CAPS?"},
        {0x00419A30, "GPCS_TPCS_SM_BULK_CAPS?"},

        // MPC (Multiprocessor Controller)
        {0x00419C00, "GPCS_TPCS_MPC_STATUS"},
        {0x00419C04, "GPCS_TPCS_MPC_CAPS"},

        // PPC (Programmable Pipeline Controller)
        {0x0041BE00, "GPCS_PPCS_STATUS"},
        {0x0041BE04, "GPCS_PPCS_CAPS"},

        // ZCULL
        {0x00418900, "GPCS_ZCULL_STATUS"},
        {0x00418904, "GPCS_ZCULL_CAPS"},

        // ROP
        {0x00410000, "GR_PRI_ROP_STATUS"},
        {0x00410004, "GR_PRI_ROP_CAPS"},

        {0, NULL}};

    for (int i = 0; sm_caps[i].name; i++)
    {
        uint32_t val = gpu_rd32(m, sm_caps[i].addr);
        if (!is_dead(val))
        {
            printf("    [0x%06X] %-35s = 0x%08X", sm_caps[i].addr, sm_caps[i].name, val);

            // Особый интерес: SM_ARCH содержит ISA version и capability bits
            if (sm_caps[i].addr == 0x00419A04)
            {
                // SM_ARCH обычно содержит: [31:24]=major [23:16]=minor [15:0]=caps
                uint32_t major = (val >> 24) & 0xFF;
                uint32_t minor = (val >> 16) & 0xFF;
                printf("\n           SM_ARCH: major=%u minor=%u (sm_%u%u)",
                       major, minor, major, minor);
                if (major == 9 && minor == 0)
                    printf(" → HOPPER (sm_90) !!!");
                if (major == 8 && minor == 9)
                    printf(" → ADA (sm_89)");
                printf("\n           Cap bits: 0x%04X", val & 0xFFFF);
                // Decode individual capability bits
                printf("\n           ");
                for (int b = 15; b >= 0; b--)
                {
                    printf("%d", (val >> b) & 1);
                    if (b % 4 == 0)
                        printf(" ");
                }
            }
            printf("\n");
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(void)
{
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  TMA (Tensor Memory Accelerator) Probe — RTX 4090  ║\n");
    printf("║  Ищем скрытый DMA-движок в AD102 silicon           ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    if (geteuid() != 0)
    {
        printf("[!] sudo ./tma_probe\n\n");
    }

    GPU gpu = {0};
    if (!find_gpu(&gpu))
    {
        printf("GPU не найден\n");
        return 1;
    }
    printf("GPU: %s (0x%04X) BAR0=%luMB\n\n", gpu.bdf, gpu.device_id, gpu.bar0_size / (1024 * 1024));

    MMIO *m = map_bar0(&gpu);
    if (!m)
        return 1;

    // Chip ID
    uint32_t boot0 = rd32(m, 0);
    printf("BOOT_0 = 0x%08X → AD102 (sm_89)\n\n", boot0);

    // 1. Full discovery table decode
    decode_discovery_table(m);

    // 2. SM capability registers
    check_sm_caps(m);

    // 3. TMA-specific MMIO
    scan_tma_mmio(m);

    // 4. Full engine map
    full_engine_map(m);

    // 5. Upper MMIO scan
    scan_upper_tma(m);

    // Summary
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    TMA ANALYSIS                            ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║                                                            ║\n");
    printf("║  TMA на Hopper реализован как:                             ║\n");
    printf("║  1. SM instruction decoder (cp.async.bulk.tensor PTX)      ║\n");
    printf("║  2. CTA-level DMA unit (аппаратный prefetch)               ║\n");
    printf("║  3. Shared Memory controller extension                     ║\n");
    printf("║                                                            ║\n");
    printf("║  На Ada (sm_89) cp.async (без .bulk.tensor) ЕСТЬ —         ║\n");
    printf("║  это упрощённый async copy без тензорных дескрипторов.     ║\n");
    printf("║                                                            ║\n");
    printf("║  Что искать в результатах:                                 ║\n");
    printf("║  • Discovery table type 0x1F/0x30+ → отдельный TMA IP     ║\n");
    printf("║  • SM_ARCH capability bits → скрытая ISA extension         ║\n");
    printf("║  • CE extended registers → TMA как расширение Copy Engine  ║\n");
    printf("║  • Unknown LIVE regions → незадокументированные engines    ║\n");
    printf("║  • Upper MMIO live blocks → скрытые accelerators           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    munmap(m->base, m->size);
    free(m);
    return 0;
}