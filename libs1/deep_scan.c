/*
 * deep_scan.c - Детальное сканирование неизвестных MMIO блоков RTX 4090
 *
 * Фокус: 0x300000-0x3F0000 (1MB полностью живой, неизвестный engine)
 * Плюс все остальные unknown live regions
 *
 * Методы идентификации:
 *   1. Полный дамп каждого регистра — ищем сигнатуры (version, ID, caps)
 *   2. Анализ паттернов — повторяющиеся структуры = массив instances
 *   3. Сравнение с известными engine fingerprints
 *   4. Read-write тест — какие регистры R/W vs R/O
 *   5. Cross-reference с open-gpu-kernel register headers
 *
 * gcc -O2 -o deep_scan deep_scan.c
 * sudo ./deep_scan 2>&1 | tee deep_scan_output.txt
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
#include <math.h>

// ============================================================================
// MMIO infrastructure (reused from probes)
// ============================================================================

static bool is_pri_error(uint32_t val)
{
    return (val & 0xFFFF0000) == 0xBADF0000;
}

static bool is_dead(uint32_t val)
{
    return val == 0 || val == 0xFFFFFFFF || val == 0xDEADDEAD || is_pri_error(val);
}

typedef struct
{
    void *base;
    size_t size;
    bool writable;
} MMIO;

static inline uint32_t rd32(MMIO *m, uint32_t off)
{
    if (off + 4 > m->size)
        return 0xDEADDEAD;
    return *(volatile uint32_t *)((uint8_t *)m->base + off);
}

static inline void wr32(MMIO *m, uint32_t off, uint32_t val)
{
    if (!m->writable || off + 4 > m->size)
        return;
    *(volatile uint32_t *)((uint8_t *)m->base + off) = val;
}

MMIO *open_bar0(void)
{
    glob_t g;
    if (glob("/sys/bus/pci/devices/*/vendor", 0, NULL, &g) != 0)
        return NULL;

    for (size_t i = 0; i < g.gl_pathc; i++)
    {
        char buf[64];
        FILE *f = fopen(g.gl_pathv[i], "r");
        if (!f)
            continue;
        int is_nvidia = 0;
        if (fgets(buf, sizeof(buf), f) && strtoul(buf, NULL, 0) == 0x10de)
            is_nvidia = 1;
        fclose(f);
        if (!is_nvidia)
            continue;

        char *dir = strdup(g.gl_pathv[i]);
        *strrchr(dir, '/') = 0;

        // Check class
        char path[600];
        snprintf(path, sizeof(path), "%s/class", dir);
        f = fopen(path, "r");
        uint32_t cls = 0;
        if (f)
        {
            if (fgets(buf, sizeof(buf), f))
                cls = strtoul(buf, NULL, 0);
            fclose(f);
        }
        if ((cls >> 16) != 0x03)
        {
            free(dir);
            continue;
        }

        // Get BAR0 size
        snprintf(path, sizeof(path), "%s/resource", dir);
        f = fopen(path, "r");
        uint64_t bar0_start = 0, bar0_end = 0;
        if (f)
        {
            if (fgets(buf, sizeof(buf), f))
                sscanf(buf, "%lx %lx", &bar0_start, &bar0_end);
            fclose(f);
        }
        size_t sz = bar0_end - bar0_start + 1;
        if (sz == 0 || sz > 256 * 1024 * 1024)
            sz = 16 * 1024 * 1024;

        // Map BAR0
        snprintf(path, sizeof(path), "%s/resource0", dir);
        int fd = open(path, O_RDWR | O_SYNC);
        bool writable = true;
        if (fd < 0)
        {
            fd = open(path, O_RDONLY | O_SYNC);
            writable = false;
        }
        if (fd < 0)
        {
            free(dir);
            continue;
        }

        int prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
        void *p = mmap(NULL, sz, prot, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED && writable)
        {
            p = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
            writable = false;
        }
        close(fd);
        free(dir);

        if (p == MAP_FAILED)
            continue;

        MMIO *m = malloc(sizeof(MMIO));
        m->base = p;
        m->size = sz;
        m->writable = writable;
        globfree(&g);
        return m;
    }
    globfree(&g);
    return NULL;
}

// ============================================================================
// Register classification helpers
// ============================================================================

typedef struct
{
    uint32_t offset;
    uint32_t value;
} RegEntry;

// Dump all live registers in a range, return count
int dump_live_regs(MMIO *m, uint32_t start, uint32_t end, RegEntry *out, int max_out)
{
    int count = 0;
    for (uint32_t off = start; off < end && off < m->size && count < max_out; off += 4)
    {
        uint32_t val = rd32(m, off);
        if (!is_dead(val))
        {
            out[count].offset = off;
            out[count].value = val;
            count++;
        }
    }
    return count;
}

// Check if a register is read-only by trying to flip bits
// Returns: 0=read-only, 1=read-write, -1=error
int test_rw(MMIO *m, uint32_t off)
{
    if (!m->writable)
        return -1;
    uint32_t orig = rd32(m, off);
    if (is_dead(orig))
        return -1;

    // Try flipping low bit
    wr32(m, off, orig ^ 1);
    uint32_t after = rd32(m, off);
    wr32(m, off, orig); // restore

    return (after != orig) ? 1 : 0;
}

// Detect repeating structure (instance array)
// Returns stride if pattern found, 0 otherwise
uint32_t detect_stride(MMIO *m, uint32_t start, uint32_t end)
{
    // Try common strides: 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000
    uint32_t strides[] = {0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000, 0};

    for (int s = 0; strides[s]; s++)
    {
        uint32_t stride = strides[s];
        if (start + stride * 3 > end)
            continue;

        // Compare first register at each stride offset
        int matches = 0;
        int total = 0;
        uint32_t ref = rd32(m, start);

        for (uint32_t off = start + stride; off < end && off < start + stride * 16; off += stride)
        {
            uint32_t val = rd32(m, off);
            total++;
            // Both dead or both similar magnitude
            if ((is_dead(ref) && is_dead(val)) ||
                (!is_dead(ref) && !is_dead(val) &&
                 ((ref ^ val) < 0x1000 || (ref & 0xFFFF0000) == (val & 0xFFFF0000))))
            {
                matches++;
            }
        }

        if (total >= 2 && matches >= total * 2 / 3)
        {
            return stride;
        }
    }
    return 0;
}

// ============================================================================
// Known engine signatures
// ============================================================================

typedef struct
{
    const char *name;
    uint32_t signature_offset; // offset from block base
    uint32_t signature_mask;
    uint32_t signature_value;
    const char *description;
} EngineSignature;

// Known first-register signatures from open-gpu-kernel / nouveau
static const EngineSignature SIGS[] = {
    // Falcon-based engines have 0x030-0x03C = FALCON_HWCFG
    {"FALCON", 0x000, 0x00000000, 0x00000000, "Falcon microcontroller (check +0x30 for HWCFG)"},
    // LTC has specific pattern at base
    {"LTC", 0x000, 0xFFFF0000, 0x00140000, "L2 Cache controller"},
    // FBPA
    {"FBPA", 0x000, 0xFFFF0000, 0x00000000, "FB Partition Agent"},
    // GPC/TPC/SM have version registers
    {"GPC", 0x000, 0xFF000000, 0x87000000, "Graphics Processing Cluster"},
    // XBAR
    {"XBAR", 0x000, 0xFFFF0000, 0x00040000, "Crossbar"},
    // HSHUB
    {"HSHUB", 0x000, 0xFFFF0000, 0x00050000, "High-Speed Hub"},
    {NULL, 0, 0, 0, NULL}};

// ============================================================================
// DEEP SCAN: 0x300000 - 0x3F0000
// ============================================================================

void deep_scan_region(MMIO *m, uint32_t start, uint32_t end, const char *label)
{
    printf("\n╔═══════════════════════════════════════════════════════╗\n");
    printf("║  Deep Scan: %s (0x%06X - 0x%06X)  \n", label, start, end);
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    if (end > m->size)
    {
        end = m->size;
    }
    if (start >= m->size)
    {
        printf("  Out of BAR0 range\n");
        return;
    }

    // Phase 1: Statistics
    int live = 0, dead_zero = 0, dead_pri = 0, dead_other = 0;
    int total = (end - start) / 4;

    // Value histogram — какие значения встречаются чаще всего
    typedef struct
    {
        uint32_t val;
        int count;
    } ValCount;
    ValCount top_vals[64] = {0};
    int n_unique = 0;

    for (uint32_t off = start; off < end; off += 4)
    {
        uint32_t val = rd32(m, off);
        if (val == 0)
            dead_zero++;
        else if (is_pri_error(val))
            dead_pri++;
        else if (val == 0xFFFFFFFF || val == 0xDEADDEAD)
            dead_other++;
        else
        {
            live++;
            // Track top values
            bool found = false;
            for (int i = 0; i < n_unique; i++)
            {
                if (top_vals[i].val == val)
                {
                    top_vals[i].count++;
                    found = true;
                    break;
                }
            }
            if (!found && n_unique < 64)
            {
                top_vals[n_unique].val = val;
                top_vals[n_unique].count = 1;
                n_unique++;
            }
        }
    }

    printf("  Total registers: %d\n", total);
    printf("  Live:     %d (%.1f%%)\n", live, 100.0 * live / total);
    printf("  Zero:     %d\n", dead_zero);
    printf("  PRI err:  %d\n", dead_pri);
    printf("  Other:    %d\n\n", dead_other);

    // Sort top values by count
    for (int i = 0; i < n_unique - 1; i++)
    {
        for (int j = i + 1; j < n_unique; j++)
        {
            if (top_vals[j].count > top_vals[i].count)
            {
                ValCount tmp = top_vals[i];
                top_vals[i] = top_vals[j];
                top_vals[j] = tmp;
            }
        }
    }

    printf("  Top repeated values (potential config/status patterns):\n");
    for (int i = 0; i < 20 && i < n_unique; i++)
    {
        if (top_vals[i].count < 2)
            break;
        printf("    0x%08X  ×%d\n", top_vals[i].val, top_vals[i].count);
    }

    // Phase 2: Structure detection
    printf("\n  --- Structure Detection ---\n");
    uint32_t stride = detect_stride(m, start, end);
    if (stride)
    {
        int instances = (end - start) / stride;
        printf("  Repeating structure: stride=0x%X (%d bytes), ~%d instances\n",
               stride, stride, instances);

        // Показать первые слова каждой instance
        printf("  Instance headers (first 4 words each):\n");
        for (int inst = 0; inst < instances && inst < 24; inst++)
        {
            uint32_t base = start + inst * stride;
            uint32_t w0 = rd32(m, base);
            uint32_t w1 = rd32(m, base + 4);
            uint32_t w2 = rd32(m, base + 8);
            uint32_t w3 = rd32(m, base + 12);

            if (is_dead(w0) && is_dead(w1) && is_dead(w2) && is_dead(w3))
            {
                printf("    [%2d] @ 0x%06X: (all dead)\n", inst, base);
            }
            else
            {
                printf("    [%2d] @ 0x%06X: %08X %08X %08X %08X",
                       inst, base, w0, w1, w2, w3);
                if (!is_dead(w0))
                    printf(" ←");
                printf("\n");
            }
        }
    }
    else
    {
        printf("  No repeating structure detected (irregular layout)\n");
    }

    // Phase 3: Sub-block analysis - scan in 4KB chunks
    printf("\n  --- Sub-block Map (4KB granularity) ---\n");
    for (uint32_t blk = start; blk < end; blk += 0x1000)
    {
        int blk_live = 0, blk_total = 0;
        uint32_t first_val = 0;
        uint32_t first_off = 0;

        for (uint32_t off = blk; off < blk + 0x1000 && off < end; off += 4)
        {
            uint32_t val = rd32(m, off);
            blk_total++;
            if (!is_dead(val))
            {
                blk_live++;
                if (!first_off)
                {
                    first_off = off;
                    first_val = val;
                }
            }
        }

        if (blk_live > 0)
        {
            float density = 100.0f * blk_live / blk_total;
            printf("  0x%06X: %3d/%3d live (%5.1f%%)", blk, blk_live, blk_total, density);
            printf("  first=[0x%06X]=0x%08X\n", first_off, first_val);
        }
    }

    // Phase 4: Full register dump (first 2KB)
    printf("\n  --- Register Dump (first 512 bytes) ---\n");
    for (uint32_t off = start; off < start + 512 && off < end; off += 4)
    {
        uint32_t val = rd32(m, off);
        if (!is_dead(val))
        {
            printf("  [0x%06X] = 0x%08X", off, val);

            // R/W test
            if (m->writable)
            {
                int rw = test_rw(m, off);
                printf("  %s", rw == 1 ? "R/W" : rw == 0 ? "R/O"
                                                         : "???");
            }
            printf("\n");
        }
    }

    // Phase 5: Signature matching
    printf("\n  --- Signature Check ---\n");
    uint32_t base_val = rd32(m, start);
    printf("  Base register [0x%06X] = 0x%08X\n", start, base_val);

    // Check for Falcon microcontroller signature
    // Falcon has: +0x000 = FALCON_IRQSSET, +0x030 = FALCON_HWCFG, +0x100 = FALCON_CPUCTL
    uint32_t falcon_hwcfg = rd32(m, start + 0x030);
    uint32_t falcon_cpuctl = rd32(m, start + 0x100);
    uint32_t falcon_bootvec = rd32(m, start + 0x104);
    uint32_t falcon_dmactl = rd32(m, start + 0x10C);

    if (!is_dead(falcon_hwcfg) && !is_dead(falcon_cpuctl))
    {
        printf("  >>> Possible FALCON engine!\n");
        printf("      HWCFG    [+0x030] = 0x%08X\n", falcon_hwcfg);
        printf("      CPUCTL   [+0x100] = 0x%08X\n", falcon_cpuctl);
        printf("      BOOTVEC  [+0x104] = 0x%08X\n", falcon_bootvec);
        printf("      DMACTL   [+0x10C] = 0x%08X\n", falcon_dmactl);
    }

    // Check for version/ID register pattern (common in NVIDIA engines)
    // Many engines have a version at +0x000 or +0x004
    for (int voff = 0; voff <= 0x10; voff += 4)
    {
        uint32_t v = rd32(m, start + voff);
        if (is_dead(v))
            continue;
        uint32_t major = (v >> 24) & 0xFF;
        uint32_t minor = (v >> 16) & 0xFF;
        if (major > 0 && major < 10 && minor < 20)
        {
            printf("  Possible version at +0x%03X: v%d.%d (0x%08X)\n", voff, major, minor, v);
        }
    }
}

// ============================================================================
// CE (Copy Engine) deep analysis — TMA candidate
// ============================================================================

void deep_scan_ce(MMIO *m)
{
    printf("\n╔═══════════════════════════════════════════════════════╗\n");
    printf("║  Copy Engine Deep Analysis — TMA Candidate           ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    // CE registers typically at 0x104000+
    // From discovery table we have 4 CE instances
    // CE base addresses from discovery: need to decode from table

    // Dump CE0 region thoroughly
    uint32_t ce_bases[] = {0x104000, 0x105000, 0x106000, 0x107000};

    for (int ce = 0; ce < 4; ce++)
    {
        uint32_t base = ce_bases[ce];
        printf("  CE%d @ 0x%06X:\n", ce, base);

        int live_count = 0;
        for (uint32_t off = base; off < base + 0x1000 && off < m->size; off += 4)
        {
            uint32_t val = rd32(m, off);
            if (!is_dead(val))
            {
                printf("    [+0x%03X] = 0x%08X", off - base, val);
                live_count++;

                // Annotate known CE register offsets
                uint32_t reg_off = off - base;
                if (reg_off == 0x000)
                    printf("  (CE_STATUS?)");
                if (reg_off == 0x004)
                    printf("  (CE_CAPS?)");
                if (reg_off == 0x010)
                    printf("  (CE_CAPS2?)");
                if (reg_off == 0x100)
                    printf("  (CE_EXTENDED?)");
                if (reg_off == 0x104)
                    printf("  (CE_EXTENDED2?)");
                if (reg_off == 0x200)
                    printf("  (CE_CONFIG?)");
                if (reg_off == 0x204)
                    printf("  (CE_CONFIG2?)");

                // Capability decode attempt
                if (reg_off == 0x100 && val != 0)
                {
                    printf("\n           Capability bits: ");
                    for (int b = 31; b >= 0; b--)
                    {
                        printf("%d", (val >> b) & 1);
                        if (b % 4 == 0)
                            printf(" ");
                    }
                }
                if (reg_off == 0x200 && val != 0)
                {
                    printf("\n           Config bits: ");
                    for (int b = 31; b >= 0; b--)
                    {
                        printf("%d", (val >> b) & 1);
                        if (b % 4 == 0)
                            printf(" ");
                    }
                }
                printf("\n");
            }
        }
        printf("    Live registers: %d\n\n", live_count);
    }
}

// ============================================================================
// HSHUB / XBAR / PRI_MASTER — internal interconnect (may be DMA path)
// ============================================================================

void scan_interconnect(MMIO *m)
{
    printf("\n╔═══════════════════════════════════════════════════════╗\n");
    printf("║  Internal Interconnect & Hub Scan                     ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    // HSHUB — High Speed Hub (data path between engines)
    // On Ada typically 0x1F0000-ish or part of unknown regions
    // XBAR — Crossbar (GPC<->LTC<->DRAM routing)
    // PRI Master Hub — routes all MMIO accesses

    typedef struct
    {
        uint32_t start;
        uint32_t end;
        const char *name;
    } Region;
    Region regions[] = {
        {0x120000, 0x140000, "Post-LTC / XBAR?"},
        {0x180000, 0x1C0000, "Post-PMU region"},
        {0x1C0000, 0x200000, "Unknown engines"},
        {0x200000, 0x240000, "Extended config?"},
        {0x240000, 0x300000, "Unknown engines 2"},
        {0xB60000, 0xC00000, "Post-NVLink unknown"},
        {0xC00000, 0xC40000, "High MMIO unknown"},
        {0, 0, NULL}};

    for (int r = 0; regions[r].name; r++)
    {
        uint32_t start = regions[r].start;
        uint32_t end = regions[r].end;
        if (start >= m->size)
            continue;
        if (end > m->size)
            end = m->size;

        int live = 0, timeout = 0, error = 0;
        for (uint32_t off = start; off < end; off += 64)
        {
            uint32_t val = rd32(m, off);
            if (!is_dead(val))
                live++;
            else if (is_pri_error(val))
            {
                if ((val & 0xF000) == 0x1000)
                    timeout++;
                else
                    error++;
            }
        }

        int total = (end - start) / 64;
        if (live > 0 || timeout > 0)
        {
            printf("  [%s] 0x%06X-0x%06X\n", regions[r].name, start, end);
            printf("    live=%d timeout=%d error=%d (of %d samples)\n\n",
                   live, timeout, error, total);

            // Dump first few live regs
            if (live > 0)
            {
                int shown = 0;
                for (uint32_t off = start; off < end && shown < 16; off += 4)
                {
                    uint32_t val = rd32(m, off);
                    if (!is_dead(val))
                    {
                        printf("    [0x%06X] = 0x%08X\n", off, val);
                        shown++;
                    }
                }
                printf("\n");
            }
        }
    }
}

// ============================================================================
// Falcon engine detector — scan for microcontroller patterns
// Many engines on NVIDIA GPUs are based on Falcon (SEC2, GSP, PMU, etc.)
// ============================================================================

void find_falcons(MMIO *m)
{
    printf("\n╔═══════════════════════════════════════════════════════╗\n");
    printf("║  Falcon Microcontroller Detector                      ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    // Falcon engines have recognizable register layout:
    // +0x030 = FALCON_HWCFG (contains IMEM/DMEM sizes)
    // +0x034 = FALCON_HWCFG1
    // +0x100 = FALCON_CPUCTL
    // +0x104 = FALCON_BOOTVEC
    // +0x108 = FALCON_HWCFG2
    // +0x10C = FALCON_DMACTL
    // +0x110 = FALCON_DMATRFBASE
    // +0x118 = FALCON_DMATRFCMD

    // Scan all live regions for Falcon fingerprint
    for (uint32_t base = 0; base + 0x200 < m->size; base += 0x1000)
    {
        uint32_t hwcfg = rd32(m, base + 0x030);
        uint32_t cpuctl = rd32(m, base + 0x100);
        uint32_t bootvec = rd32(m, base + 0x104);

        if (is_dead(hwcfg) || is_dead(cpuctl))
            continue;

        // Falcon HWCFG contains IMEM/DMEM size in specific fields
        // Typical: bits[8:0] = IMEM size / 256, bits[17:9] = DMEM size / 256
        uint32_t imem_sz = (hwcfg & 0x1FF) * 256;
        uint32_t dmem_sz = ((hwcfg >> 9) & 0x1FF) * 256;

        // Sanity check: IMEM/DMEM should be 4KB-1MB range
        if (imem_sz >= 1024 && imem_sz <= 1048576 &&
            dmem_sz >= 1024 && dmem_sz <= 1048576)
        {
            printf("  FALCON @ 0x%06X\n", base);
            printf("    HWCFG   = 0x%08X  (IMEM=%uKB, DMEM=%uKB)\n",
                   hwcfg, imem_sz / 1024, dmem_sz / 1024);
            printf("    CPUCTL  = 0x%08X\n", cpuctl);
            printf("    BOOTVEC = 0x%08X\n", bootvec);

            uint32_t hwcfg2 = rd32(m, base + 0x108);
            uint32_t dmactl = rd32(m, base + 0x10C);
            if (!is_dead(hwcfg2))
                printf("    HWCFG2  = 0x%08X\n", hwcfg2);
            if (!is_dead(dmactl))
                printf("    DMACTL  = 0x%08X\n", dmactl);

            // Check what engine this might be based on address
            printf("    Possible ID: ");
            if (base >= 0x840000 && base < 0x850000)
                printf("NVDEC\n");
            else if (base >= 0x980000 && base < 0x990000)
                printf("NVENC\n");
            else if (base >= 0x170000 && base < 0x180000)
                printf("PMU\n");
            else if (base >= 0x300000 && base < 0x400000)
                printf("*** UNKNOWN FALCON in mystery region! ***\n");
            else if (base >= 0xB60000 && base < 0xC00000)
                printf("*** UNKNOWN FALCON in post-NVLink! ***\n");
            else
                printf("Unknown (addr=0x%06X)\n", base);
            printf("\n");
        }
    }
}

// ============================================================================
// NV_PGRAPH internal dump — look for TMA-like units inside GR
// ============================================================================

void scan_gr_internals(MMIO *m)
{
    printf("\n╔═══════════════════════════════════════════════════════╗\n");
    printf("║  GR (Graphics Engine) Internal Block Scan             ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    printf("  Looking for DMA/TMA-like sub-blocks inside GR engine\n\n");

    // GR engine is 0x400000-0x600000
    // Internal structure:
    //   0x400000 = GR global
    //   0x405000-0x406000 = PPC
    //   0x408000-0x410000 = GR_PRI registers
    //   0x418000-0x41C000 = GPCS broadcast registers
    //   0x419000-0x41A000 = GPCS_TPCS
    //   0x500000+ = Per-GPC registers

    // Scan GR_PRI area for undocumented sub-blocks
    printf("  GR_PRI region (0x400000-0x420000):\n");
    for (uint32_t blk = 0x400000; blk < 0x420000 && blk < m->size; blk += 0x1000)
    {
        int live = 0;
        for (uint32_t off = blk; off < blk + 0x1000; off += 64)
        {
            if (!is_dead(rd32(m, off)))
                live++;
        }
        if (live > 0)
        {
            printf("    0x%06X: %d live samples", blk, live);

            // Annotate known sub-blocks
            if (blk == 0x400000)
                printf("  (GR_GLOBAL)");
            else if (blk == 0x404000)
                printf("  (GR_FECS)");
            else if (blk == 0x405000)
                printf("  (GR_PPC)");
            else if (blk == 0x408000)
                printf("  (GR_PRI_BE)");
            else if (blk == 0x409000)
                printf("  (GR_PRI_FE/SKED/CWD)");
            else if (blk == 0x40A000)
                printf("  (GR_PRI_DS)");
            else if (blk == 0x40B000)
                printf("  (GR_PRI_SSYNC)");
            else if (blk == 0x418000)
                printf("  (GPCS)");
            else if (blk == 0x419000)
                printf("  (GPCS_TPCS — SM caps here)");
            else if (blk == 0x41A000)
                printf("  (GPCS_TPCS_SM)");
            else if (blk == 0x41B000)
                printf("  (GPCS_PPCS?)");
            else
                printf("  *** UNKNOWN GR SUB-BLOCK ***");

            printf("\n");
        }
    }

    // SM extended capabilities — detailed dump
    printf("\n  SM Extended Capability Registers (0x419A00-0x419B00):\n");
    for (uint32_t off = 0x419A00; off < 0x419B00 && off < m->size; off += 4)
    {
        uint32_t val = rd32(m, off);
        if (!is_dead(val))
        {
            printf("    [0x%06X] (+0x%02X from SM_BASE) = 0x%08X  bits: ",
                   off, off - 0x419A00, val);
            int set_bits = 0;
            for (int b = 31; b >= 0; b--)
            {
                if ((val >> b) & 1)
                    set_bits++;
                printf("%d", (val >> b) & 1);
                if (b % 8 == 0)
                    printf(" ");
            }
            printf(" (%d bits set)", set_bits);
            printf("\n");
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Deep MMIO Scanner — Unknown Engine Identification          ║\n");
    printf("║  RTX 4090 AD102                                            ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    if (geteuid() != 0)
    {
        printf("[!] sudo ./deep_scan\n");
        return 1;
    }

    MMIO *m = open_bar0();
    if (!m)
    {
        printf("Cannot map BAR0\n");
        return 1;
    }
    printf("BAR0: %zuMB, %s\n\n", m->size / (1024 * 1024), m->writable ? "R/W" : "R/O");

    // === PRIMARY TARGET: 0x300000-0x3F0000 (1MB unknown LIVE) ===
    deep_scan_region(m, 0x300000, 0x400000, "MAIN UNKNOWN (1MB live)");

    // === Copy Engine deep analysis ===
    deep_scan_ce(m);

    // === Falcon detector ===
    find_falcons(m);

    // === Other unknown live regions ===
    deep_scan_region(m, 0x040000, 0x060000, "Unknown 0x04x-0x05x");
    deep_scan_region(m, 0x080000, 0x0A0000, "Unknown 0x08x-0x09x");
    deep_scan_region(m, 0x180000, 0x220000, "Unknown 0x18x-0x21x");
    deep_scan_region(m, 0x610000, 0x700000, "Unknown 0x61x-0x6Fx");
    deep_scan_region(m, 0xB60000, 0xC00000, "Unknown 0xB6x-0xBFx");

    // === GR internals ===
    scan_gr_internals(m);

    // === Internal interconnect ===
    scan_interconnect(m);

    printf("\n═══ INTERPRETATION GUIDE ═══\n\n");
    printf("Repeating structure with stride → array of identical units (SMs, LTC slices, etc.)\n");
    printf("Falcon pattern (+0x030/+0x100) → microcontroller-based engine\n");
    printf("Dense live registers → active complex engine (not just config)\n");
    printf("R/W registers → control/config (vs R/O status/caps)\n");
    printf("High bit count in cap regs → many features enabled\n\n");
    printf("Key question: is 0x300000-0x3F0000 a known engine (XBAR/HSHUB/PRI)\n");
    printf("or something undocumented (DMA accelerator / TMA / other)?\n");

    munmap(m->base, m->size);
    free(m);
    return 0;
}