// 060: доказательство инъективности маркера-кандидата ДО использования.
// Формула агента: byte = ((row & 0x3F) << 2) | ((col_byte & 0x3F) >> 4)
// Домен: row 0..63 (6 бит), col_byte 0..127 (7 бит) = 8192 unique locations
// Разрядность byte: 8 бит = 256 values

#include <cstdio>
#include <cstdint>
#include <map>
#include <vector>

int main() {
    // 1. Проверка формулы-кандидата агента
    printf("=== Кандидат 058: byte = ((row & 0x3F) << 2) | ((col_byte & 0x3F) >> 4) ===\n");
    std::map<uint8_t, std::vector<std::pair<int,int>>> mp;
    for (int row = 0; row < 64; ++row) {
        for (int col = 0; col < 128; ++col) {
            uint8_t marker = ((row & 0x3F) << 2) | ((col & 0x3F) >> 4);
            mp[marker].push_back({row, col});
        }
    }
    int unique = 0, collided_locs = 0;
    int worst_bucket = 0;
    for (auto& [m, locs] : mp) {
        if (locs.size() == 1) unique++;
        else {
            collided_locs += locs.size();
            if ((int)locs.size() > worst_bucket) worst_bucket = locs.size();
        }
    }
    printf("Всего маркеров использовано: %zu из 256\n", mp.size());
    printf("Уникальных samples: %d, коллидированных: %d, worst bucket: %d\n",
           unique, collided_locs, worst_bucket);
    printf("АЛИАСИНГ: %s\n", collided_locs > 0 ? "ЕСТЬ (маркер НЕ инъективен)" : "нет");
    // Показать пример коллизий:
    printf("Примеры коллизий:\n");
    int examples = 0;
    for (auto& [m, locs] : mp) {
        if (locs.size() >= 2 && examples < 5) {
            printf("  marker=0x%02x collides at:", m);
            for (auto& [r, c] : locs) printf(" (r=%d,c=%d)", r, c);
            printf("\n");
            examples++;
        }
    }

    // 2. НОВЫЙ маркер (2-byte pair): byte@even_col = row, byte@odd_col = col
    printf("\n=== 060 NEW маркер (2-byte pair):\n");
    printf("       byte@(row, col) = row if col%%2==0 else col & 0xFF\n");
    std::map<std::pair<uint8_t,uint8_t>, std::vector<std::pair<int,int>>> mp2;
    for (int row = 0; row < 64; ++row) {
        for (int col = 0; col < 128; col += 2) {
            uint8_t m_even = (uint8_t)row;
            uint8_t m_odd  = (uint8_t)col; // even col carries row, next byte carries col
            mp2[{m_even, m_odd}].push_back({row, col});
        }
    }
    int un2 = 0, coll2 = 0;
    for (auto& [m, locs] : mp2) {
        if (locs.size() == 1) un2++;
        else coll2 += locs.size();
    }
    printf("Всего pair-маркеров: %zu из %d возможных (256*256=65536)\n", mp2.size(), 65536);
    printf("Уникальных pair samples: %d, коллидированных: %d\n", un2, coll2);
    printf("Домен coverage: %d rows × 64 pair-positions = %d samples\n", 64, 64*64);
    printf("АЛИАСИНГ пары: %s\n", coll2 > 0 ? "ЕСТЬ" : "нет — pair injective ✓");

    // 3. Простой row-only marker (для basic LDSM layout probe)
    printf("\n=== ПРОСТОЙ row-only marker (для basic probe):\n");
    printf("       byte@(row, col) = row (all col_bytes of row hold row-id)\n");
    printf("Injective по row (64 unique row-values); col_byte position восстанавливается\n");
    printf("через известное mapping LDSM lane → tile_row при row_ptr = swz_byte(...)\n");
    printf("Позволяет проверить: layer LDSM корректный если ВСЕ 32 lanes видят ТОЛЬКО те row-байты,\n");
    printf("которые ожидаются по формуле row_ptr 049-B. Coverage: (lane, kb, np, lo/hi, reg, byte)=32768.\n");

    return 0;
}
