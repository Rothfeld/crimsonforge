/* Standalone PaChecksum DLL — no Python dependency.
 * Loaded via ctypes for ~100x speedup over pure Python.
 *
 * Build: gcc -shared -O3 -o pa_checksum.dll _pa_checksum_dll.c
 */

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static inline uint32_t rot(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

EXPORT uint32_t pa_checksum(const uint8_t *data, size_t len) {
    uint32_t a, b, c;
    a = b = c = (uint32_t)len - 0x2145E233u;

    while (len > 12) {
        a += *(const uint32_t*)(data);
        b += *(const uint32_t*)(data + 4);
        c += *(const uint32_t*)(data + 8);

        a -= c; a ^= rot(c, 4);  c += b;
        b -= a; b ^= rot(a, 6);  a += c;
        c -= b; c ^= rot(b, 8);  b += a;
        a -= c; a ^= rot(c, 16); c += b;
        b -= a; b ^= rot(a, 19); a += c;
        c -= b; c ^= rot(b, 4);  b += a;

        data += 12;
        len -= 12;
    }

    switch (len) {
        case 12: c += (uint32_t)data[11] << 24; /* fall through */
        case 11: c += (uint32_t)data[10] << 16; /* fall through */
        case 10: c += (uint32_t)data[9] << 8;   /* fall through */
        case 9:  c += data[8];                   /* fall through */
        case 8:  b += (uint32_t)data[7] << 24;  /* fall through */
        case 7:  b += (uint32_t)data[6] << 16;  /* fall through */
        case 6:  b += (uint32_t)data[5] << 8;   /* fall through */
        case 5:  b += data[4];                   /* fall through */
        case 4:  a += (uint32_t)data[3] << 24;  /* fall through */
        case 3:  a += (uint32_t)data[2] << 16;  /* fall through */
        case 2:  a += (uint32_t)data[1] << 8;   /* fall through */
        case 1:  a += data[0]; break;
        case 0:  return c;
    }

    c ^= b; c -= rot(b, 14);
    a ^= c; a -= rot(c, 11);
    b ^= a; b -= rot(a, 25);
    c ^= b; c -= rot(b, 16);
    a ^= c; a -= rot(c, 4);
    b ^= a; b -= rot(a, 14);
    c ^= b; c -= rot(b, 24);
    return c;
}
