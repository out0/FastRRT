#include <cstdint>
#include <cstring>
#include <algorithm>

struct RandState {
    uint32_t s[4];
};

// Fast state init — just splitmix64 to fill the 4 words
void setupRandomGen(RandState* states, int size, uint64_t seed) {
    for (int i = 0; i < size; ++i) {
        uint64_t z = seed + i * 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        memcpy(states[i].s, &z, 8);
        uint64_t z2 = z + 0x9e3779b97f4a7c15ULL;
        z2 = (z2 ^ (z2 >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z2 = (z2 ^ (z2 >> 27)) * 0x94d049bb133111ebULL;
        z2 = z2 ^ (z2 >> 31);
        memcpy(states[i].s + 2, &z2, 8);
    }
}

// xoshiro128+ core — one step
inline uint32_t xoshiro_next(RandState& r) {
    const uint32_t result = r.s[0] + r.s[3];
    const uint32_t t = r.s[1] << 9;
    r.s[2] ^= r.s[0];
    r.s[3] ^= r.s[1];
    r.s[1] ^= r.s[2];
    r.s[0] ^= r.s[3];
    r.s[2] ^= t;
    r.s[3] = (r.s[3] << 11) | (r.s[3] >> 21); // rotl
    return result;
}

// Maps uint32 to float [0, 1) with zero division/branching
inline float to_float01(uint32_t x) {
    // Sets exponent bits to 127 (i.e. 1.x) then subtracts 1.0
    x = (x >> 9) | 0x3f800000u;
    float f;
    memcpy(&f, &x, 4);
    return f - 1.0f;
}

float generateRandom(RandState* states, int pos, float min_val, float max_val) {
    float u = to_float01(xoshiro_next(states[pos]));
    return std::max(min_val, max_val * u);
}

float generateRandomNeg(RandState* states, int pos, float max_val) {
    float u = to_float01(xoshiro_next(states[pos]));
    return (2.0f * u - 1.0f) * max_val;
}