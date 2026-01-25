# Gap 3: FP4 Packing and Vectorization

## FP4 Packed Format
```
torch.float4_e2m1fn_x2 storage:
- 2 FP4 values per uint8 byte
- First FP4: bits [3:0] (low nibble)
- Second FP4: bits [7:4] (high nibble)

Tensor shape transformation:
- Logical: [M, K, L] with K FP4 elements
- Physical: [M, K//2, L] uint8 bytes
```

## E2M1 Value Encoding
```
Bit pattern → Float value:
0000 (+0)  → 0.0
0001 (+1)  → 0.5
0010 (+2)  → 1.0
0011 (+3)  → 1.5
0100 (+4)  → 2.0
0101 (+5)  → 3.0
0110 (+6)  → 4.0
0111 (+7)  → 6.0
1000 (-0)  → -0.0
1001 (-1)  → -0.5
...
1111 (-7)  → -6.0

Note: Task constrains to {0, ±0.5, ±1, ±1.5} by masking:
ref_i8 = ref_i8 & 0b1011_1011  # Clear bit 2 in each nibble
```

## Alignment Requirements
```cpp
// Minimum alignment for vectorized loads
static_assert(alignof(ElementA) >= 1);  // FP4 packed as bytes

// For 128-bit (16-byte) vectorized loads:
assert(reinterpret_cast<uintptr_t>(a_ptr) % 16 == 0);

// For 256-bit (32-byte) loads (recommended for Blackwell):
assert(reinterpret_cast<uintptr_t>(a_ptr) % 32 == 0);

// K tile alignment (K=256 means 128 bytes per row tile)
static_assert(K_TILE % 256 == 0);  // 256 FP4 = 128 bytes
```

## Vectorized Load Pattern
```cpp
// Load 32 bytes = 64 FP4 values at once
using LoadType = uint4;  // 128 bits = 16 bytes
// Or for Blackwell:
using LoadType = uint8;  // 256 bits = 32 bytes = 64 FP4

// Unpack in registers
__device__ void unpack_fp4x2(uint8_t packed, float& v0, float& v1) {
    uint8_t low = packed & 0x0F;
    uint8_t high = packed >> 4;
    v0 = fp4_lut[low];   // Lookup table
    v1 = fp4_lut[high];
}
```

## Validation Gates
```cpp
// Gate 1: Pointer alignment
void validate_alignment(const void* ptr, size_t align, const char* name) {
    assert(reinterpret_cast<uintptr_t>(ptr) % align == 0 
           && "Pointer alignment violation");
}

// Gate 2: K tile alignment with packing
assert(K % 256 == 0);  // Ensures K/2 bytes is 128-byte aligned
assert((K / 2) % 32 == 0);  // 32-byte load alignment

// Gate 3: Verify unpacking
__host__ void test_unpack() {
    uint8_t packed = 0x31;  // low=1 (0.5), high=3 (1.5)
    float v0, v1;
    unpack_fp4x2(packed, v0, v1);
    assert(v0 == 0.5f && v1 == 1.5f);
}
```

## Common Pitfalls

### 1. Nibble Order Confusion
The low nibble (bits 0-3) contains the FIRST FP4 value, high nibble (bits 4-7) contains the SECOND. Reversing this silently corrupts data.

**CRITICAL**: This nibble order MUST be verified against PyTorch's `torch.float4_e2m1fn_x2`
convention. If wrong, ALL numerical results will be silently corrupted.

**Golden Vector Test (run before any kernel work):**
```python
# Byte 0x21 should unpack to (0.5, 1.0)
# Low nibble = 0x1 = 0.5, High nibble = 0x2 = 1.0
#
# If you get (1.0, 0.5) instead, your nibbles are SWAPPED!
byte_val = 0x21
low_nibble = byte_val & 0x0F   # 0x1 -> 0.5 (FIRST value)
high_nibble = byte_val >> 4    # 0x2 -> 1.0 (SECOND value)
```

See `tests/test_step2_fp4_nibble_order.py` for comprehensive validation.

### 2. Stride Calculation
When computing strides for packed tensors, remember that logical K elements become K/2 physical bytes:
```cpp
// WRONG: Using logical stride
size_t stride = K * sizeof(uint8_t);  

// CORRECT: Using physical stride
size_t stride = (K / 2) * sizeof(uint8_t);
```

### 3. Boundary Loads
When K is not a multiple of the vector load width, the last partial load may read garbage. Either:
- Pad K to alignment boundary
- Use masked loads for final iteration
- Ensure K % (2 * vector_width_bytes) == 0

## Integration with CUTLASS

CUTLASS 3.x expects specific alignment traits for custom element types:
```cpp
template<>
struct alignof_t<cutlass::float_e2m1_t> {
    static constexpr int value = 1;  // Byte aligned when packed
};

// For packed FP4x2:
template<>
struct sizeof_bits<cutlass::float_e2m1_t> {
    static constexpr int value = 4;  // 4 bits per element
};
```
