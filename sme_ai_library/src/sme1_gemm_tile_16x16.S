    .text
    .align 4
    .global sme1_gemm_tile_16x16_k
    .type sme1_gemm_tile_16x16_k, %function
    .global sme1_gemm_tile_16x16_strided_k
    .type sme1_gemm_tile_16x16_strided_k, %function

// SME1 16x16 GEMM micro-kernel
//
// x0 = packed A, layout [k][16]
// x1 = packed B, layout [k][16]
// x2 = output tile, contiguous [16][16]
// x3 = k iterations
//
// Caller contract:
// - streaming mode is already enabled
// - this helper only computes one 16x16 tile body
sme1_gemm_tile_16x16_k:
    ptrue p0.s
    zero {za}

1:
    cbz x3, 2f
    ldr z0, [x0]
    ldr z1, [x1]
    fmopa za0.s, p0/m, p0/m, z0.s, z1.s
    add x0, x0, #64
    add x1, x1, #64
    sub x3, x3, #1
    b 1b

2:
    mov w12, #0
    mov x4, #16

3:
    str za[w12, #0], [x2]
    add w12, w12, #4
    add x2, x2, #64
    sub x4, x4, #1
    cbnz x4, 3b

    ret

// SME1 16x16 GEMM micro-kernel with strided output rows
//
// x0 = packed A, layout [k][16]
// x1 = packed B, layout [k][16]
// x2 = output tile base, row-major matrix
// x3 = k iterations
// x4 = output row stride in bytes
//
// Caller contract:
// - streaming mode is already enabled
// - the output tile is a full 16x16 tile
sme1_gemm_tile_16x16_strided_k:
    ptrue p0.s
    zero {za}

4:
    cbz x3, 5f
    ldr z0, [x0]
    ldr z1, [x1]
    fmopa za0.s, p0/m, p0/m, z0.s, z1.s
    add x0, x0, #64
    add x1, x1, #64
    sub x3, x3, #1
    b 4b

5:
    mov w12, #0
    mov x5, #16

6:
    str za[w12, #0], [x2]
    add w12, w12, #4
    add x2, x2, x4
    sub x5, x5, #1
    cbnz x5, 6b

    ret
