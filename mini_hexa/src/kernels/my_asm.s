    .text
    .p2align 2
    .p2align 4,,15
    .globl simple_function_asm
    .type simple_function_asm, @function
simple_function_asm:
    loop0(loop_simple_function_am, #10)
loop_simple_function_am:
    { r0 = add(r0, #13 ) }: endloop0

    { jumpr r31 } // return
    .size	simple_function_asm, .-simple_function_asm


    .text
    .p2align 2
    .p2align 4,,15
    .globl micro_hvx_qf32
    .type micro_hvx_qf32, @function
micro_hvx_qf32:
    loop0(loop_micro_hvx_qf32, r0)
loop_micro_hvx_qf32:
    {   v0.qf32 = vmpy( v30.qf32, v31.qf32 ),
        v1.qf32 = vadd( v30.qf32, v31.qf32 ) }: endloop0

    { r0 = #32*2 }

    { jumpr r31 } // return
    .size	micro_hvx_qf32, .-micro_hvx_qf32


    .p2align 2
    .p2align 4,,15
    .globl gemm_asm_cdsp_192_4_128
    .type gemm_asm_cdsp_192_4_128, @function
gemm_asm_cdsp_192_4_128:
    // TODO: finished implementation

    { jumpr r31 } // return
    .size	gemm_asm_cdsp_192_4_128, .-gemm_asm_cdsp_192_4_128
