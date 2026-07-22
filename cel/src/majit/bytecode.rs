//! Flat `i64`-word bytecode for the majit-traceable CEL subset, plus the majit
//! `#[jit_interp]` mainloop that evaluates it and a plain-`match` reference
//! interpreter (`clean_interp`) used as the correctness oracle and the honest
//! perf baseline.
//!
//! The instruction set is a simple three-address register machine over `i64`
//! words. Booleans are represented as `0`/`1` `i64` values. Every operand is a
//! register index (`usize`), every immediate an `i64`. Operator opcodes read
//! two source registers and write one destination; comparisons write `1`/`0`.
//! See [`super::lower`] for how a CEL AST is compiled into this form.

#![allow(dead_code)]

use core::sync::atomic::{AtomicUsize, Ordering};

/// A flat bytecode program: a stream of `i64` words (opcode followed by its
/// operands). Byte-index positions are word positions here.
pub type Code = [i64];

pub const OP_LOAD_CONST: i64 = 0; // [LOAD_CONST, imm, dst]        regs[dst] = imm
pub const OP_MOV: i64 = 1; // [MOV, src, dst]              regs[dst] = regs[src]
pub const OP_ADD: i64 = 2; // [ADD, a, b, dst]             regs[dst] = a + b
pub const OP_SUB: i64 = 3; // [SUB, a, b, dst]             regs[dst] = a - b
pub const OP_MUL: i64 = 4; // [MUL, a, b, dst]             regs[dst] = a * b
pub const OP_NEG: i64 = 5; // [NEG, a, dst]                regs[dst] = -a
pub const OP_GE: i64 = 6; // [GE, a, b, dst]              regs[dst] = (a >= b) as 0/1
pub const OP_GT: i64 = 7; // [GT, a, b, dst]              regs[dst] = (a >  b) as 0/1
pub const OP_LE: i64 = 8; // [LE, a, b, dst]              regs[dst] = (a <= b) as 0/1
pub const OP_LT: i64 = 9; // [LT, a, b, dst]              regs[dst] = (a <  b) as 0/1
pub const OP_EQ: i64 = 10; // [EQ, a, b, dst]              regs[dst] = (a == b) as 0/1
pub const OP_NE: i64 = 11; // [NE, a, b, dst]              regs[dst] = (a != b) as 0/1
pub const OP_AND: i64 = 12; // [AND, a, b, dst]             regs[dst] = a & b  (a,b in {0,1})
pub const OP_OR: i64 = 13; // [OR, a, b, dst]              regs[dst] = a | b  (a,b in {0,1})
pub const OP_NOT: i64 = 14; // [NOT, a, dst]                regs[dst] = 1 - a  (a in {0,1})
pub const OP_SELECT: i64 = 15; // [SELECT, c, t, f, dst]        regs[dst] = if regs[c]!=0 {regs[t]} else {regs[f]}
pub const OP_JUMP_IF_ABOVE: i64 = 16; // [JIA, a, b, tgt]  if regs[a] > regs[b] { pc = tgt } (loop back-edge)
pub const OP_RETURN: i64 = 17; // [RETURN, reg]                return regs[reg]
pub const OP_DIV: i64 = 18; // [DIV, a, b, dst]             regs[dst] = a / b   (b != 0; trunc toward zero)
pub const OP_MOD: i64 = 19; // [MOD, a, b, dst]             regs[dst] = a % b   (b != 0)
pub const OP_COL_LOAD: i64 = 20; // [COL_LOAD, base, ea, dst]  regs[dst] = *(regs[base] + regs[ea])

/// Raw native-memory load intrinsic recognized by the `#[jit_interp]` proc
/// macro (lowered to `raw_load_i`); at the interpreter tier this real fn runs.
/// `base` is a column buffer's base address, `ea` a byte offset — reading
/// `col[i]` at a data-dependent (red) row index `i` when `ea == i * 8`. The
/// base is a loop-invariant carried in the register file (NOT a scalar state
/// field, which would force the virtualizable frame to a Ref and trip
/// `VirtualStatesCantMatch` at loop close), exactly as a loop-invariant `rffi`
/// pointer is in PyPy. This is what lets a compiled trace read real context
/// columns per row instead of baking one row's inputs as constants.
#[inline]
fn majit_raw_load_i64(base: i64, ea: i64) -> i64 {
    // SAFETY: `base + ea` addresses element `ea/8` of a live `&[i64]` column
    // whose length the batch builder guarantees covers every row index.
    unsafe { core::ptr::read_unaligned((base as usize).wrapping_add(ea as usize) as *const i64) }
}

/// Counts hot loops majit compiled — evidence the JIT tier traced + compiled.
pub static COMPILES: AtomicUsize = AtomicUsize::new(0);

/// Counts guard-failure deopts.
pub static GUARD_FAILS: AtomicUsize = AtomicUsize::new(0);

struct VmState {
    regs: Vec<i64>,
}

#[majit_macros::jit_interp(
    state = VmState,
    env = Code,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
    },
)]
fn run_mainloop(program: &Code, num_regs: usize, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<VmState> =
        majit_metainterp::JitDriver::new(threshold);
    driver.set_on_compile_loop(|_green_key, _ops_before, _ops_after| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    driver.set_on_guard_failure(|_green_key, _a, _b| {
        GUARD_FAILS.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let _stacksize: i32 = 0;
    let mut state = VmState {
        regs: vec![0; num_regs],
    };

    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    loop {
        jit_merge_point!();
        let opcode = program[pc];
        match opcode {
            OP_LOAD_CONST => {
                let val = program[pc + 1];
                let dst = program[pc + 2] as usize;
                state.regs[dst] = val;
                pc += 3;
            }
            OP_MOV => {
                let src = program[pc + 1] as usize;
                let dst = program[pc + 2] as usize;
                state.regs[dst] = state.regs[src];
                pc += 3;
            }
            OP_ADD => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] + state.regs[b];
                pc += 4;
            }
            OP_SUB => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] - state.regs[b];
                pc += 4;
            }
            OP_MUL => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] * state.regs[b];
                pc += 4;
            }
            OP_DIV => {
                let a = state.regs[program[pc + 1] as usize];
                let b = state.regs[program[pc + 2] as usize];
                let d = program[pc + 3] as usize;
                // Truncating (toward-zero) division = cel's `/`. majit lowers a
                // bare `/` to Python floor division (toward -inf); floor and
                // truncation coincide on non-negative operands, so dividing the
                // magnitudes and reapplying the sign branchlessly stays bit-exact
                // and lowers identically in the interpreter and compiled tiers.
                // Inlined (not a helper call): a residual call aborts the trace.
                let ma = a >> 63; // 0 or -1 (sign mask of a)
                let mb = b >> 63;
                let ua = (a ^ ma) - ma; // |a|
                let ub = (b ^ mb) - mb; // |b|
                let uq = ua / ub; // non-negative quotient: floor == trunc here
                let s = ma ^ mb; // -1 iff signs differ
                state.regs[d] = (uq ^ s) - s; // negate quotient iff signs differ
                pc += 4;
            }
            OP_MOD => {
                let a = state.regs[program[pc + 1] as usize];
                let b = state.regs[program[pc + 2] as usize];
                let d = program[pc + 3] as usize;
                // Truncating remainder = cel's `%` (sign of the dividend).
                let ma = a >> 63;
                let mb = b >> 63;
                let ua = (a ^ ma) - ma;
                let ub = (b ^ mb) - mb;
                let ur = ua % ub; // non-negative remainder
                state.regs[d] = (ur ^ ma) - ma; // reapply the dividend's sign
                pc += 4;
            }
            OP_NEG => {
                let a = program[pc + 1] as usize;
                let d = program[pc + 2] as usize;
                state.regs[d] = -state.regs[a];
                pc += 3;
            }
            OP_GE => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = if state.regs[a] >= state.regs[b] { 1 } else { 0 };
                pc += 4;
            }
            OP_GT => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = if state.regs[a] > state.regs[b] { 1 } else { 0 };
                pc += 4;
            }
            OP_LE => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = if state.regs[a] <= state.regs[b] { 1 } else { 0 };
                pc += 4;
            }
            OP_LT => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = if state.regs[a] < state.regs[b] { 1 } else { 0 };
                pc += 4;
            }
            OP_EQ => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = if state.regs[a] == state.regs[b] { 1 } else { 0 };
                pc += 4;
            }
            OP_NE => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = if state.regs[a] != state.regs[b] { 1 } else { 0 };
                pc += 4;
            }
            OP_AND => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] & state.regs[b];
                pc += 4;
            }
            OP_OR => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] | state.regs[b];
                pc += 4;
            }
            OP_NOT => {
                let a = program[pc + 1] as usize;
                let d = program[pc + 2] as usize;
                state.regs[d] = 1 - state.regs[a];
                pc += 3;
            }
            OP_SELECT => {
                let c = program[pc + 1] as usize;
                let t = program[pc + 2] as usize;
                let f = program[pc + 3] as usize;
                let d = program[pc + 4] as usize;
                // Branchless blend: the ternary condition is a bool (0/1), so
                // `f + c*(t-f)` == `if c!=0 {t} else {f}`. Branchless keeps the
                // ternary out of a data-dependent guard (which would deopt every
                // time the condition flips) and off majit's virt-array blackhole
                // resume path.
                state.regs[d] =
                    state.regs[f] + state.regs[c] * (state.regs[t] - state.regs[f]);
                pc += 5;
            }
            OP_COL_LOAD => {
                let base = state.regs[program[pc + 1] as usize];
                let ea = state.regs[program[pc + 2] as usize];
                let d = program[pc + 3] as usize;
                state.regs[d] = majit_raw_load_i64(base, ea);
                pc += 4;
            }
            OP_JUMP_IF_ABOVE => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let tgt = program[pc + 3] as usize;
                if state.regs[a] > state.regs[b] {
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                pc += 4;
            }
            OP_RETURN => {
                let r = program[pc + 1] as usize;
                return state.regs[r];
            }
            _ => break,
        }
    }
    panic!("fell off end of code");
}

/// Evaluate `program` on the majit-traced mainloop. `threshold` is the JIT hot
/// threshold (`u32::MAX` disables compilation, giving the interpreter tier).
pub fn run_jit(program: &Code, num_regs: usize, threshold: u32) -> i64 {
    run_mainloop(program, num_regs, threshold)
}

/// Columnar **batch** evaluation of a lowered CEL expression: reduce
/// `sum over rows i of expr(col_0[i], col_1[i], ..)` where `columns[k]` is
/// slot `k`'s `i64` data column (aligned to [`super::lower::Lowered::slots`],
/// all the same length). For a boolean predicate this counts matching rows;
/// for an arithmetic expression it sums the per-row values. `threshold` is the
/// JIT hot threshold (`u32::MAX` = interpreter tier).
///
/// This is the real throughput path: the compiled trace reads each column at
/// the red row index via `raw_load` (base carried loop-invariant in a register)
/// instead of re-baking one row's inputs as constants per call.
pub fn eval_batch_sum(
    lowered: &super::lower::Lowered,
    columns: &[&[i64]],
    threshold: u32,
) -> i64 {
    assert_eq!(
        columns.len(),
        lowered.slots.len(),
        "eval_batch_sum: column count {} != slot count {}",
        columns.len(),
        lowered.slots.len()
    );
    let n = columns.first().map_or(0, |c| c.len());
    for (k, c) in columns.iter().enumerate() {
        assert_eq!(c.len(), n, "eval_batch_sum: column {k} length {} != {n}", c.len());
    }
    if n == 0 {
        return 0;
    }
    let bases: Vec<i64> = columns.iter().map(|c| c.as_ptr() as i64).collect();
    let (prog, total_regs) = lowered.batch_sum_program(&bases, n as i64);
    let result = run_mainloop(&prog, total_regs, threshold);
    // The raw pointers in `prog` alias `columns`; keep the borrow live across
    // the run so the buffers cannot be dropped underneath the trace.
    core::hint::black_box(columns);
    result
}

/// Reference interpreter: a plain `match` over the same bytecode with no majit
/// machinery. The correctness oracle for the lowering and the honest baseline
/// for "did the JIT actually speed anything up".
pub fn clean_interp(program: &Code, num_regs: usize) -> i64 {
    let mut regs = vec![0i64; num_regs];
    let mut pc = 0usize;
    loop {
        match program[pc] {
            OP_LOAD_CONST => {
                regs[program[pc + 2] as usize] = program[pc + 1];
                pc += 3;
            }
            OP_MOV => {
                regs[program[pc + 2] as usize] = regs[program[pc + 1] as usize];
                pc += 3;
            }
            OP_ADD => {
                regs[program[pc + 3] as usize] =
                    regs[program[pc + 1] as usize] + regs[program[pc + 2] as usize];
                pc += 4;
            }
            OP_SUB => {
                regs[program[pc + 3] as usize] =
                    regs[program[pc + 1] as usize] - regs[program[pc + 2] as usize];
                pc += 4;
            }
            OP_MUL => {
                regs[program[pc + 3] as usize] =
                    regs[program[pc + 1] as usize] * regs[program[pc + 2] as usize];
                pc += 4;
            }
            OP_DIV => {
                regs[program[pc + 3] as usize] =
                    regs[program[pc + 1] as usize] / regs[program[pc + 2] as usize];
                pc += 4;
            }
            OP_MOD => {
                regs[program[pc + 3] as usize] =
                    regs[program[pc + 1] as usize] % regs[program[pc + 2] as usize];
                pc += 4;
            }
            OP_NEG => {
                regs[program[pc + 2] as usize] = -regs[program[pc + 1] as usize];
                pc += 3;
            }
            OP_GE => {
                regs[program[pc + 3] as usize] =
                    (regs[program[pc + 1] as usize] >= regs[program[pc + 2] as usize]) as i64;
                pc += 4;
            }
            OP_GT => {
                regs[program[pc + 3] as usize] =
                    (regs[program[pc + 1] as usize] > regs[program[pc + 2] as usize]) as i64;
                pc += 4;
            }
            OP_LE => {
                regs[program[pc + 3] as usize] =
                    (regs[program[pc + 1] as usize] <= regs[program[pc + 2] as usize]) as i64;
                pc += 4;
            }
            OP_LT => {
                regs[program[pc + 3] as usize] =
                    (regs[program[pc + 1] as usize] < regs[program[pc + 2] as usize]) as i64;
                pc += 4;
            }
            OP_EQ => {
                regs[program[pc + 3] as usize] =
                    (regs[program[pc + 1] as usize] == regs[program[pc + 2] as usize]) as i64;
                pc += 4;
            }
            OP_NE => {
                regs[program[pc + 3] as usize] =
                    (regs[program[pc + 1] as usize] != regs[program[pc + 2] as usize]) as i64;
                pc += 4;
            }
            OP_AND => {
                regs[program[pc + 3] as usize] =
                    regs[program[pc + 1] as usize] & regs[program[pc + 2] as usize];
                pc += 4;
            }
            OP_OR => {
                regs[program[pc + 3] as usize] =
                    regs[program[pc + 1] as usize] | regs[program[pc + 2] as usize];
                pc += 4;
            }
            OP_NOT => {
                regs[program[pc + 2] as usize] = 1 - regs[program[pc + 1] as usize];
                pc += 3;
            }
            OP_SELECT => {
                let c = regs[program[pc + 1] as usize];
                let t = regs[program[pc + 2] as usize];
                let f = regs[program[pc + 3] as usize];
                regs[program[pc + 4] as usize] = f + c * (t - f);
                pc += 5;
            }
            OP_COL_LOAD => {
                let base = regs[program[pc + 1] as usize];
                let ea = regs[program[pc + 2] as usize];
                regs[program[pc + 3] as usize] = majit_raw_load_i64(base, ea);
                pc += 4;
            }
            OP_JUMP_IF_ABOVE => {
                let tgt = program[pc + 3] as usize;
                if regs[program[pc + 1] as usize] > regs[program[pc + 2] as usize] {
                    pc = tgt;
                } else {
                    pc += 4;
                }
            }
            OP_RETURN => return regs[program[pc + 1] as usize],
            _ => panic!("bad op {}", program[pc]),
        }
    }
}
