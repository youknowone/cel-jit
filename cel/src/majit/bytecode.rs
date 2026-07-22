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
pub const OP_JUMP_IF_ABOVE: i64 = 15; // [JIA, a, b, tgt]  if regs[a] > regs[b] { pc = tgt } (loop back-edge)
pub const OP_RETURN: i64 = 16; // [RETURN, reg]                return regs[reg]

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
