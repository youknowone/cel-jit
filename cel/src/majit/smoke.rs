//! M1 smoke loop: a tinyframe-style register machine authored in majit's
//! traceable subset, proving majit traces, compiles, and *correctly runs* a
//! looping native trace from inside the `cel` crate. See [`super`] for why.
//!
//! Shape:
//! - `env` is a `[u8]` byte stream (`program[pc]`), byte-addressed bytecode.
//! - the register file is `[int; virt]` (virtualizable), NOT plain `[int]`: a
//!   loop-carried plain `[int]` element is held in a trace register and is not
//!   restored to the array on a CloseLoop guard deopt (it reads back as the
//!   pre-loop value). A virt array writes through to the heap Vec, which the
//!   deopt path reads directly.

// M1 scaffolding: `run_smoke`/the counters are exercised only from tests, and
// the `stacksize` local is template boilerplate the macro expects.
#![allow(dead_code, unused_variables, unused_mut)]

use core::sync::atomic::{AtomicUsize, Ordering};

/// The `env`: a byte-addressed bytecode stream (`program[pc]`). Mirrors
/// tinyframe's `pub type Bytecode = [u8];`.
pub type Bytecode = [u8];

const OP_LOAD: u8 = 4; // [LOAD, imm, dst_reg]
const OP_ADD: u8 = 0; // [ADD, a_reg, b_reg, dst_reg]
const OP_JUMP_IF_ABOVE: u8 = 8; // [JIA, a_reg, b_reg, target_pc]
const OP_RETURN: u8 = 6; // [RETURN, reg]

/// Counts hot loops majit compiled — evidence the JIT tier traced + compiled,
/// not just interpreted.
pub static SMOKE_COMPILES: AtomicUsize = AtomicUsize::new(0);

/// Counts guard-failure deopts. Distinguishes "compiled trace runs the loop
/// itself" (≈1 deopt = the loop-exit side exit) from "compiled trace bails to
/// the interpreter every iteration" (≈N deopts).
pub static SMOKE_GUARD_FAILS: AtomicUsize = AtomicUsize::new(0);

struct SmokeState {
    regs: Vec<i64>,
}

#[majit_macros::jit_interp(
    state = SmokeState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        regs: [int; virt],
    },
)]
fn smoke_mainloop(program: &Bytecode, num_regs: usize, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<SmokeState> =
        majit_metainterp::JitDriver::new(threshold);
    driver.set_on_compile_loop(|_green_key, _ops_before, _ops_after| {
        SMOKE_COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    driver.set_on_guard_failure(|_green_key, _a, _b| {
        SMOKE_GUARD_FAILS.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = SmokeState {
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
            OP_LOAD => {
                let val = program[pc + 1] as i64;
                let reg = program[pc + 2] as usize;
                state.regs[reg] = val;
                pc += 3;
            }
            OP_ADD => {
                let a = program[pc + 1] as usize;
                let b = program[pc + 2] as usize;
                let d = program[pc + 3] as usize;
                state.regs[d] = state.regs[a] + state.regs[b];
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

/// Run the smoke mainloop. `num_regs` registers, JIT `threshold`.
pub fn run_smoke(program: &Bytecode, num_regs: usize, threshold: u32) -> i64 {
    smoke_mainloop(program, num_regs, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the smoke tests, which all drive `run_smoke` and so increment
    /// the global [`SMOKE_COMPILES`] / [`SMOKE_GUARD_FAILS`] evidence counters: a
    /// concurrent smoke run would otherwise pollute another test's
    /// reset-run-assert window. Poison-tolerant so one failure doesn't cascade.
    fn smoke_serial_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `r0` counts up by 1 while `r2`(=n) > `r0`, so it exits at `r0 == n`.
    /// Loop body at pc 9; back-edge target `9 < pc`. `n` must fit a byte.
    fn count_to_program(n: u8) -> Vec<u8> {
        vec![
            OP_LOAD,
            1,
            1, // r1 = 1
            OP_LOAD,
            n,
            2, // r2 = n
            OP_LOAD,
            0,
            0, // r0 = 0
            // @l1 = pc 9
            OP_ADD,
            0,
            1,
            0, // r0 = r0 + r1
            OP_JUMP_IF_ABOVE,
            2,
            0,
            9, // if r2 > r0 goto @l1
            OP_RETURN,
            0, // return r0
        ]
    }

    /// Proves the full integration: the majit meta-tracer traces this loop
    /// inside the `cel` crate, **compiles a native loop**, and yields the
    /// correct result.
    ///
    /// `greens = [pc, program]` is load-bearing: without it `pc` is not green,
    /// so the operand reads (`program[pc + N]`) cannot constant-fold and the
    /// trace aborts at the loop body. With it, the loop compiles (observe with
    /// `MAJIT_LOG=1`: `trace action … -> CloseLoop` + `[jit][compile-loop]`).
    #[test]
    fn smoke_loop_count_to_40_and_compiles() {
        let _serial = smoke_serial_guard();
        SMOKE_COMPILES.store(0, Ordering::Relaxed);
        let program = count_to_program(40);
        let result = run_smoke(&program, 3, 3);
        assert_eq!(result, 40, "majit-traced loop must compute 40");
        assert!(
            SMOKE_COMPILES.load(Ordering::Relaxed) >= 1,
            "majit should have compiled the hot loop at least once"
        );
    }

    /// Correctness across inputs: run the SAME compiled loop for several
    /// targets `n`, exercising the CloseLoop guard-exit deopt on each.
    #[test]
    fn smoke_loop_varies_n() {
        let _serial = smoke_serial_guard();
        for n in [1_u8, 2, 5, 10, 40, 100, 200] {
            let program = count_to_program(n);
            let result = run_smoke(&program, 3, 3);
            assert_eq!(result, n as i64, "count_to({n}) compiled-trace mismatch");
        }
    }

    /// The COMPILED trace runs the loop itself rather than bailing to the
    /// interpreter every iteration: counting to 200 with threshold 3 deopts a
    /// small constant number of times (the loop-exit side exit + warmup), not
    /// ≈200 times.
    #[test]
    fn smoke_compiled_trace_executes_hot_path() {
        let _serial = smoke_serial_guard();
        SMOKE_COMPILES.store(0, Ordering::Relaxed);
        SMOKE_GUARD_FAILS.store(0, Ordering::Relaxed);
        let program = count_to_program(200);
        let result = run_smoke(&program, 3, 3);
        assert_eq!(result, 200);
        let compiles = SMOKE_COMPILES.load(Ordering::Relaxed);
        let deopts = SMOKE_GUARD_FAILS.load(Ordering::Relaxed);
        std::eprintln!("[smoke] count_to(200): compiles={compiles} guard_fails={deopts}");
        assert!(compiles >= 1, "must compile");
        assert!(
            deopts <= 5,
            "compiled trace should run the hot loop itself (≈1 deopt at exit), \
             got {deopts} deopts — indicates per-iteration bail to interpreter"
        );
    }
}
