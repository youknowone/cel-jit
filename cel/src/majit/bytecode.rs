//! Flat `i64`-word bytecode for the majit-traceable CEL subset, plus the majit
//! `#[jit_interp]` mainloop that evaluates it ([`float_bank::run_mainloop_f`])
//! and a plain-`match` reference interpreter ([`float_bank::clean_interp_seeded_f`])
//! used as the correctness oracle and the honest perf baseline.
//!
//! The instruction set is a three-address register machine over two banks: an
//! `i64` bank (`regs`) carrying ints, bools as `0`/`1`, `uint` as a raw bit
//! pattern, strings as content hashes and temporals as nanos, and a parallel
//! `f64` bank (`fregs`). Every operand is a register index (`usize`), every
//! immediate an `i64`. Operator opcodes read two source registers and write one
//! destination; comparisons write `1`/`0` into the int bank whichever bank the
//! operands live in. See [`super::lower`] for how a CEL AST is compiled into
//! this form.

#![allow(dead_code)]

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

// Float-bank opcodes (the two-bank machine, see the `float_bank` module). The
// int opcodes above address the `regs: [int; virt]` bank (addressing, bool
// results, count); these address the `fregs: [float; virt]` bank (column
// values). A float comparison crosses banks: float operands, int `0`/`1`
// result. Semantics mirror the CEL tree-walker's f64 rules — arithmetic needs
// both operands float (no int promotion), division is plain IEEE (no
// divide-by-zero guard), there is no float modulo; comparisons match `f64`.
pub const OP_COL_LOAD_F: i64 = 21; // [base, ea, fdst]  fregs[fdst] = *f64(regs[base] + regs[ea])
pub const OP_LOAD_CONST_F: i64 = 22; // [bits, fdst]    fregs[fdst] = f64::from_bits(bits as u64)
pub const OP_FMOV: i64 = 23; // [fsrc, fdst]            fregs[fdst] = fregs[fsrc]
pub const OP_FADD: i64 = 24; // [fa, fb, fdst]          fregs[fdst] = fregs[fa] + fregs[fb]
pub const OP_FSUB: i64 = 25; // [fa, fb, fdst]          fregs[fdst] = fregs[fa] - fregs[fb]
pub const OP_FMUL: i64 = 26; // [fa, fb, fdst]          fregs[fdst] = fregs[fa] * fregs[fb]
pub const OP_FDIV: i64 = 27; // [fa, fb, fdst]          fregs[fdst] = fregs[fa] / fregs[fb]  (IEEE)
pub const OP_FNEG: i64 = 28; // [fa, fdst]              fregs[fdst] = -fregs[fa]
pub const OP_FGE: i64 = 29; // [fa, fb, dst]            regs[dst] = (fregs[fa] >= fregs[fb]) as 0/1
pub const OP_FGT: i64 = 30; // [fa, fb, dst]            regs[dst] = (fregs[fa] >  fregs[fb]) as 0/1
pub const OP_FLE: i64 = 31; // [fa, fb, dst]            regs[dst] = (fregs[fa] <= fregs[fb]) as 0/1
pub const OP_FLT: i64 = 32; // [fa, fb, dst]            regs[dst] = (fregs[fa] <  fregs[fb]) as 0/1
pub const OP_FEQ: i64 = 33; // [fa, fb, dst]            regs[dst] = (fregs[fa] == fregs[fb]) as 0/1
pub const OP_FNE: i64 = 34; // [fa, fb, dst]            regs[dst] = (fregs[fa] != fregs[fb]) as 0/1
pub const OP_I2F: i64 = 35; // [src, fdst]             fregs[fdst] = regs[src] as f64  (cast_int_to_float)
pub const OP_RETURN_F: i64 = 36; // [RETURN_F, freg]        return fregs[freg].to_bits()  (float total)
pub const OP_FSELECT: i64 = 37; // [FSELECT, c, ft, ff, fdst]  fregs[fdst] = if regs[c]!=0 {fregs[ft]} else {fregs[ff]}
pub const OP_ULT: i64 = 38; // [a, b, dst]              regs[dst] = ((regs[a] as u64) <  (regs[b] as u64)) as 0/1
pub const OP_ULE: i64 = 39; // [a, b, dst]              regs[dst] = ((regs[a] as u64) <= (regs[b] as u64)) as 0/1

// Overflow-checked user arithmetic. `OP_ADD`/`OP_SUB`/`OP_MUL` above stay plain
// (wrapping) for the batch machinery's own counter/address/accumulator, which
// operate on controlled values and whose cross-row sum must match the oracle's
// plain `+=`. These carry the overflow-is-checked semantics for the user
// expression: the no-overflow path is a fused `int_*_jump_if_ovf` (traced to
// `Int*Ovf` + `GuardNoOverflow`); on overflow the guard deopts into the None
// arm, which the blackhole runs on the virtualizable resume path.
//
// The tree-walker RAISES `ExecutionError::Overflow` on these, so a wrapped
// result is not an answer — the None arm records the event in `regs[trap]` and
// the batch driver turns a set flag into "no result, use the tree-walker".
// `trap` is a plain register, so the write costs nothing on the hot path (it
// only executes on the guard-exit resume) and needs no memory channel.
pub const OP_ADD_OVF: i64 = 40; // [a, b, dst, trap]    regs[dst] = ovfchecked(a + b)
pub const OP_SUB_OVF: i64 = 41; // [a, b, dst, trap]    regs[dst] = ovfchecked(a - b)
pub const OP_MUL_OVF: i64 = 42; // [a, b, dst, trap]    regs[dst] = ovfchecked(a * b)
/// Publish the overflow flag to the caller. Emitted **once** in the batch
/// epilogue (after the loop's back-edge), never in the traced body: the single
/// i64 a mainloop returns is the accumulated sum, so the flag needs its own
/// channel out. `regs[addr]` holds the address of a caller-owned i64 word, the
/// same loop-invariant-pointer-in-a-register shape the column bases use.
pub const OP_TRAP_STORE: i64 = 43; // [addr, flag]      *(regs[addr]) = regs[flag]
/// Narrow a float-bank value into the int bank (`cast_float_to_int`): Rust's
/// `as i64`, truncating toward zero and SATURATING at the i64 bounds, with NaN
/// mapping to 0. That is total — the walker's `int(double)` is the same plain
/// `as` cast (`common/types/int.rs:237-239`) and raises nothing — so no guard
/// is needed. The inverse of [`OP_I2F`].
pub const OP_F2I: i64 = 44; // [fsrc, dst]             regs[dst] = fregs[fsrc] as i64  (cast_float_to_int)

// Domain-guarded division. The tree-walker's `/` and `%` are partial on both
// numeric banks: a zero divisor raises `DivisionByZero`/`RemainderByZero`, and
// `INT_MIN / -1` raises `Overflow` (`common/types/int.rs:119-143` uses
// `checked_div`/`checked_rem`). RPython spells the same partiality as two
// guards emitted UPSTREAM of the division helper — `int_eq(rhs, 0)` and
// `(lhs == INT_MIN) & (rhs == -1)`, both `guard_false`, inlined from
// `rint.py:429 ll_int_py_div_ovf_zer` — so these carry that guard and record
// the trap on its failure, exactly as the `OP_*_OVF` group does.
//
// The unguarded [`OP_DIV`]/[`OP_MOD`] above stay for the divisions this
// lowering creates itself: the temporal accessors divide by a green constant
// that is nonzero by construction, and paying a per-row guard for a divisor the
// optimizer can see is a constant would be waste. They are also what the legacy
// int-only `lower` path emits, which reserves no trap register.
pub const OP_DIV_CHK: i64 = 45; // [a, b, dst, trap]  regs[dst] = a / b   (trunc toward zero)
pub const OP_MOD_CHK: i64 = 46; // [a, b, dst, trap]  regs[dst] = a % b   (sign of dividend)
/// Unsigned division on the int bank, the `uint` peer of [`OP_DIV_CHK`]. Only
/// the zero divisor is guarded — every pair of `u64` operands with a nonzero
/// divisor has a representable quotient, so there is no `INT_MIN / -1` corner.
pub const OP_UDIV: i64 = 47; // [a, b, dst, trap]  regs[dst] = (a as u64) / (b as u64)
pub const OP_UMOD: i64 = 48; // [a, b, dst, trap]  regs[dst] = (a as u64) % (b as u64)

// Overflow-checked UNSIGNED arithmetic, the `uint` peers of `OP_*_OVF`. The
// values these produce are bit-identical to the signed ops (two's complement
// `+ - *` do not care about signedness) — the CHECK is what differs, and the
// signed `Int*Ovf` guard answers it wrongly in both directions: `2^63 + 1` is a
// fine `uint` but overflows signed, and `0u - 1u` is the reverse. The
// tree-walker uses `u64::checked_*` (`common/types/uint.rs:78-196`), so the
// unsigned condition is the one to guard.
//
// RPython has no unsigned overflow resop either (`int_add_ovf` is signed-only,
// and `r_uint` arithmetic simply WRAPS in RPython — it is CEL, not RPython,
// that makes these partial). The condition is therefore built from ops the
// trace already has: a carry is `sum <u lhs`, a borrow is `lhs <u rhs`, and a
// product overflows exactly when the high word of the 128-bit result is
// nonzero, which is what `uint_mul_high` returns.
pub const OP_UADD_OVF: i64 = 49; // [a, b, dst, trap]  regs[dst] = ovfchecked_u(a + b)
pub const OP_USUB_OVF: i64 = 50; // [a, b, dst, trap]  regs[dst] = ovfchecked_u(a - b)
pub const OP_UMUL_OVF: i64 = 51; // [a, b, dst, trap]  regs[dst] = ovfchecked_u(a * b)

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

/// Raw native-memory store intrinsic (`raw_store_i`), the write-side analogue of
/// [`majit_raw_load_i64`]. Used only by [`OP_TRAP_STORE`] to publish the
/// overflow flag to the batch driver.
#[inline]
fn majit_raw_store_i64(base: i64, ea: i64, val: i64) {
    // SAFETY: `base + ea` addresses the caller's live `i64` trap word, which the
    // batch driver keeps alive across the whole run.
    unsafe {
        core::ptr::write_unaligned((base as usize).wrapping_add(ea as usize) as *mut i64, val)
    }
}

/// One input column for the two-bank batch evaluator: an `i64` column for an
/// int/bool slot, or an `f64` column for a `double` slot. Its base pointer (an
/// `i64` regardless of bank) is what a compiled trace reads per row.
pub enum Column<'a> {
    Int(&'a [i64]),
    Float(&'a [f64]),
}

impl Column<'_> {
    /// Base address of the column buffer, as the `i64` a `raw_load` base holds.
    pub fn base(&self) -> i64 {
        match self {
            Column::Int(c) => c.as_ptr() as i64,
            Column::Float(c) => c.as_ptr() as i64,
        }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            Column::Int(c) => c.len(),
            Column::Float(c) => c.len(),
        }
    }

    /// True if the column is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn matches(&self, ty: super::lower::ValType) -> bool {
        use super::lower::ValType;
        // A `uint` slot is backed by an int-bit column (the int register file
        // carries the raw 64-bit pattern); a `string` slot is backed by an int
        // column of content-hash ids.
        matches!(
            (self, ty),
            (
                Column::Int(_),
                ValType::Int
                    | ValType::UInt
                    | ValType::Str
                    | ValType::Timestamp
                    | ValType::Duration
            ) | (Column::Float(_), ValType::Float)
        )
    }
}

/// Build the batch program for `n` rows over `columns` and run it with `run`,
/// which selects the tier. Shared by [`eval_batch_sum_f`] (the majit tier) and
/// [`clean_batch_sum_f`] (the oracle tier) so the trap protocol — allocate the
/// caller-owned word, seed its address into the initial register bank, read it back — is
/// written once.
fn batch_sum_with(
    lowered: &super::lower::LoweredF,
    columns: &[Column],
    n: usize,
    what: &str,
    run: impl FnOnce(&Code, &[i64], usize) -> i64,
) -> Option<i64> {
    assert_eq!(
        columns.len(),
        lowered.slots.len(),
        "{what}: column count {} != slot count {}",
        columns.len(),
        lowered.slots.len()
    );
    for (k, (col, slot)) in columns.iter().zip(&lowered.slots).enumerate() {
        assert!(
            col.matches(slot.ty),
            "{what}: column {k} bank mismatch vs slot `{}` ({:?})",
            slot.path,
            slot.ty
        );
    }
    // `n` is the caller's row count, not something the columns can be asked
    // for: a list's flattened ELEMENT column is as long as the batch's total
    // element count, and an expression that folds to a constant has no column
    // at all. Every ROW column must agree with it.
    use super::lower::SlotKind;
    for (k, (c, slot)) in columns.iter().zip(&lowered.slots).enumerate() {
        if slot.kind != SlotKind::Row {
            continue;
        }
        assert_eq!(c.len(), n, "{what}: column {k} length {} != {n}", c.len());
    }
    if n == 0 {
        return Some(0);
    }
    let bases: Vec<i64> = columns.iter().map(|c| c.base()).collect();
    // The trap word must outlive the run and must not be aliased by a reference
    // while the program writes it through the raw pointer seeded into `init_regs`.
    let mut trap: Box<i64> = Box::new(0);
    let trap_addr = (&mut *trap) as *mut i64 as i64;
    let shape = lowered.batch_sum_shape(true);
    let init_regs = shape.seed.regs(&bases, n as i64, trap_addr);
    // The words are the same for every batch of this expression, so interning
    // them keeps the JIT's green key — and with it the compiled loop the driver
    // holds — from changing between batches.
    let code = float_bank::intern_program(shape.code);
    let result = run(&code, &init_regs, shape.num_float_regs);
    // The raw pointers in `init_regs` alias `columns`; keep the borrow live
    // across the run so the buffers cannot be dropped underneath the trace.
    core::hint::black_box(columns);
    if *trap != 0 {
        return None;
    }
    Some(result)
}

/// Columnar **batch** evaluation of a typed (two-bank) lowered expression:
/// reduce `sum over rows i of expr(col_0[i], ..)` where `columns[k]` is slot
/// `k`'s data column, aligned to [`super::lower::LoweredF::slots`] and matching
/// each slot's bank. For a boolean predicate this counts matching rows. The
/// compiled trace reads each column at the red row index via `raw_load` (base
/// carried loop-invariant in an int register), int columns as `i64`, float
/// columns as `f64`. `threshold == u32::MAX` gives the interpreter tier.
///
/// `n` is the number of rows. It is the caller's to state: a lowering may have
/// no row column to read it off (an ELEMENT column of a runtime-length list is
/// as long as the flattened element count, and a constant-folded expression has
/// no slots at all). Every row column must have length `n`, which is asserted.
///
/// Returns `None` when a row's `int` arithmetic OVERFLOWED. The tree-walker
/// raises `ExecutionError::Overflow` there, so no sum is the right answer; the
/// caller falls back to the tree-walker, which owns the error. This is the
/// batch transposition of PyPy's `guard_no_overflow` deopt: the guard exits to
/// the interpreter, and the interpreter is what raises.
pub fn eval_batch_sum_f(
    lowered: &super::lower::LoweredF,
    columns: &[Column],
    n: usize,
    threshold: u32,
) -> Option<i64> {
    batch_sum_with(lowered, columns, n, "eval_batch_sum_f", |prog, regs, nf| {
        float_bank::run_jit_persistent_f(prog, regs, nf, threshold)
    })
}

/// [`eval_batch_sum_f`] on the oracle tier: the same batch program run by the
/// plain-`match` [`float_bank::clean_interp_seeded_f`], with no tracing or
/// compilation
/// machinery in the loop. This is what a majit result is checked against.
pub fn clean_batch_sum_f(
    lowered: &super::lower::LoweredF,
    columns: &[Column],
    n: usize,
) -> Option<i64> {
    batch_sum_with(
        lowered,
        columns,
        n,
        "clean_batch_sum_f",
        float_bank::clean_interp_seeded_f,
    )
}

/// Columnar batch sum for a **float-valued** lowering: the per-row result is a
/// float and the running total is a float accumulator, so the returned `i64`
/// bits ([`OP_RETURN_F`]) are reinterpreted as the `f64` total. The batch loop
/// sums left to right in row order, matching the tree-walker oracle bit for bit
/// (float addition is order-sensitive, so the order must agree). Requires
/// `lowered.result_bank == ValType::Float`.
pub fn eval_batch_sum_float(
    lowered: &super::lower::LoweredF,
    columns: &[Column],
    n: usize,
    threshold: u32,
) -> Option<f64> {
    debug_assert_eq!(
        lowered.result_bank,
        super::lower::ValType::Float,
        "eval_batch_sum_float requires a float-valued lowering"
    );
    eval_batch_sum_f(lowered, columns, n, threshold).map(|bits| f64::from_bits(bits as u64))
}
/// Two-bank machine: the same three-address VM extended with a parallel
/// `fregs: [float; virt]` bank so a compiled trace can read `f64` context
/// columns and evaluate float predicates/arithmetic. Int opcodes address
/// `regs` exactly as the single-bank machine; float opcodes address `fregs`;
/// a float comparison crosses banks (float operands -> int `0`/`1`). Kept in
/// its own module because two `#[jit_interp]` mainloops in one module emit
/// colliding items. The single-bank int path above is untouched.
pub mod float_bank {
    use super::Code;
    use core::sync::atomic::AtomicUsize;

    /// Compile / guard-failure counters for the float path. Separate from the
    /// int-path statics so parallel tests don't race on a shared counter.
    pub static COMPILES: AtomicUsize = AtomicUsize::new(0);
    pub static GUARD_FAILS: AtomicUsize = AtomicUsize::new(0);
    /// Traces the tracer started and threw away. A loop that never appears in
    /// [`COMPILES`] is either aborting (counted here) or never reaching its
    /// merge point hot enough to be traced at all; the two have different
    /// causes, and only this counter tells them apart.
    pub static TRACE_ABORTS: AtomicUsize = AtomicUsize::new(0);

    use super::{
        OP_ADD, OP_ADD_OVF, OP_AND, OP_COL_LOAD, OP_COL_LOAD_F, OP_DIV, OP_DIV_CHK, OP_EQ, OP_F2I,
        OP_FADD, OP_FDIV, OP_FEQ, OP_FGE, OP_FGT, OP_FLE, OP_FLT, OP_FMOV, OP_FMUL, OP_FNE,
        OP_FNEG, OP_FSELECT, OP_FSUB, OP_GE, OP_GT, OP_I2F, OP_JUMP_IF_ABOVE, OP_LE, OP_LOAD_CONST,
        OP_LOAD_CONST_F, OP_LT, OP_MOD, OP_MOD_CHK, OP_MOV, OP_MUL, OP_MUL_OVF, OP_NE, OP_NEG,
        OP_NOT, OP_OR, OP_RETURN, OP_RETURN_F, OP_SELECT, OP_SUB, OP_SUB_OVF, OP_TRAP_STORE,
        OP_UADD_OVF, OP_UDIV, OP_ULE, OP_ULT, OP_UMOD, OP_UMUL_OVF, OP_USUB_OVF,
    };
    use core::sync::atomic::Ordering;

    /// Raw native-memory store intrinsic (`raw_store_i`) — see the int-bank
    /// [`super::majit_raw_store_i64`]. Duplicated here because the `#[jit_interp]`
    /// macro recognizes the call by name within the traced function's module.
    #[inline]
    fn majit_raw_store_i64(base: i64, ea: i64, val: i64) {
        // SAFETY: `base + ea` addresses the caller's live `i64` trap word, which
        // the batch driver keeps alive across the whole run.
        unsafe {
            core::ptr::write_unaligned((base as usize).wrapping_add(ea as usize) as *mut i64, val)
        }
    }

    #[inline]
    fn majit_raw_load_f(base: i64, ea: i64) -> f64 {
        // SAFETY: `base + ea` addresses element `ea/8` of a live `&[f64]`
        // column whose length the batch builder guarantees covers every row.
        unsafe {
            core::ptr::read_unaligned((base as usize).wrapping_add(ea as usize) as *const f64)
        }
    }

    /// Reinterpret a float's 64-bit pattern as an int — recognized by the
    /// `#[jit_interp]` proc macro as `convert_float_bytes_to_longlong`; at the
    /// interpreter tier this real fn runs. Used by the branchless float select.
    #[inline]
    fn majit_f64_to_bits(x: f64) -> i64 {
        x.to_bits() as i64
    }

    /// The inverse bitcast — `convert_longlong_bytes_to_float`.
    #[inline]
    fn majit_bits_to_f64(x: i64) -> f64 {
        f64::from_bits(x as u64)
    }

    /// Unsigned `<` on the int bank — recognized by the `#[jit_interp]` proc
    /// macro as `uint_lt`; at the interpreter tier this real fn runs. The int
    /// register file carries uint columns as their raw 64-bit pattern.
    #[inline]
    fn majit_uint_lt(a: i64, b: i64) -> i64 {
        ((a as u64) < (b as u64)) as i64
    }

    /// Unsigned `<=` — `uint_le`.
    #[inline]
    fn majit_uint_le(a: i64, b: i64) -> i64 {
        ((a as u64) <= (b as u64)) as i64
    }

    /// Unsigned `/` on the int bank. The macro recognizes the name and lowers it
    /// to the `int.udiv` oopspec residual call (`ll_uint_py_div`), NOT to a
    /// trace opcode: RPython deleted `UINT_FLOORDIV` from the resop set in 2016
    /// and routes unsigned division through that elidable call instead. A bare
    /// Rust `/` would lower to the SIGNED `int.py_div`, which disagrees with
    /// this fn for any operand above `2^63` — the interpreter and compiled tiers
    /// would then diverge silently.
    ///
    /// Caller must guarantee `b != 0`; the helper divides unconditionally.
    #[inline]
    fn majit_uint_div(a: i64, b: i64) -> i64 {
        ((a as u64) / (b as u64)) as i64
    }

    /// Unsigned `%` — `int.umod` / `ll_uint_py_mod`. See [`majit_uint_div`].
    #[inline]
    fn majit_uint_mod(a: i64, b: i64) -> i64 {
        ((a as u64) % (b as u64)) as i64
    }

    /// High 64 bits of the 128-bit unsigned product — the `uint_mul_high`
    /// resop, reached through the mainloop's `native_int_binops` alias rather
    /// than a hard-coded intrinsic name. It is zero exactly when `a * b` fits in
    /// a `u64`, which is the unsigned multiply-overflow test.
    ///
    /// The `u128` here is interpreter-tier only: the alias rewrites the CALL to
    /// the opcode, so the trace never looks inside this body (the backends emit
    /// `mulhi`/`umulh` for it).
    #[inline]
    fn majit_uint_mul_high(a: i64, b: i64) -> i64 {
        (((a as u64 as u128) * (b as u64 as u128)) >> 64) as u64 as i64
    }

    struct VmStateF {
        regs: Vec<i64>,
        fregs: Vec<f64>,
    }

    #[majit_macros::jit_interp(
        state = VmStateF,
        env = Code,
        greens = [pc, program],
        state_fields = {
            regs: [int; virt],
            fregs: [float; virt],
        },
        // `opcode_for_binop` has no unsigned spelling and `BindingKind` carries
        // no signedness, so the unsigned multiply-high resop is reached by
        // aliasing the helper call to the opcode.
        native_int_binops = { majit_uint_mul_high => UintMulHigh },
    )]
    fn run_mainloop_f(
        mut driver: &mut majit_metainterp::JitDriver<VmStateF>,
        program: &Code,
        init_regs: &[i64],
        num_fregs: usize,
    ) -> i64 {
        let mut pc: usize = 0;
        // Only the int bank is seeded: the column bases, the row count and the
        // trap address all live there, and no float ever enters from outside.
        let mut state = VmStateF {
            regs: init_regs.to_vec(),
            fregs: vec![0.0; num_fregs],
        };

        loop {
            jit_merge_point!();
            let opcode = program[pc];
            match opcode {
                OP_LOAD_CONST => {
                    state.regs[program[pc + 2] as usize] = program[pc + 1];
                    pc += 3;
                }
                OP_MOV => {
                    state.regs[program[pc + 2] as usize] = state.regs[program[pc + 1] as usize];
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
                OP_ADD_OVF => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    let t = program[pc + 4] as usize;
                    // Overflow-checked USER add (see the `OP_ADD_OVF` docs): the
                    // no-overflow path is a fused `int_add_jump_if_ovf`; the None
                    // arm runs only on the guard-exit resume and records the
                    // event, which invalidates the whole batch result.
                    state.regs[d] = match state.regs[a].checked_add(state.regs[b]) {
                        Some(s) => s,
                        None => {
                            state.regs[t] = 1;
                            state.regs[a].wrapping_add(state.regs[b])
                        }
                    };
                    pc += 5;
                }
                OP_SUB_OVF => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    let t = program[pc + 4] as usize;
                    state.regs[d] = match state.regs[a].checked_sub(state.regs[b]) {
                        Some(s) => s,
                        None => {
                            state.regs[t] = 1;
                            state.regs[a].wrapping_sub(state.regs[b])
                        }
                    };
                    pc += 5;
                }
                OP_MUL_OVF => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    let t = program[pc + 4] as usize;
                    state.regs[d] = match state.regs[a].checked_mul(state.regs[b]) {
                        Some(s) => s,
                        None => {
                            state.regs[t] = 1;
                            state.regs[a].wrapping_mul(state.regs[b])
                        }
                    };
                    pc += 5;
                }
                OP_TRAP_STORE => {
                    let addr = program[pc + 1] as usize;
                    let flag = program[pc + 2] as usize;
                    majit_raw_store_i64(state.regs[addr], 0, state.regs[flag]);
                    pc += 3;
                }
                OP_DIV => {
                    let a = state.regs[program[pc + 1] as usize];
                    let b = state.regs[program[pc + 2] as usize];
                    let d = program[pc + 3] as usize;
                    // Same toward-zero divide as the single-bank machine (see the
                    // `OP_DIV` arm in `run_mainloop`): majit lowers a bare `/` to
                    // Python floor division, so divide the magnitudes — where
                    // floor and truncation agree — and reapply the sign
                    // branchlessly. Inlined, not a helper call: a residual call
                    // in the mainloop aborts the trace.
                    let ma = a >> 63;
                    let mb = b >> 63;
                    let ua = (a ^ ma).wrapping_sub(ma);
                    let ub = (b ^ mb).wrapping_sub(mb);
                    let uq = ua / ub;
                    let s = ma ^ mb;
                    state.regs[d] = (uq ^ s).wrapping_sub(s);
                    pc += 4;
                }
                OP_MOD => {
                    let a = state.regs[program[pc + 1] as usize];
                    let b = state.regs[program[pc + 2] as usize];
                    let d = program[pc + 3] as usize;
                    // cel `%` takes the sign of the DIVIDEND (Rust/C remainder),
                    // while majit's floor `%` takes the sign of the divisor, so
                    // build the remainder from magnitudes and reapply the
                    // dividend's sign.
                    let ma = a >> 63;
                    let mb = b >> 63;
                    let ua = (a ^ ma).wrapping_sub(ma);
                    let ub = (b ^ mb).wrapping_sub(mb);
                    let ur = ua % ub;
                    state.regs[d] = (ur ^ ma).wrapping_sub(ma);
                    pc += 4;
                }
                OP_DIV_CHK => {
                    let a = state.regs[program[pc + 1] as usize];
                    let b = state.regs[program[pc + 2] as usize];
                    let d = program[pc + 3] as usize;
                    let t = program[pc + 4] as usize;
                    if b == 0 {
                        // `int_eq(rhs, 0) -> guard_false`. The walker raises
                        // `DivisionByZero`, so there is no answer to give.
                        state.regs[t] = 1;
                        state.regs[d] = 0;
                    } else {
                        // Truncating (toward-zero) division = cel's `/`. The
                        // magnitudes are divided UNSIGNED: `|i64::MIN|` is 2^63,
                        // which is not an i64, so a signed divide of the
                        // magnitude would come back negative and the sign
                        // reapplication would then flip it (`i64::MIN / 2`
                        // answering `+2^62`). Read as u64 the magnitude is exact.
                        let ma = a >> 63; // 0 or -1 (sign mask of a)
                        let mb = b >> 63;
                        // The magnitude subtraction WRAPS by construction: for
                        // `a == i64::MIN` it lands on 2^63, which is the case
                        // this whole arm exists to get right. `IntSub` is what
                        // both `-` and `wrapping_sub` trace to, so the compiled
                        // tier is unchanged and only the debug-build overflow
                        // panic goes away.
                        let ua = (a ^ ma).wrapping_sub(ma); // |a|, exact as a u64 bit pattern
                        let ub = (b ^ mb).wrapping_sub(mb); // |b|
                        let uq = majit_uint_div(ua, ub);
                        let s = ma ^ mb; // -1 iff signs differ
                        if ua < 0 && ub == 1 && s == 0 {
                            // `(lhs == INT_MIN) & (rhs == -1) -> guard_false`:
                            // `ua` reads negative only for `a == i64::MIN`, and
                            // with `|b| == 1` and matching signs that is the
                            // `INT_MIN / -1` corner, whose true quotient `2^63`
                            // is not an i64. `checked_div` reports it as
                            // `Overflow` and so must we.
                            state.regs[t] = 1;
                        }
                        state.regs[d] = (uq ^ s).wrapping_sub(s); // negate iff signs differ
                    }
                    pc += 5;
                }
                OP_MOD_CHK => {
                    let a = state.regs[program[pc + 1] as usize];
                    let b = state.regs[program[pc + 2] as usize];
                    let d = program[pc + 3] as usize;
                    let t = program[pc + 4] as usize;
                    if b == 0 {
                        state.regs[t] = 1;
                        state.regs[d] = 0;
                    } else {
                        // Truncating remainder = cel's `%` (sign of the
                        // dividend); unsigned magnitudes for the same
                        // `|i64::MIN|` reason as `OP_DIV_CHK`.
                        let ma = a >> 63;
                        let mb = b >> 63;
                        let ua = (a ^ ma).wrapping_sub(ma);
                        let ub = (b ^ mb).wrapping_sub(mb);
                        let ur = majit_uint_mod(ua, ub);
                        if ua < 0 && ub == 1 && (ma ^ mb) == 0 {
                            // `INT_MIN % -1` is mathematically 0, but
                            // `checked_rem` reports it as `Overflow` and the
                            // walker raises, so trap the same corner `/` does.
                            state.regs[t] = 1;
                        }
                        state.regs[d] = (ur ^ ma).wrapping_sub(ma); // reapply dividend's sign
                    }
                    pc += 5;
                }
                OP_UDIV => {
                    let a = state.regs[program[pc + 1] as usize];
                    let b = state.regs[program[pc + 2] as usize];
                    let d = program[pc + 3] as usize;
                    let t = program[pc + 4] as usize;
                    if b == 0 {
                        state.regs[t] = 1;
                        state.regs[d] = 0;
                    } else {
                        state.regs[d] = majit_uint_div(a, b);
                    }
                    pc += 5;
                }
                OP_UMOD => {
                    let a = state.regs[program[pc + 1] as usize];
                    let b = state.regs[program[pc + 2] as usize];
                    let d = program[pc + 3] as usize;
                    let t = program[pc + 4] as usize;
                    if b == 0 {
                        state.regs[t] = 1;
                        state.regs[d] = 0;
                    } else {
                        state.regs[d] = majit_uint_mod(a, b);
                    }
                    pc += 5;
                }
                OP_UADD_OVF => {
                    let a = state.regs[program[pc + 1] as usize];
                    let b = state.regs[program[pc + 2] as usize];
                    let d = program[pc + 3] as usize;
                    let t = program[pc + 4] as usize;
                    let s = a.wrapping_add(b);
                    // Carry out of bit 63: the wrapped sum lands strictly below
                    // either addend exactly when the true sum did not fit.
                    if majit_uint_lt(s, a) != 0 {
                        state.regs[t] = 1;
                    }
                    state.regs[d] = s;
                    pc += 5;
                }
                OP_USUB_OVF => {
                    let a = state.regs[program[pc + 1] as usize];
                    let b = state.regs[program[pc + 2] as usize];
                    let d = program[pc + 3] as usize;
                    let t = program[pc + 4] as usize;
                    // Borrow: an unsigned difference is representable iff the
                    // minuend is not below the subtrahend.
                    if majit_uint_lt(a, b) != 0 {
                        state.regs[t] = 1;
                    }
                    state.regs[d] = a.wrapping_sub(b);
                    pc += 5;
                }
                OP_UMUL_OVF => {
                    let a = state.regs[program[pc + 1] as usize];
                    let b = state.regs[program[pc + 2] as usize];
                    let d = program[pc + 3] as usize;
                    let t = program[pc + 4] as usize;
                    // The full product is 128 bits wide; it fits in a u64 iff
                    // the high word is zero.
                    if majit_uint_mul_high(a, b) != 0 {
                        state.regs[t] = 1;
                    }
                    state.regs[d] = a.wrapping_mul(b);
                    pc += 5;
                }
                OP_NEG => {
                    state.regs[program[pc + 2] as usize] = -state.regs[program[pc + 1] as usize];
                    pc += 3;
                }
                OP_GE => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = (state.regs[a] >= state.regs[b]) as i64;
                    pc += 4;
                }
                OP_GT => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = (state.regs[a] > state.regs[b]) as i64;
                    pc += 4;
                }
                OP_LE => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = (state.regs[a] <= state.regs[b]) as i64;
                    pc += 4;
                }
                OP_LT => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = (state.regs[a] < state.regs[b]) as i64;
                    pc += 4;
                }
                OP_ULT => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = majit_uint_lt(state.regs[a], state.regs[b]);
                    pc += 4;
                }
                OP_ULE => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = majit_uint_le(state.regs[a], state.regs[b]);
                    pc += 4;
                }
                OP_EQ => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = (state.regs[a] == state.regs[b]) as i64;
                    pc += 4;
                }
                OP_NE => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = (state.regs[a] != state.regs[b]) as i64;
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
                    state.regs[program[pc + 2] as usize] = 1 - state.regs[program[pc + 1] as usize];
                    pc += 3;
                }
                OP_SELECT => {
                    let c = state.regs[program[pc + 1] as usize];
                    let t = state.regs[program[pc + 2] as usize];
                    let f = state.regs[program[pc + 3] as usize];
                    state.regs[program[pc + 4] as usize] = f + c * (t - f);
                    pc += 5;
                }
                OP_FSELECT => {
                    // Branchless float select on an int/bool condition (0/1):
                    // reinterpret each arm as i64 bits, blend with a full 0/-1
                    // mask, reinterpret back. Bit-exact (an arithmetic float
                    // blend is not) and overflow-free (the blend is bitwise).
                    let c = state.regs[program[pc + 1] as usize];
                    let tb = majit_f64_to_bits(state.fregs[program[pc + 2] as usize]);
                    let fb = majit_f64_to_bits(state.fregs[program[pc + 3] as usize]);
                    let m = -c; // all-ones when c==1, zero when c==0
                    let nm = c - 1; // the complement mask (== !m for c in {0,1})
                    state.fregs[program[pc + 4] as usize] = majit_bits_to_f64((tb & m) | (fb & nm));
                    pc += 5;
                }
                OP_COL_LOAD => {
                    let base = state.regs[program[pc + 1] as usize];
                    let ea = state.regs[program[pc + 2] as usize];
                    state.regs[program[pc + 3] as usize] = super::majit_raw_load_i64(base, ea);
                    pc += 4;
                }
                OP_COL_LOAD_F => {
                    let base = state.regs[program[pc + 1] as usize];
                    let ea = state.regs[program[pc + 2] as usize];
                    state.fregs[program[pc + 3] as usize] = majit_raw_load_f(base, ea);
                    pc += 4;
                }
                OP_LOAD_CONST_F => {
                    state.fregs[program[pc + 2] as usize] = f64::from_bits(program[pc + 1] as u64);
                    pc += 3;
                }
                OP_I2F => {
                    state.fregs[program[pc + 2] as usize] =
                        state.regs[program[pc + 1] as usize] as f64;
                    pc += 3;
                }
                OP_F2I => {
                    state.regs[program[pc + 2] as usize] =
                        state.fregs[program[pc + 1] as usize] as i64;
                    pc += 3;
                }
                OP_FMOV => {
                    state.fregs[program[pc + 2] as usize] = state.fregs[program[pc + 1] as usize];
                    pc += 3;
                }
                OP_FADD => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.fregs[d] = state.fregs[a] + state.fregs[b];
                    pc += 4;
                }
                OP_FSUB => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.fregs[d] = state.fregs[a] - state.fregs[b];
                    pc += 4;
                }
                OP_FMUL => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.fregs[d] = state.fregs[a] * state.fregs[b];
                    pc += 4;
                }
                OP_FDIV => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.fregs[d] = state.fregs[a] / state.fregs[b];
                    pc += 4;
                }
                OP_FNEG => {
                    state.fregs[program[pc + 2] as usize] = -state.fregs[program[pc + 1] as usize];
                    pc += 3;
                }
                OP_FGE => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = (state.fregs[a] >= state.fregs[b]) as i64;
                    pc += 4;
                }
                OP_FGT => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = (state.fregs[a] > state.fregs[b]) as i64;
                    pc += 4;
                }
                OP_FLE => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = (state.fregs[a] <= state.fregs[b]) as i64;
                    pc += 4;
                }
                OP_FLT => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = (state.fregs[a] < state.fregs[b]) as i64;
                    pc += 4;
                }
                OP_FEQ => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = (state.fregs[a] == state.fregs[b]) as i64;
                    pc += 4;
                }
                OP_FNE => {
                    let a = program[pc + 1] as usize;
                    let b = program[pc + 2] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = (state.fregs[a] != state.fregs[b]) as i64;
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
                    return state.regs[program[pc + 1] as usize];
                }
                OP_RETURN_F => {
                    return state.fregs[program[pc + 1] as usize].to_bits() as i64;
                }
                _ => break,
            }
        }
        panic!("fell off end of code");
    }

    /// Reference two-bank interpreter — correctness oracle for the float path.
    pub fn clean_interp_f(program: &Code, num_regs: usize, num_fregs: usize) -> i64 {
        clean_interp_seeded_f(program, &vec![0i64; num_regs], num_fregs)
    }

    /// [`clean_interp_f`] over a caller-supplied initial int register bank, for
    /// programs whose data (column bases, row count, trap-word address) arrives
    /// in registers instead of as immediates.
    pub fn clean_interp_seeded_f(program: &Code, init_regs: &[i64], num_fregs: usize) -> i64 {
        let mut regs = init_regs.to_vec();
        let mut fregs = vec![0.0f64; num_fregs];
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
                // The reference tier mirrors the fused-ovf None arm (wrapping
                // value + trap flag) so it agrees with the JIT's overflow deopt
                // bit-for-bit.
                OP_ADD_OVF => {
                    let a = regs[program[pc + 1] as usize];
                    let b = regs[program[pc + 2] as usize];
                    if a.checked_add(b).is_none() {
                        regs[program[pc + 4] as usize] = 1;
                    }
                    regs[program[pc + 3] as usize] = a.wrapping_add(b);
                    pc += 5;
                }
                OP_SUB_OVF => {
                    let a = regs[program[pc + 1] as usize];
                    let b = regs[program[pc + 2] as usize];
                    if a.checked_sub(b).is_none() {
                        regs[program[pc + 4] as usize] = 1;
                    }
                    regs[program[pc + 3] as usize] = a.wrapping_sub(b);
                    pc += 5;
                }
                OP_MUL_OVF => {
                    let a = regs[program[pc + 1] as usize];
                    let b = regs[program[pc + 2] as usize];
                    if a.checked_mul(b).is_none() {
                        regs[program[pc + 4] as usize] = 1;
                    }
                    regs[program[pc + 3] as usize] = a.wrapping_mul(b);
                    pc += 5;
                }
                OP_TRAP_STORE => {
                    majit_raw_store_i64(
                        regs[program[pc + 1] as usize],
                        0,
                        regs[program[pc + 2] as usize],
                    );
                    pc += 3;
                }
                OP_DIV => {
                    // The reference tier is plain Rust, whose `/` and `%` already
                    // truncate toward zero — the semantics cel wants. The traced
                    // mainloop has to reconstruct that from magnitudes because
                    // majit lowers a bare `/` to floor division.
                    regs[program[pc + 3] as usize] =
                        regs[program[pc + 1] as usize] / regs[program[pc + 2] as usize];
                    pc += 4;
                }
                OP_MOD => {
                    regs[program[pc + 3] as usize] =
                        regs[program[pc + 1] as usize] % regs[program[pc + 2] as usize];
                    pc += 4;
                }
                // The guarded forms mirror the traced tier's two guards. The
                // reference tier can spell them directly — `checked_div` is
                // exactly the walker's own test — and where the guard fires the
                // result value is irrelevant: a set trap makes every tier return
                // `None`, so no value comparison survives it.
                OP_DIV_CHK => {
                    let a = regs[program[pc + 1] as usize];
                    let b = regs[program[pc + 2] as usize];
                    match a.checked_div(b) {
                        Some(q) => regs[program[pc + 3] as usize] = q,
                        None => {
                            regs[program[pc + 4] as usize] = 1;
                            regs[program[pc + 3] as usize] = 0;
                        }
                    }
                    pc += 5;
                }
                OP_MOD_CHK => {
                    let a = regs[program[pc + 1] as usize];
                    let b = regs[program[pc + 2] as usize];
                    match a.checked_rem(b) {
                        Some(r) => regs[program[pc + 3] as usize] = r,
                        None => {
                            regs[program[pc + 4] as usize] = 1;
                            regs[program[pc + 3] as usize] = 0;
                        }
                    }
                    pc += 5;
                }
                OP_UDIV => {
                    let a = regs[program[pc + 1] as usize] as u64;
                    let b = regs[program[pc + 2] as usize] as u64;
                    match a.checked_div(b) {
                        Some(q) => regs[program[pc + 3] as usize] = q as i64,
                        None => {
                            regs[program[pc + 4] as usize] = 1;
                            regs[program[pc + 3] as usize] = 0;
                        }
                    }
                    pc += 5;
                }
                OP_UMOD => {
                    let a = regs[program[pc + 1] as usize] as u64;
                    let b = regs[program[pc + 2] as usize] as u64;
                    match a.checked_rem(b) {
                        Some(r) => regs[program[pc + 3] as usize] = r as i64,
                        None => {
                            regs[program[pc + 4] as usize] = 1;
                            regs[program[pc + 3] as usize] = 0;
                        }
                    }
                    pc += 5;
                }
                // The unsigned overflow trio: the reference tier spells the
                // condition as `u64::checked_*`, which is the tree-walker's own
                // test, and always stores the wrapped value like the traced tier.
                OP_UADD_OVF => {
                    let a = regs[program[pc + 1] as usize] as u64;
                    let b = regs[program[pc + 2] as usize] as u64;
                    if a.checked_add(b).is_none() {
                        regs[program[pc + 4] as usize] = 1;
                    }
                    regs[program[pc + 3] as usize] = a.wrapping_add(b) as i64;
                    pc += 5;
                }
                OP_USUB_OVF => {
                    let a = regs[program[pc + 1] as usize] as u64;
                    let b = regs[program[pc + 2] as usize] as u64;
                    if a.checked_sub(b).is_none() {
                        regs[program[pc + 4] as usize] = 1;
                    }
                    regs[program[pc + 3] as usize] = a.wrapping_sub(b) as i64;
                    pc += 5;
                }
                OP_UMUL_OVF => {
                    let a = regs[program[pc + 1] as usize] as u64;
                    let b = regs[program[pc + 2] as usize] as u64;
                    if a.checked_mul(b).is_none() {
                        regs[program[pc + 4] as usize] = 1;
                    }
                    regs[program[pc + 3] as usize] = a.wrapping_mul(b) as i64;
                    pc += 5;
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
                OP_ULT => {
                    regs[program[pc + 3] as usize] = majit_uint_lt(
                        regs[program[pc + 1] as usize],
                        regs[program[pc + 2] as usize],
                    );
                    pc += 4;
                }
                OP_ULE => {
                    regs[program[pc + 3] as usize] = majit_uint_le(
                        regs[program[pc + 1] as usize],
                        regs[program[pc + 2] as usize],
                    );
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
                OP_FSELECT => {
                    let c = regs[program[pc + 1] as usize];
                    let tb = majit_f64_to_bits(fregs[program[pc + 2] as usize]);
                    let fb = majit_f64_to_bits(fregs[program[pc + 3] as usize]);
                    let m = -c;
                    let nm = c - 1;
                    fregs[program[pc + 4] as usize] = majit_bits_to_f64((tb & m) | (fb & nm));
                    pc += 5;
                }
                OP_COL_LOAD => {
                    let base = regs[program[pc + 1] as usize];
                    let ea = regs[program[pc + 2] as usize];
                    regs[program[pc + 3] as usize] = super::majit_raw_load_i64(base, ea);
                    pc += 4;
                }
                OP_COL_LOAD_F => {
                    let base = regs[program[pc + 1] as usize];
                    let ea = regs[program[pc + 2] as usize];
                    fregs[program[pc + 3] as usize] = majit_raw_load_f(base, ea);
                    pc += 4;
                }
                OP_LOAD_CONST_F => {
                    fregs[program[pc + 2] as usize] = f64::from_bits(program[pc + 1] as u64);
                    pc += 3;
                }
                OP_I2F => {
                    fregs[program[pc + 2] as usize] = regs[program[pc + 1] as usize] as f64;
                    pc += 3;
                }
                OP_F2I => {
                    regs[program[pc + 2] as usize] = fregs[program[pc + 1] as usize] as i64;
                    pc += 3;
                }
                OP_FMOV => {
                    fregs[program[pc + 2] as usize] = fregs[program[pc + 1] as usize];
                    pc += 3;
                }
                OP_FADD => {
                    fregs[program[pc + 3] as usize] =
                        fregs[program[pc + 1] as usize] + fregs[program[pc + 2] as usize];
                    pc += 4;
                }
                OP_FSUB => {
                    fregs[program[pc + 3] as usize] =
                        fregs[program[pc + 1] as usize] - fregs[program[pc + 2] as usize];
                    pc += 4;
                }
                OP_FMUL => {
                    fregs[program[pc + 3] as usize] =
                        fregs[program[pc + 1] as usize] * fregs[program[pc + 2] as usize];
                    pc += 4;
                }
                OP_FDIV => {
                    fregs[program[pc + 3] as usize] =
                        fregs[program[pc + 1] as usize] / fregs[program[pc + 2] as usize];
                    pc += 4;
                }
                OP_FNEG => {
                    fregs[program[pc + 2] as usize] = -fregs[program[pc + 1] as usize];
                    pc += 3;
                }
                OP_FGE => {
                    regs[program[pc + 3] as usize] =
                        (fregs[program[pc + 1] as usize] >= fregs[program[pc + 2] as usize]) as i64;
                    pc += 4;
                }
                OP_FGT => {
                    regs[program[pc + 3] as usize] =
                        (fregs[program[pc + 1] as usize] > fregs[program[pc + 2] as usize]) as i64;
                    pc += 4;
                }
                OP_FLE => {
                    regs[program[pc + 3] as usize] =
                        (fregs[program[pc + 1] as usize] <= fregs[program[pc + 2] as usize]) as i64;
                    pc += 4;
                }
                OP_FLT => {
                    regs[program[pc + 3] as usize] =
                        (fregs[program[pc + 1] as usize] < fregs[program[pc + 2] as usize]) as i64;
                    pc += 4;
                }
                OP_FEQ => {
                    regs[program[pc + 3] as usize] =
                        (fregs[program[pc + 1] as usize] == fregs[program[pc + 2] as usize]) as i64;
                    pc += 4;
                }
                OP_FNE => {
                    regs[program[pc + 3] as usize] =
                        (fregs[program[pc + 1] as usize] != fregs[program[pc + 2] as usize]) as i64;
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
                OP_RETURN_F => return fregs[program[pc + 1] as usize].to_bits() as i64,
                _ => panic!("bad op {}", program[pc]),
            }
        }
    }

    /// Build a driver for one state shape and install its canonical liveness
    /// once. The install is program-independent (the generated `build_meta`
    /// ignores its args), so a caller may keep one driver per
    /// `(num_regs, num_fregs)` shape and run every program of that shape on it.
    fn new_driver_f(
        threshold: u32,
        program: &Code,
        num_regs: usize,
        num_fregs: usize,
    ) -> majit_metainterp::JitDriver<VmStateF> {
        // No quasi-immutable state exists here (a fixed batch program over plain
        // integer and float reds), so skip the periodic loop-invalidation timer:
        // it has nothing to invalidate and would only force a persistent driver
        // to re-trace what it already compiled (`jitdriver.rs with_options`).
        let mut driver: majit_metainterp::JitDriver<VmStateF> =
            majit_metainterp::JitDriver::with_options(threshold, false);
        driver.set_on_compile_loop(|_green_key, _ops_before, _ops_after| {
            COMPILES.fetch_add(1, Ordering::Relaxed);
        });
        driver.set_on_guard_failure(|_green_key, _a, _b| {
            GUARD_FAILS.fetch_add(1, Ordering::Relaxed);
        });
        driver.set_on_trace_abort(|_green_key, _permanent| {
            TRACE_ABORTS.fetch_add(1, Ordering::Relaxed);
        });
        let seed = VmStateF {
            regs: vec![0; num_regs],
            fregs: vec![0.0; num_fregs],
        };
        {
            use majit_metainterp::JitState as _;
            seed.build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        driver
    }

    /// Run the two-bank mainloop on a one-off driver with every register zeroed,
    /// so the program itself supplies all of its inputs. `threshold == u32::MAX`
    /// gives the interpreter tier; a small value enables JIT compilation.
    pub fn run_jit_f(program: &Code, num_regs: usize, num_fregs: usize, threshold: u32) -> i64 {
        run_jit_seeded_f(program, &vec![0i64; num_regs], num_fregs, threshold)
    }

    /// [`run_jit_f`] over a caller-supplied initial int register bank. The
    /// seeded values are plain reds — the trace reads them as loop-invariant
    /// inputs, not as constants, so one compiled loop serves every batch.
    pub fn run_jit_seeded_f(
        program: &Code,
        init_regs: &[i64],
        num_fregs: usize,
        threshold: u32,
    ) -> i64 {
        let mut driver = new_driver_f(threshold, program, init_regs.len(), num_fregs);
        run_mainloop_f(&mut driver, program, init_regs, num_fregs)
    }

    std::thread_local! {
        /// Batch programs interned by their own words.
        ///
        /// The `#[jit_interp]` green key is the program **pointer** plus pc
        /// (`trace_ctx.rs` `green_key_raw`), so a compiled loop is only reused
        /// when the next batch runs the same allocation. A batch program's words
        /// depend only on the expression's shape, so every batch of one
        /// expression builds identical words and shares this entry. Entries are
        /// never removed, which is what keeps the address stable.
        static PROGRAMS: core::cell::RefCell<std::collections::HashSet<std::rc::Rc<[i64]>>> =
            core::cell::RefCell::new(std::collections::HashSet::new());

        /// Drivers kept across calls, keyed by the state shape they were built
        /// for and the threshold they compile at.
        ///
        /// The compiled loop lives in the driver, so a driver per call is a
        /// recompile per call. RPython keeps it on the greens-keyed JitCell
        /// instead (`warmstate.py:157-199` `wref_procedure_token`, held for
        /// `max_age` generations by `memmgr.py:23-69`) and never recompiles per
        /// invocation. The threshold is part of the key so the interpreter tier
        /// (`u32::MAX`) can never pick up the JIT tier's compiled loop.
        static DRIVERS: core::cell::RefCell<
            std::collections::HashMap<(usize, usize, u32), majit_metainterp::JitDriver<VmStateF>>,
        > = core::cell::RefCell::new(std::collections::HashMap::new());
    }

    /// Intern a batch program's words, returning a handle whose address stays
    /// put for the rest of the process so the green key keyed on it does too.
    pub fn intern_program(code: Vec<i64>) -> std::rc::Rc<[i64]> {
        PROGRAMS.with(|p| {
            let mut p = p.borrow_mut();
            if let Some(interned) = p.get(&code[..]) {
                return interned.clone();
            }
            let interned: std::rc::Rc<[i64]> = code.into();
            p.insert(interned.clone());
            interned
        })
    }

    /// Drop this thread's interned programs and persistent drivers, so the next
    /// batch traces and compiles from cold.
    ///
    /// The two caches are cleared together and must always be: a driver's
    /// compiled loops are keyed on program **addresses**, so keeping the drivers
    /// while freeing the programs would let a freshly interned program land on a
    /// freed address and pick up another program's compiled loop.
    pub fn reset_persistent_state() {
        DRIVERS.with(|d| d.borrow_mut().clear());
        PROGRAMS.with(|p| p.borrow_mut().clear());
    }

    /// [`run_jit_seeded_f`] on a driver that outlives the call, so a program
    /// already compiled by an earlier call runs compiled from its first row.
    pub fn run_jit_persistent_f(
        program: &Code,
        init_regs: &[i64],
        num_fregs: usize,
        threshold: u32,
    ) -> i64 {
        let key = (init_regs.len(), num_fregs, threshold);
        // Take the driver out of the map for the duration of the run instead of
        // holding the borrow across it: a re-entrant call then builds its own
        // driver rather than panicking on the `RefCell`.
        let mut driver = DRIVERS
            .with(|d| d.borrow_mut().remove(&key))
            .unwrap_or_else(|| new_driver_f(threshold, program, init_regs.len(), num_fregs));
        // The store majit decodes guard and resume metadata through is one slot
        // per thread, written when a driver registers its dispatch jitcode. We
        // keep a driver per shape, so aim it back at this one before it can
        // compile anything: another shape's store decodes at the same pcs and
        // returns a mistyped frame count rather than failing.
        driver.republish_state_field_fvc();
        let result = run_mainloop_f(&mut driver, program, init_regs, num_fregs);
        DRIVERS.with(|d| {
            d.borrow_mut().insert(key, driver);
        });
        result
    }
}
