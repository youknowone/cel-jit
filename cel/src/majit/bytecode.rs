//! Flat `i64`-word bytecode for the majit-traceable CEL subset, plus the majit
//! `#[jit_interp]` mainloop that evaluates it ([`float_bank::run_mainloop_f`])
//! and a plain-`match` reference interpreter ([`float_bank::clean_interp_seeded_f`])
//! used as the correctness oracle and the honest perf baseline.
//!
//! The instruction set is a three-address register machine over two banks: an
//! `i64` bank (`regs`) carrying ints, bools as `0`/`1`, `uint` as a raw bit
//! pattern, strings as ranked ids and temporals as nanos, and a parallel
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

/// Widen a `uint` to `double`. The unsigned twin of [`OP_I2F`]: the source word
/// is a uint's raw bit pattern, so it converts through `u64`, and reading it as
/// `i64` would turn every value above `i64::MAX` negative. Comparison is the
/// only place a uint meets a double — `1u == 1.0` is `true`, `1u < 2.0` is
/// `true` — and the widening is the same `as f64` the tree-walker performs.
pub const OP_U2F: i64 = 52; // [src, fdst]   fregs[fdst] = (regs[src] as u64) as f64

/// Narrow a `double` to a `uint`. The unsigned twin of [`OP_F2I`], and total
/// for the same reason: `as u64` truncates toward zero and SATURATES, with a
/// negative double clamping to `0` and NaN mapping to `0`. It saturates at
/// DIFFERENT bounds than `as i64`, which is why it cannot reuse `OP_F2I`:
/// `uint(-1.5)` is `0u` where `int(-1.5)` is `-1`.
pub const OP_F2U: i64 = 53; // [fsrc, dst]   regs[dst] = (fregs[fsrc] as u64) as i64

/// The write side of [`OP_COL_LOAD`], and what a per-row output loop ends each
/// iteration with instead of accumulating: `*(regs[base] + regs[ea]) = regs[src]`.
///
/// The base is loop-invariant in a register exactly as a read column's is, and
/// the effective address is the same `i * 8` the reads use, so a row's output
/// lands at the row's index.
pub const OP_COL_STORE: i64 = 54; // [base, ea, src]

/// [`OP_COL_STORE`] from the float bank, storing `fregs[fsrc]`'s 64-BIT PATTERN.
///
/// Writing the bits rather than the `f64` keeps the output buffer one `i64`
/// buffer for every result bank, and costs nothing: it reuses the two
/// intrinsics the tracer already recognizes (`convert_float_bytes_to_longlong`
/// then `raw_store_i`) instead of adding a float store to the machine.
pub const OP_COL_STORE_F: i64 = 55; // [base, ea, fsrc]

/// [`OP_COL_LOAD`] from a ONE-BYTE unsigned column, for a `bool` slot backed by
/// the caller's own `&[bool]`.
///
/// The `ea` is the ROW INDEX itself, not `i * 8`: a byte column's stride is one,
/// so the row counter already holds the effective address and the machine emits
/// no scaling op for it at all.
///
/// The result is still an `i64` `0`/`1` in the int register file — the width
/// lives on the load's descr (`majit_raw_load_u8`), not on the value, which is
/// upstream's arrangement: `jtransform.py:1165-1171` builds the descr from the
/// LOADED type and `history.py:45-63 getkind` calls every sub-word primitive an
/// `int`, so nothing downstream of the load has to know the column was narrow.
pub const OP_COL_LOAD_B: i64 = 56; // [base, ea, dst]

/// [`OP_MUL`] and [`OP_ADD`] against an IMMEDIATE carried in the word stream.
///
/// The second operand is `program[pc + 2]` itself, not a register holding it.
/// `program` is a green of the mainloop's merge point, so the traced form is
/// `int_mul(reg, ConstInt(imm))` — a constant the optimizer can fold and, more
/// to the point, the one shape `dependency.py:896-948` recognizes when it
/// builds an `IndexVar` for a memory reference. The register form
/// (`OP_MUL r_i, r_stride`) reads a value stored before the merge point, so
/// inside the loop it is an opaque input argument: no `IndexVar`, no adjacent
/// memory refs, and the loop vectorizer declines.
///
/// Loop induction and address scaling are the two places a batch loop needs a
/// literal, and both are literals of the SHAPE, not of the batch — they belong
/// in the bytecode the same way a register index does.
pub const OP_MUL_IMM: i64 = 57; // [a, imm, dst]           regs[dst] = regs[a] * imm
/// [`OP_MUL_IMM`] for addition — the loop's `i += 1` step.
pub const OP_ADD_IMM: i64 = 58; // [a, imm, dst]           regs[dst] = regs[a] + imm

/// What one word after an opcode means, for a consumer that walks a program
/// without running it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    /// A literal in the word stream — a constant, a byte stride, a scale.
    Imm,
    /// An int-bank register the instruction reads.
    Int,
    /// An int-bank register the instruction writes.
    IntOut,
    /// An int-bank register only the TRAP path writes, leaving it as it was on
    /// every other path — so its old value survives the instruction and it is
    /// read and written both.
    IntTrap,
    /// A float-bank register the instruction reads.
    Float,
    /// A float-bank register the instruction writes.
    FloatOut,
    /// A program address, in words from the start of the program.
    Target,
}

use Operand::{Float, FloatOut, Imm, Int, IntOut, IntTrap, Target};

/// Every opcode's operand roles, indexed by the opcode itself. An entry's
/// length is the instruction's width less its opcode word, so this table also
/// says how far the next instruction is.
///
/// One table rather than a walk built into each consumer: a consumer that
/// pattern-matches the opcodes it happens to know silently mis-decodes the ones
/// it does not — and mis-decoding a width desynchronizes the whole rest of the
/// program, not just the instruction that was missed.
///
/// The length is written in terms of the last opcode rather than as a literal,
/// so an opcode added past it without a row here is a compile error on this
/// array instead of an out-of-bounds index the first time that opcode is
/// decoded.
pub const OPERANDS: [&[Operand]; OP_ADD_IMM as usize + 1] = [
    &[Imm, IntOut],                 // 0  LOAD_CONST
    &[Int, IntOut],                 // 1  MOV
    &[Int, Int, IntOut],            // 2  ADD
    &[Int, Int, IntOut],            // 3  SUB
    &[Int, Int, IntOut],            // 4  MUL
    &[Int, IntOut],                 // 5  NEG
    &[Int, Int, IntOut],            // 6  GE
    &[Int, Int, IntOut],            // 7  GT
    &[Int, Int, IntOut],            // 8  LE
    &[Int, Int, IntOut],            // 9  LT
    &[Int, Int, IntOut],            // 10 EQ
    &[Int, Int, IntOut],            // 11 NE
    &[Int, Int, IntOut],            // 12 AND
    &[Int, Int, IntOut],            // 13 OR
    &[Int, IntOut],                 // 14 NOT
    &[Int, Int, Int, IntOut],       // 15 SELECT
    &[Int, Int, Target],            // 16 JUMP_IF_ABOVE
    &[Int],                         // 17 RETURN
    &[Int, Int, IntOut],            // 18 DIV
    &[Int, Int, IntOut],            // 19 MOD
    &[Int, Int, IntOut],            // 20 COL_LOAD
    &[Int, Int, FloatOut],          // 21 COL_LOAD_F
    &[Imm, FloatOut],               // 22 LOAD_CONST_F
    &[Float, FloatOut],             // 23 FMOV
    &[Float, Float, FloatOut],      // 24 FADD
    &[Float, Float, FloatOut],      // 25 FSUB
    &[Float, Float, FloatOut],      // 26 FMUL
    &[Float, Float, FloatOut],      // 27 FDIV
    &[Float, FloatOut],             // 28 FNEG
    &[Float, Float, IntOut],        // 29 FGE
    &[Float, Float, IntOut],        // 30 FGT
    &[Float, Float, IntOut],        // 31 FLE
    &[Float, Float, IntOut],        // 32 FLT
    &[Float, Float, IntOut],        // 33 FEQ
    &[Float, Float, IntOut],        // 34 FNE
    &[Int, FloatOut],               // 35 I2F
    &[Float],                       // 36 RETURN_F
    &[Int, Float, Float, FloatOut], // 37 FSELECT
    &[Int, Int, IntOut],            // 38 ULT
    &[Int, Int, IntOut],            // 39 ULE
    &[Int, Int, IntOut, IntTrap],   // 40 ADD_OVF
    &[Int, Int, IntOut, IntTrap],   // 41 SUB_OVF
    &[Int, Int, IntOut, IntTrap],   // 42 MUL_OVF
    &[Int, Int],                    // 43 TRAP_STORE
    &[Float, IntOut],               // 44 F2I
    &[Int, Int, IntOut, IntTrap],   // 45 DIV_CHK
    &[Int, Int, IntOut, IntTrap],   // 46 MOD_CHK
    &[Int, Int, IntOut, IntTrap],   // 47 UDIV
    &[Int, Int, IntOut, IntTrap],   // 48 UMOD
    &[Int, Int, IntOut, IntTrap],   // 49 UADD_OVF
    &[Int, Int, IntOut, IntTrap],   // 50 USUB_OVF
    &[Int, Int, IntOut, IntTrap],   // 51 UMUL_OVF
    &[Int, FloatOut],               // 52 U2F
    &[Float, IntOut],               // 53 F2U
    &[Int, Int, Int],               // 54 COL_STORE
    &[Int, Int, Float],             // 55 COL_STORE_F
    &[Int, Int, IntOut],            // 56 COL_LOAD_B
    &[Int, Imm, IntOut],            // 57 MUL_IMM
    &[Int, Imm, IntOut],            // 58 ADD_IMM
];

use super::inline::{Inline, INLINE_SLOTS};
use super::lower::BatchReduce;

/// One input column for the two-bank batch evaluator: an `i64` column for an
/// int/bool slot, an `f64` column for a `double` slot, or raw strings for a
/// `string` slot. Its base pointer (an `i64` regardless of bank) is what a
/// compiled trace reads per row.
#[derive(Debug, Clone, Copy)]
pub enum Column<'a> {
    Int(&'a [i64]),
    Float(&'a [f64]),
    /// A `bool` column, read where the caller keeps it: ONE BYTE per row.
    ///
    /// The register file is still `i64`, and the trace still holds a `0`/`1`
    /// int — but the load that produces it is a 1-byte unsigned one
    /// ([`OP_COL_LOAD_B`]), so the column is not copied into an `i64` buffer to
    /// be read. That copy is what a batch pays twice for: once to build it, and
    /// again on every run, streaming eight bytes where one carries the value.
    ///
    /// `jtransform.py:1165-1171 rewrite_op_raw_load` takes the descr from the
    /// LOADED type (`arraydescrof(rffi.CArray(T))`), and `history.py:45-63
    /// getkind` calls every primitive no wider than a word an `int` — so a
    /// 1-byte load is a `raw_load_i` whose descr says 1, and the backend widens
    /// it into the register. Nothing about the narrow column reaches the trace
    /// as a different KIND of value.
    Bool(&'a [bool]),
    /// A `string` column, handed over as its raw content.
    ///
    /// The machine runs on `i64` ids, but **which** ids is [`prepare_batch`]'s
    /// to decide, not the caller's: it ranks the batch's distinct strings (see
    /// [`StrDict`]) so that one order covers the column values and the
    /// expression's own literals alike. A caller cannot pre-encode, because a
    /// second ranking would not agree with the one the literals came from.
    Str(&'a [String]),
}

impl Column<'_> {
    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            Column::Int(c) => c.len(),
            Column::Float(c) => c.len(),
            Column::Bool(c) => c.len(),
            Column::Str(c) => c.len(),
        }
    }

    /// True if the column is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn matches(&self, ty: super::lower::ValType) -> bool {
        use super::lower::ValType;
        // A `uint` slot is backed by an int-bit column (the int register file
        // carries the raw 64-bit pattern) and a `string` slot takes the strings
        // themselves.
        //
        // A `bool` slot takes a `bool` column and NOTHING ELSE. The lowering
        // reads it with a one-byte load at the row index, so an `i64` column of
        // `0`/`1` under a `bool` slot would be read at one eighth of its stride
        // and answer with whichever byte happened to be there. That is a silent
        // wrong answer, which is why the pairing is checked rather than widened.
        matches!(
            (self, ty),
            (
                Column::Int(_),
                ValType::Int | ValType::UInt | ValType::Timestamp | ValType::Duration
            ) | (Column::Bool(_), ValType::Bool)
                | (Column::Float(_), ValType::Float)
                | (Column::Str(_), ValType::Str)
        )
    }
}

/// A batch's strings encoded as `i64` ids, injective by construction — so
/// unlike a content hash there is no collision to check for and no
/// data-dependent bail.
///
/// The ids are **order-preserving** — id `k` is the `k`-th smallest distinct
/// string — only when the program asks for it ([`super::lower::LoweredF::orders_strings`]).
/// That is what makes an id compare a content compare for ORDERING and not only
/// for equality, and it is why string `<` no longer bails. It is not free:
/// ordering the ids means sorting the distinct set at every bind, which is
/// `O(d log d)` STRING comparisons — and a batch whose values are mostly
/// distinct has `d ≈ rows`, where that sort is over half the encoding's cost.
/// Equality reads no order, so a program that only compares strings for `==`
/// and `!=` takes ids in arrival order and pays none of it.
///
/// Built over every string the batch carries AND every literal the expression
/// mentions, so column-vs-column and column-vs-literal share one assignment.
struct StrDict<'s> {
    rank: std::collections::HashMap<&'s str, i64>,
}

impl<'s> StrDict<'s> {
    /// The distinct strings in id order, which is what a predicate table is
    /// indexed by. In ID order and not in lexicographic order: the two coincide
    /// only on the ordered arm, and both consumers index by id.
    fn sorted(&self) -> Vec<&'s str> {
        let mut out: Vec<(i64, &'s str)> = self.rank.iter().map(|(s, &k)| (k, *s)).collect();
        out.sort_unstable();
        out.into_iter().map(|(_, s)| s).collect()
    }

    /// Rank every string the batch can be asked about — the `Column::Str`
    /// values and the expression's own literals — and encode each string column
    /// into the ids the machine runs on.
    ///
    /// One call rather than a build and then an encode, because the UNORDERED
    /// arm can do both in a single pass over the rows: a string's id is known
    /// the moment it is first seen, so the same lookup that files it also
    /// writes it. An ordered assignment cannot — no string has an id until
    /// every string has been seen and sorted — so that arm still reads the rows
    /// twice, and one shape covering both is what keeps the two from being
    /// spelled differently at the two call sites.
    ///
    /// `ordered` is [`super::lower::LoweredF::needs_ordered_str_ids`]: pass it,
    /// do not guess it. Passing `true` when the program never orders is only
    /// slow; passing `false` when it does is a WRONG ANSWER, because the
    /// ordering lowered to a signed compare of the two ids.
    fn build_columns(
        columns: &'s [Column<'s>],
        seeds: impl Iterator<Item = &'s str>,
        ordered: bool,
    ) -> (Self, Vec<Box<[i64]>>) {
        let str_cols = || {
            columns.iter().filter_map(|c| match c {
                Column::Str(s) => Some(*s),
                _ => None,
            })
        };
        // One entry per input string. Exact where the batch's values are
        // distinct, which is where the table would otherwise grow through every
        // power of two on the way up; an over-reserve bounded by the input
        // where they are not, which measured as no slower and is smaller than
        // the `Vec<String>` it is reading.
        let capacity = str_cols().map(<[String]>::len).sum::<usize>();
        if !ordered {
            let mut rank: std::collections::HashMap<&'s str, i64> =
                std::collections::HashMap::with_capacity(capacity);
            let mut ids = Vec::with_capacity(str_cols().count());
            for col in str_cols() {
                let mut out: Vec<i64> = Vec::with_capacity(col.len());
                for s in col {
                    let next = rank.len() as i64;
                    out.push(*rank.entry(s.as_str()).or_insert(next));
                }
                ids.push(out.into_boxed_slice());
            }
            // After the columns, so a literal the columns already carry keeps
            // the id its rows gave it. Which of the two files it first cannot
            // matter — the assignment is injective either way — but ranking the
            // seeds first would push every column's ids up by the number of
            // literals, for no reason a reader could see.
            for s in seeds {
                let next = rank.len() as i64;
                rank.entry(s).or_insert(next);
            }
            return (StrDict { rank }, ids);
        }
        // Dedup BEFORE sorting, so the sort is over the DISTINCT set rather
        // than the rows. That is the whole of the saving where a batch is many
        // rows over few values; where the values are mostly distinct there is
        // nothing to dedup and the sort is the encoding's largest single cost,
        // which is what the arm above exists to skip.
        let mut distinct: std::collections::HashSet<&'s str> =
            std::collections::HashSet::with_capacity(capacity);
        distinct.extend(str_cols().flatten().map(String::as_str));
        distinct.extend(seeds);
        let mut distinct: Vec<&'s str> = distinct.into_iter().collect();
        distinct.sort_unstable();
        let dict = StrDict {
            rank: distinct
                .into_iter()
                .enumerate()
                .map(|(k, s)| (s, k as i64))
                .collect(),
        };
        let ids = str_cols().map(|c| dict.encode(c)).collect();
        (dict, ids)
    }

    /// The id of a string that was in the build set.
    fn id(&self, s: &str) -> i64 {
        self.rank[s]
    }

    fn encode(&self, col: &[String]) -> Box<[i64]> {
        col.iter().map(|s| self.id(s)).collect()
    }
}

/// One batch of rows, prepared: a handle on the program words, the seeded
/// initial register bank (column bases, row count, trap address) and the trap
/// word itself. Everything that depends on the expression and the data but not
/// on the tier, built once so a caller running the same batch more than once —
/// a benchmark sweeping tiers, a cross-tier check — pays for it once.
///
/// Holds raw base pointers into `columns`, so the borrow is carried in `'a`.
pub struct BatchRun<'a> {
    /// Shared with the [`super::lower::LoweredF`] that owns it, so the address
    /// the green key is taken over is the program's own and outlives this run.
    code: std::sync::Arc<[i64]>,
    init_regs: Vec<i64>,
    num_float_regs: usize,
    /// The word the program publishes the overflow flag to. Boxed so its
    /// address is stable, and never aliased by a reference while the program
    /// writes it through the raw pointer seeded into `init_regs`.
    trap: Box<i64>,
    rows: usize,
    /// Id columns materialized from the caller's [`Column::Str`] buffers.
    /// `init_regs` holds raw pointers into these, so they are kept alive here
    /// for as long as the run is. Each is separately heap-allocated, so moving
    /// the `BatchRun` moves the box pointers and not the buffers they address.
    ///
    /// Never read after preparation: a `string` projection reads its ids
    /// there, and nothing else consults them.
    str_ids: Vec<Box<[i64]>>,
    /// Whether this run is a projection ([`super::lower::LoweredF::is_row_projection`],
    /// bound [`BatchReduce::PerRow`]), whose output buffer
    /// [`prepare_batch_reduce`] has already filled.
    projection: bool,
    /// One `i64` per row, under [`BatchReduce::PerRow`]: where the loop stores
    /// each row's result. Boxed for the same reason `trap` is — `init_regs`
    /// holds a raw pointer to it, which must survive the run moving.
    out: Option<Box<[i64]>>,
    /// One buffer per output field, for a LIST-valued result: the flat element
    /// stream the loop wrote, which `out` indexes into by running length.
    list_out: Vec<Box<[i64]>>,
    /// The batch's distinct strings in rank order, so a `string`-banked output
    /// id can be read back as the string it stands for. Kept only where the
    /// result is a string; ranking is otherwise write-only.
    ///
    /// OWNED, not borrowed from the columns: the set includes the expression's
    /// own string literals, which live in the lowering, and a decode table that
    /// borrowed from both would tie the run's lifetime to the program's.
    distinct: Vec<String>,
    /// Scratch register banks, kept here so a batch run repeatedly does not
    /// re-allocate them. Only the clean tier reads them: the tracing tiers go
    /// through [`float_bank::run_mainloop_f`], whose banks are built inside the
    /// traced portal and cannot be handed in without adding a parameter to it —
    /// and its greens bind by position, so a new parameter is not a local
    /// change.
    banks: float_bank::Banks,
    columns: core::marker::PhantomData<&'a ()>,
}

impl<'a> BatchRun<'a> {
    /// Run the prepared batch with `run`, which selects the tier. `None` means a
    /// row trapped (`int` overflow, division by zero), where the tree-walker
    /// raises and no sum is the right answer.
    /// The program reaches `run` as the `Arc` this batch holds, not as a slice:
    /// a tier that files a compiled loop under its address has to be able to
    /// hold the words, and only the owning handle lets it.
    ///
    /// The banks come through as a fourth argument rather than being built by
    /// the tier, so a tier that can take them — the clean one — allocates
    /// nothing per run. A tier that cannot is free to ignore them.
    pub fn run(
        &mut self,
        run: impl FnOnce(&std::sync::Arc<Code>, &[i64], usize, &mut float_bank::Banks) -> i64,
    ) -> Option<i64> {
        // A zero-row batch reduces to the accumulator's initial value without
        // entering the loop, and its column bases point at nothing.
        if self.rows == 0 {
            return Some(0);
        }
        *self.trap = 0;
        let result = run(
            &self.code,
            &self.init_regs,
            self.num_float_regs,
            &mut self.banks,
        );
        if *self.trap != 0 {
            return None;
        }
        Some(result)
    }

    /// Whether this run is a PROJECTION, whose output buffer already holds the
    /// answer.
    ///
    /// A projection's row loop reads one column element and stores it, so the
    /// buffer it produces is the input column in the output's encoding. The
    /// columns are borrowed for as long as the run is and cannot change under
    /// it, which makes that buffer a function of data already fixed when the
    /// batch was prepared -- so [`prepare_batch_reduce`] writes it there, once,
    /// and a call reads what is already in it rather than copying the column
    /// again. For a batch of any height that is the same answer without the
    /// five instructions per row that spell the copy out, and without a pass
    /// over the column per call.
    ///
    /// Nothing here can fail: a projection does no arithmetic, so the trap flag
    /// a loop's epilogue would publish stays the clear one the run starts at.
    /// That is also why running the SAME program on a tracing tier leaves the
    /// buffer intact -- the loop writes the same values back over it.
    ///
    /// `false` means this run is not a projection and the caller must run the
    /// program. That is the only thing it can mean.
    ///
    /// Not part of the public surface: which runs take the shortcut is the
    /// batch layer's decision, made once in [`super::batch::BoundBatch`], and a
    /// caller reaching past it could take it on a tier whose counters are about
    /// what the program costs.
    pub(super) fn project(&self) -> bool {
        self.projection
    }

    /// The per-row results the last [`BatchRun::run`] stored, one `i64` per row
    /// in the result's bank encoding: a `double` as its bit pattern, a `string`
    /// as a rank into [`BatchRun::distinct`], a temporal value as nanoseconds.
    ///
    /// Empty unless the run was prepared [`BatchReduce::PerRow`]. Exactly as
    /// long as the batch has rows: the buffer carries one spare element so a
    /// zero-row batch still has a non-null address to seed, and that element is
    /// never a result.
    pub fn output(&self) -> &[i64] {
        self.out.as_deref().map_or(&[], |b| &b[..self.rows])
    }

    /// The batch's distinct strings in rank order, indexed by an output id.
    pub fn distinct(&self) -> &[String] {
        &self.distinct
    }

    /// The flat element buffers a LIST-valued result wrote, one per output
    /// field. Row `i`'s elements are the `output()[i]` entries starting at the
    /// sum of the counts before it.
    pub fn list_output(&self) -> Vec<&[i64]> {
        self.list_out.iter().map(|b| &**b).collect()
    }
}

/// Build the batch program for `n` rows over `columns`. Panics on a column set
/// that does not match `lowered.slots` — count, bank or row length — since that
/// is a caller bug rather than a property of the data.
pub fn prepare_batch<'a>(
    lowered: &super::lower::LoweredF,
    columns: &[Column<'a>],
    n: usize,
    what: &str,
) -> BatchRun<'a> {
    prepare_batch_reduce(lowered, columns, n, what, BatchReduce::Sum)
}

/// [`prepare_batch`] for a chosen reduction. Under [`BatchReduce::PerRow`] the
/// run owns an `n`-element output buffer and the loop stores through its
/// address, seeded like any other batch datum.
/// Extra passes of one named stage of [`prepare_batch_reduce`], so that what a
/// stage costs is a difference taken inside ONE binary.
///
/// The same discipline as [`float_bank::EntryStageRepeats`] and for the same
/// reason: `bind` is tens of nanoseconds on the shapes that matter and
/// `Instant::now()` costs 20-25 ns here, so a clock per stage would report
/// itself. Time the call with every count at zero, again with one count at `k`,
/// and the difference over `k` is that stage.
///
/// Read ONCE per call, ahead of every stage, so no stage's difference carries
/// the read. A build carrying this feature therefore pays one thread-local read
/// and seven zero-trip loop tests per bind that a default build does not, so no
/// arm's ABSOLUTE figure is what the shipping door costs -- only differences are
/// claims.
///
/// ⚠ Every stage here is repeated by RE-RUNNING it and discarding the answer,
/// which is sound only because each is a pure function of inputs the loop does
/// not mutate. The allocating stages ([`EncodeStageRepeats::trap`],
/// [`EncodeStageRepeats::out`]) therefore price an allocation AND its matching
/// free, which is an over-estimate of the allocation alone and is the honest
/// reading of them.
#[cfg(feature = "__encode-stage-probe")]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct EncodeStageRepeats {
    /// Extra passes of the two assertion sweeps: column count and bank match,
    /// then the row-length check over every `SlotKind::Row` column.
    pub asserts: u32,
    /// Extra `temporal_out_of_domain` scans. The shipping path runs this TWICE
    /// per bind -- once as a checked `BatchError` in `BatchProgram::bind`'s
    /// encode half, then again as the assert here -- so this stage priced at
    /// `k` describes ONE of the two.
    pub temporal: u32,
    /// Extra `StrDict::build_columns` calls over the same columns.
    ///
    /// The stage this probe exists to settle. It runs UNCONDITIONALLY, with no
    /// guard on whether the batch carries a `Column::Str` or the expression a
    /// `SeedKind::StrId`, so a scalar activation with no strings anywhere still
    /// pays whatever it costs with nothing to rank.
    ///
    /// It covers the id COLUMNS as well as the table, because the two are one
    /// pass where the ids need no order — there is no separate encode stage
    /// left to amplify on that arm.
    pub strdict: u32,
    /// Extra `bases` vectors: one `i64` per column, read back out of the
    /// caller's buffers and the materialized id columns.
    pub bases: u32,
    /// Extra trap-word `Box`es.
    pub trap: u32,
    /// Extra per-row output buffers. Zero-trip under `BatchReduce::Sum`, which
    /// allocates none.
    pub out: u32,
    /// Extra passes of an amplification loop with NO stage in it: the counter,
    /// and the one optimization barrier every other field's loop also carries.
    /// Subtracting it is what leaves a stage's own cost rather than its cost
    /// plus the machinery that made it repeat.
    pub barrier: u32,
}

#[cfg(feature = "__encode-stage-probe")]
std::thread_local! {
    /// The counts the next bind on this thread runs with.
    static ENCODE_STAGE_REPEATS: core::cell::Cell<EncodeStageRepeats> = const {
        core::cell::Cell::new(EncodeStageRepeats {
            asserts: 0,
            temporal: 0,
            strdict: 0,
            bases: 0,
            trap: 0,
            out: 0,
            barrier: 0,
        })
    };
}

#[cfg(feature = "__encode-stage-probe")]
std::thread_local! {
    /// How many times the amplified `out` stage actually ran its body.
    ///
    /// Published by the stage rather than inferred outside it, because the two
    /// ways that stage can read ~0 ns -- zero-trip under `BatchReduce::Sum`,
    /// and elimination of an allocation nothing observes -- are
    /// indistinguishable in the timing alone.
    static ENCODE_STAGE_OUT_PASSES: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

/// Passes the amplified `out` stage has run since [`reset_encode_stage_passes`].
#[cfg(feature = "__encode-stage-probe")]
pub fn encode_stage_out_passes() -> u64 {
    ENCODE_STAGE_OUT_PASSES.with(core::cell::Cell::get)
}

/// Zero the pass counter, so a section counts only its own arms.
#[cfg(feature = "__encode-stage-probe")]
pub fn reset_encode_stage_passes() {
    ENCODE_STAGE_OUT_PASSES.with(|c| c.set(0));
}

/// Set the repeat counts for subsequent binds on this thread, answering what
/// they were.
///
/// A setter rather than a wrapper around one call, for the same reason
/// [`float_bank::set_entry_stage_repeats`] is one: the whole-cost figure this
/// splits is read through `BatchProgram::bind_per_row_resolved`, so the stages
/// have to be read through it too, or the parts and the whole describe
/// different doors.
#[cfg(feature = "__encode-stage-probe")]
pub fn set_encode_stage_repeats(repeats: EncodeStageRepeats) -> EncodeStageRepeats {
    ENCODE_STAGE_REPEATS.with(|slot| slot.replace(repeats))
}

pub fn prepare_batch_reduce<'a>(
    lowered: &super::lower::LoweredF,
    columns: &[Column<'a>],
    n: usize,
    what: &str,
    reduce: BatchReduce,
) -> BatchRun<'a> {
    // Read once per call, ahead of every stage, so no stage's difference
    // carries it. See [`EncodeStageRepeats`].
    #[cfg(feature = "__encode-stage-probe")]
    let repeats = ENCODE_STAGE_REPEATS.with(core::cell::Cell::get);
    #[cfg(feature = "__encode-stage-probe")]
    for _ in 0..repeats.barrier {
        std::hint::black_box(&columns);
    }
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
    #[cfg(feature = "__encode-stage-probe")]
    for _ in 0..repeats.asserts {
        // The INPUTS are barriered, not only the results. Both `columns` and
        // `lowered` are loop-invariant and these two checks are pure, so
        // barriering only the answers would let the work be computed once and
        // the same values handed to the barrier every pass -- a stage that
        // counts its passes and prices nothing. The allocating stages below do
        // not need this: an allocation has effects the optimizer must keep
        // once its result is observed. A PURE stage has no such protection.
        let columns = std::hint::black_box(columns);
        let lowered = std::hint::black_box(lowered);
        for (col, slot) in columns.iter().zip(&lowered.slots) {
            std::hint::black_box(col.matches(slot.ty));
        }
        for (c, slot) in columns.iter().zip(&lowered.slots) {
            if slot.kind != SlotKind::Row {
                continue;
            }
            std::hint::black_box(c.len() == n);
        }
    }
    #[cfg(feature = "__encode-stage-probe")]
    for _ in 0..repeats.temporal {
        // Inputs barriered for the reason given on `asserts` above: this is a
        // pure, loop-invariant computation and is the stage most exposed to
        // being hoisted.
        let columns = std::hint::black_box(columns);
        let lowered = std::hint::black_box(lowered);
        std::hint::black_box(lowered.temporal_out_of_domain(columns).is_none());
    }
    // Temporal arithmetic is exact only inside the domain the lowering
    // recorded. Callers on the public path ask first and get a `BatchError`;
    // reaching here out of domain is a harness bug, and a wrong sum is a worse
    // outcome than a panic.
    assert!(
        lowered.temporal_out_of_domain(columns).is_none(),
        "{what}: temporal column outside the ±{:?}ns arithmetic domain",
        lowered.temporal_bound
    );
    // The unconditional dictionary build, amplified. Re-running it is sound
    // because it reads only `columns` and `lowered`, neither of which any pass
    // mutates.
    #[cfg(feature = "__encode-stage-probe")]
    for _ in 0..repeats.strdict {
        std::hint::black_box(StrDict::build_columns(
            columns,
            lowered.scalar_seeds.iter().filter_map(|s| match &s.kind {
                super::lower::SeedKind::StrId(t) => Some(t.as_str()),
                super::lower::SeedKind::StrPredicate(_) => None,
            }),
            lowered.needs_ordered_str_ids(),
        ));
    }
    // Rank every string this batch can be asked about — the column values and
    // the expression's literals together — so all of them share one assignment,
    // and encode the columns in the same call.
    let (dict, str_ids) = StrDict::build_columns(
        columns,
        // An id seed is compared against column ids, so it must share their
        // assignment. A PREDICATE's argument is not — it is never an id — so it
        // stays out of the dictionary.
        lowered.scalar_seeds.iter().filter_map(|s| match &s.kind {
            super::lower::SeedKind::StrId(t) => Some(t.as_str()),
            super::lower::SeedKind::StrPredicate(_) => None,
        }),
        lowered.needs_ordered_str_ids(),
    );
    // `bases` reads the id columns positionally, so an amplified pass needs its
    // own cursor rather than sharing the real one.
    #[cfg(feature = "__encode-stage-probe")]
    for _ in 0..repeats.bases {
        let mut id_col = 0;
        let v: Vec<i64> = columns
            .iter()
            .map(|c| match c {
                Column::Int(x) => x.as_ptr() as i64,
                Column::Float(x) => x.as_ptr() as i64,
                Column::Bool(x) => x.as_ptr() as i64,
                Column::Str(_) => {
                    let base = str_ids[id_col].as_ptr() as i64;
                    id_col += 1;
                    base
                }
            })
            .collect();
        std::hint::black_box(v);
    }
    let mut next_id_col = 0;
    // Frame-held: `regs_list` copies these into the seed it returns, so nothing
    // downstream reads the list of bases itself. See [`super::inline`].
    let mut bases: Inline<i64, INLINE_SLOTS> = Inline::with_capacity(0, columns.len());
    for c in columns {
        bases.push(match c {
            Column::Int(x) => x.as_ptr() as i64,
            Column::Float(x) => x.as_ptr() as i64,
            Column::Bool(x) => x.as_ptr() as i64,
            Column::Str(_) => {
                let base = str_ids[next_id_col].as_ptr() as i64;
                next_id_col += 1;
                base
            }
        });
    }
    // Allocation stages price an allocation AND its matching free, since the
    // amplified copy is dropped at the end of each pass. That is an
    // over-estimate of the allocation alone and is how they must be read.
    #[cfg(feature = "__encode-stage-probe")]
    for _ in 0..repeats.trap {
        std::hint::black_box(Box::new(0i64));
    }
    #[cfg(feature = "__encode-stage-probe")]
    for _ in 0..repeats.out {
        // Counted, not just run. This stage is zero-trip under
        // `BatchReduce::Sum`, and a stage the compiler ELIDED would report the
        // same near-zero figure as one that never executed. The counter is what
        // tells the two apart; without it the report cannot distinguish
        // "genuinely skipped" from "optimized away", and those have opposite
        // consequences for whether the cost is real.
        if reduce == BatchReduce::PerRow {
            std::hint::black_box(vec![0i64; n.max(1)].into_boxed_slice());
            ENCODE_STAGE_OUT_PASSES.with(|c| c.set(c.get() + 1));
        }
    }
    let mut trap: Box<i64> = Box::new(0);
    let trap_addr = (&mut *trap) as *mut i64 as i64;
    // A zero-row batch never enters the loop, but the seed still asserts a
    // non-null output address, so give the buffer one element to point at.
    let mut out: Option<Box<[i64]>> = match reduce {
        BatchReduce::Sum => None,
        BatchReduce::PerRow => Some(vec![0i64; n.max(1)].into_boxed_slice()),
    };
    let out_addr = out.as_mut().map_or(0, |b| b.as_mut_ptr() as i64);
    // A list-valued result writes at most as many elements as its SOURCE offers
    // across the whole batch — `map` writes exactly that many, `filter` fewer.
    // For a list column that is its flattened element count; for a literal list
    // it is the green length, once per row.
    //
    // The count is taken from the source's own `size(..)` column rather than
    // from the length of an element column, because a body that reads no
    // element at all (`nums.map(y, 1)`) leaves no element column to measure
    // while still writing one element per input element. `size(..)` is the
    // loop's trip count, so it is present whatever the body reads.
    let mut list_out: Vec<Box<[i64]>> = match (reduce, &lowered.list_output) {
        (BatchReduce::PerRow, Some(o)) => {
            let cap = match &o.source {
                super::lower::ListSource::Column(src) => {
                    let key = super::lower::size_slot_path(src);
                    columns
                        .iter()
                        .zip(&lowered.slots)
                        .find(|(_, slot)| slot.path == key)
                        .and_then(|(c, _)| match c {
                            Column::Int(v) => Some(v.iter().sum::<i64>().max(0) as usize),
                            _ => None,
                        })
                        .unwrap_or(0)
                }
                super::lower::ListSource::Literal(len) => n * len,
            };
            // One spare, so a batch that writes nothing still has an address.
            o.fields
                .iter()
                .map(|_| vec![0i64; cap.max(1)].into_boxed_slice())
                .collect()
        }
        _ => Vec::new(),
    };
    // Frame-held, like `bases`: one entry per output field, copied into the
    // seed and not kept.
    let mut list_addrs: Inline<i64, INLINE_SLOTS> = Inline::with_capacity(0, list_out.len());
    for b in list_out.iter_mut() {
        list_addrs.push(b.as_mut_ptr() as i64);
    }
    let shape = lowered.batch_shape(true, reduce);
    // Column bases, the row count, the trap address and the string literals'
    // ids are all this batch's data, and all reach the program the same way:
    // through the seeded bank, never through the words.
    // Predicate tables are answered once per DISTINCT string, so they are built
    // here and their addresses broadcast like any other seed. Held in
    // `pred_tables` for as long as the run is, alongside the id columns.
    let mut pred_tables: Vec<Box<[i64]>> = Vec::new();
    let mut distinct: Option<Vec<&str>> = None;
    // Frame-held, like `bases`: the broadcast scalars are copied into the seed.
    // The predicate TABLES they may point at are not — those go on to the run.
    let mut scalars: Inline<i64, INLINE_SLOTS> =
        Inline::with_capacity(0, lowered.scalar_seeds.len());
    for seed in &lowered.scalar_seeds {
        scalars.push(match &seed.kind {
            super::lower::SeedKind::StrId(t) => dict.id(t),
            super::lower::SeedKind::StrPredicate(p) => {
                let strings = distinct.get_or_insert_with(|| dict.sorted());
                pred_tables.push(p.table(strings).into_boxed_slice());
                pred_tables[pred_tables.len() - 1].as_ptr() as i64
            }
        });
    }
    // Appended rather than re-collected. The boxes MOVE and the buffers they
    // point at do not, so the addresses already baked into `scalars` stay
    // valid — and a batch with no predicate table, which is every batch that
    // carries no string predicate, now pays no allocation here at all where the
    // `chain(..).collect()` this replaces built a second vector unconditionally.
    let mut str_ids = str_ids;
    str_ids.append(&mut pred_tables);
    // A string-banked per-row output stores ids, which mean nothing without the
    // order they were ranked in. A COLLECTED list stores them just the same, and
    // its bank is on the output's fields rather than on the row result — which
    // is an int count whatever the elements are.
    // The same predicate `needs_ordered_str_ids` consulted when the dictionary
    // was built, asked through the same accessor: this table is what makes the
    // promise that the ids in it are comparable, so a second spelling here
    // could order the ids and then hand back a table built on the other rule.
    let str_result = lowered.has_string_result();
    let distinct: Vec<String> = match (reduce, str_result) {
        (BatchReduce::PerRow, true) => distinct
            .get_or_insert_with(|| dict.sorted())
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
        _ => Vec::new(),
    };
    let init_regs =
        shape
            .seed
            .regs_list(&bases, &scalars, n as i64, trap_addr, out_addr, &list_addrs);
    // What a projection's loop would have copied out, copied out HERE instead:
    // it is a property of the program and the columns, both of which are fixed
    // once the batch is prepared, so no later call has anything to recompute.
    // The output buffer is `i64` whatever the bank, so a `double` goes in as
    // its bit pattern and a `bool` as a widened byte. A `string` column goes in
    // as the ids the loop would have loaded rather than as the caller's
    // strings, and those are `str_ids`' only entry — a program with a body is
    // not a projection, and only a body can ask for a predicate table.
    let projection = match (reduce, lowered.is_row_projection()) {
        (BatchReduce::PerRow, true) => {
            let buf = out.as_mut().expect("a per-row run owns an output buffer");
            let buf = &mut buf[..n];
            match columns[0] {
                Column::Int(c) => buf.copy_from_slice(&c[..n]),
                Column::Float(c) => {
                    for (o, &v) in buf.iter_mut().zip(&c[..n]) {
                        *o = v.to_bits() as i64;
                    }
                }
                Column::Bool(c) => {
                    for (o, &v) in buf.iter_mut().zip(&c[..n]) {
                        *o = i64::from(v);
                    }
                }
                Column::Str(_) => buf.copy_from_slice(&str_ids[0][..n]),
            }
            true
        }
        _ => false,
    };
    // A refcount bump on the words `lowered` owns, not a copy: every batch of
    // this expression runs the same allocation, so the JIT's green key — and
    // with it the compiled loop the driver holds — stays put between batches.
    let code = shape.code.clone();
    BatchRun {
        code,
        init_regs,
        num_float_regs: shape.num_float_regs,
        trap,
        rows: n,
        str_ids,
        projection,
        out,
        list_out,
        distinct,
        banks: float_bank::Banks::default(),
        columns: core::marker::PhantomData,
    }
}

/// [`prepare_batch`] plus one run, for a caller evaluating a batch once.
fn batch_sum_with(
    lowered: &super::lower::LoweredF,
    columns: &[Column],
    n: usize,
    what: &str,
    run: impl FnOnce(&std::sync::Arc<Code>, &[i64], usize, &mut float_bank::Banks) -> i64,
) -> Option<i64> {
    prepare_batch(lowered, columns, n, what).run(run)
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
    batch_sum_with(
        lowered,
        columns,
        n,
        "eval_batch_sum_f",
        |prog, regs, nf, _| float_bank::run_jit_persistent_f(prog, regs, nf, threshold),
    )
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
        // The oracle tier keys nothing on the address, so it takes the words as
        // a plain slice and the `Arc` is dropped at the boundary.
        |code, regs, nf, banks| float_bank::clean_interp_seeded_f_in(code, regs, nf, banks),
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
    use majit_metainterp::embed::Census;

    /// The tier's counters are [`Census`]'s, not this module's.
    ///
    /// They used to be eight statics here with four hand-installed callbacks
    /// bumping six of them and the other two read off the drivers. Every one of
    /// those is a thing an embedder must not get wrong on its own, and two of
    /// them were: a tally read only from RETIRED drivers is zero for a pool
    /// whose drivers all live, and a per-counter reset is not a window.
    ///
    /// [`jit_stats`] and [`reset_jit_stats`] stay as the names this crate reads
    /// them under. What moved is who owns the counters.
    use super::{
        OP_ADD, OP_ADD_IMM, OP_ADD_OVF, OP_AND, OP_COL_LOAD, OP_COL_LOAD_B, OP_COL_LOAD_F,
        OP_COL_STORE, OP_COL_STORE_F, OP_DIV, OP_DIV_CHK, OP_EQ, OP_F2I, OP_F2U, OP_FADD, OP_FDIV,
        OP_FEQ, OP_FGE, OP_FGT, OP_FLE, OP_FLT, OP_FMOV, OP_FMUL, OP_FNE, OP_FNEG, OP_FSELECT,
        OP_FSUB, OP_GE, OP_GT, OP_I2F, OP_JUMP_IF_ABOVE, OP_LE, OP_LOAD_CONST, OP_LOAD_CONST_F,
        OP_LT, OP_MOD, OP_MOD_CHK, OP_MOV, OP_MUL, OP_MUL_IMM, OP_MUL_OVF, OP_NE, OP_NEG, OP_NOT,
        OP_OR, OP_RETURN, OP_RETURN_F, OP_SELECT, OP_SUB, OP_SUB_OVF, OP_TRAP_STORE, OP_U2F,
        OP_UADD_OVF, OP_UDIV, OP_ULE, OP_ULT, OP_UMOD, OP_UMUL_OVF, OP_USUB_OVF,
    };

    // majit's intrinsics, not a local copy: the `#[jit_interp]` macro matches a
    // call by its LAST PATH SEGMENT, so an imported name lowers to the same
    // trace op a definition here did.
    //
    // `majit_uint_mul_high` is the documented exception — it has no hard-coded
    // name and is reached only through this mainloop's `native_int_binops`
    // alias, which matches the call's FULL path against the configured key. The
    // alias and these call sites must therefore both stay on the bare name.
    //
    // The obligations stay cel's. A load's `base` is a column buffer's address
    // and `ea` a byte offset, so it reads `col[i]` at a red row index when
    // `ea == i * 8`; `base` is a loop-invariant in the register file, NOT a
    // scalar state field, which would force the virtualizable frame to a Ref
    // and trip `VirtualStatesCantMatch` at loop close. A `&[bool]` column's
    // stride is one, so `majit_raw_load_u8` takes an element index. The batch
    // builder covers every row index; the trap word `OP_TRAP_STORE` publishes
    // through outlives the run; `OP_UDIV`/`OP_UMOD` guard `b != 0` before they
    // call.
    use majit_metainterp::intrinsics::{
        majit_bits_to_f64, majit_f64_to_bits, majit_raw_load_f, majit_raw_load_i64,
        majit_raw_load_u8, majit_raw_store_i64, majit_uint_div, majit_uint_le, majit_uint_lt,
        majit_uint_mod, majit_uint_mul_high,
    };

    struct VmStateF {
        regs: majit_metainterp::virt_array::VirtArray<i64>,
        fregs: majit_metainterp::virt_array::VirtArray<f64>,
        /// What a terminal opcode hands back, as raw i64 bits.
        ///
        /// The `; state` merge point leaves the loop through `break` before it
        /// assigns the walk's resume pc, so after the loop `pc` still names the
        /// position the walk started from and `program[pc + 1]` — the operand
        /// saying which register holds the result — cannot be read there. The
        /// result has to arrive in `state`, and a scalar field is the direct way
        /// to carry it.
        ret: i64,
    }

    /// The pc a program starts at, and the position green the function-entry
    /// door keys on.
    ///
    /// It is a different key from any the row loop takes: the loop's back edge
    /// arms at its body header, which sits past the prologue that seeds the
    /// column bases and the row counter, so the two positions cannot coincide.
    /// Two keys means two traces — an entry trace covering the whole call and a
    /// loop trace covering one row — which is the shape a call-counted door has.
    const ENTRY_PC: usize = 0;

    /// The register banks a call starts from.
    ///
    /// Only the int bank is seeded: the column bases, the row count and the
    /// trap address all live there, and no float ever enters from outside.
    fn initial_state_f(init_regs: &[i64], num_fregs: usize) -> VmStateF {
        VmStateF {
            regs: majit_metainterp::virt_array::VirtArray::from_slice(init_regs),
            fregs: majit_metainterp::virt_array::VirtArray::filled(0.0, num_fregs),
            ret: 0,
        }
    }

    /// [`initial_state_f`] over banks the caller keeps between calls.
    ///
    /// The banks arrive in whatever state the previous call left them, so all
    /// three fields are re-established here rather than assumed. Only the
    /// allocation is saved, not the initialization — and saving it is the
    /// point, because a one-row call is otherwise two heap allocations of
    /// prologue for a few nanoseconds of work.
    ///
    /// Each bank is resized to the length this call needs and then overwritten
    /// whole. The float bank is filled rather than emptied first: `resize`
    /// initializes only the slots it adds, so a bank that already had the right
    /// length would keep the previous call's values, and clearing it to force
    /// the fill would give up the allocation this function exists to keep — a
    /// bank whose storage is a single exact-length block has no spare capacity
    /// to shrink into and back out of.
    fn reseed_state_f(state: &mut VmStateF, init_regs: &[i64], num_fregs: usize) {
        state.regs.resize(init_regs.len(), 0);
        state.regs.copy_from_slice(init_regs);
        state.fregs.resize(num_fregs, 0.0);
        state.fregs.fill(0.0);
        state.ret = 0;
    }

    #[majit_macros::jit_interp(
        state = VmStateF,
        env = Code,
        greens = [pc, program],
        state_fields = {
            regs: [int; virt],
            fregs: [float; virt],
            ret: int,
        },
        // `opcode_for_binop` has no unsigned spelling and `BindingKind` carries
        // no signedness, so the unsigned multiply-high resop is reached by
        // aliasing the helper call to the opcode.
        native_int_binops = { majit_uint_mul_high => UintMulHigh },
    )]
    fn run_mainloop_f(
        mut driver: &mut majit_metainterp::JitDriver<VmStateF>,
        program: &Code,
        state: &mut VmStateF,
        mut pc: usize,
    ) -> i64 {
        loop {
            // `; state` selects the single-pass close: the walk's final state is
            // transferred into `state` here and the native loop resumes at the
            // close pc, instead of re-running the circuit the walk already
            // executed. It also runs `run_pending_abort_blackhole`, which
            // finishes the opcodes a mid-opcode abort left half-executed — a
            // consumer on the bare form never reaches that, because the driver
            // drops the stale stash at the next merge-point entry.
            jit_merge_point!(driver, program, pc; *state);
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
                OP_MUL_IMM => {
                    let a = program[pc + 1] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = state.regs[a] * program[pc + 2];
                    pc += 4;
                }
                OP_ADD_IMM => {
                    let a = program[pc + 1] as usize;
                    let d = program[pc + 3] as usize;
                    state.regs[d] = state.regs[a] + program[pc + 2];
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
                OP_COL_STORE => {
                    let base = state.regs[program[pc + 1] as usize];
                    let ea = state.regs[program[pc + 2] as usize];
                    majit_raw_store_i64(base, ea, state.regs[program[pc + 3] as usize]);
                    pc += 4;
                }
                OP_COL_STORE_F => {
                    let base = state.regs[program[pc + 1] as usize];
                    let ea = state.regs[program[pc + 2] as usize];
                    let bits = majit_f64_to_bits(state.fregs[program[pc + 3] as usize]);
                    majit_raw_store_i64(base, ea, bits);
                    pc += 4;
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
                    state.regs[program[pc + 3] as usize] = majit_raw_load_i64(base, ea);
                    pc += 4;
                }
                OP_COL_LOAD_B => {
                    let base = state.regs[program[pc + 1] as usize];
                    let ea = state.regs[program[pc + 2] as usize];
                    state.regs[program[pc + 3] as usize] = majit_raw_load_u8(base, ea);
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
                OP_U2F => {
                    state.fregs[program[pc + 2] as usize] =
                        state.regs[program[pc + 1] as usize] as u64 as f64;
                    pc += 3;
                }
                OP_F2I => {
                    state.regs[program[pc + 2] as usize] =
                        state.fregs[program[pc + 1] as usize] as i64;
                    pc += 3;
                }
                OP_F2U => {
                    state.regs[program[pc + 2] as usize] =
                        state.fregs[program[pc + 1] as usize] as u64 as i64;
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
                            can_enter_jit!(driver, tgt, &mut *state, program, || {});
                        }
                        pc = tgt;
                        continue;
                    }
                    pc += 4;
                }
                // The terminal arms store into `ret` and then leave through an
                // in-arm `return`, never `{ store; break }`: `classify.rs`
                // `is_break_expr` requires the arm body to be exactly `break`, so
                // a composite body classifies `Lowerable` and its tail `break`
                // reaches `lower_stmt_fallback`, which has a guard for an
                // enclosed `return` but none for `break` — the statement is inert
                // and is silently dropped, leaving the lowered arm to fall
                // through to the dispatch back-edge. `lower_return_stmt` lowers
                // the `return` spelling in place.
                OP_RETURN => {
                    state.ret = state.regs[program[pc + 1] as usize];
                    return state.ret;
                }
                OP_RETURN_F => {
                    state.ret = majit_f64_to_bits(state.fregs[program[pc + 1] as usize]);
                    return state.ret;
                }
                // Was `_ => break` with the panic below the loop. The loop now
                // has a second way out — the merge point's own `break` on a walk
                // that reached a terminal return — so falling out of it no longer
                // identifies a bad opcode, and the panic has to move into the arm
                // that actually saw one.
                _ => panic!("fell off end of code"),
            }
        }
        // Reached only when the merge point broke out on a walk that already ran
        // a terminal opcode, so the result is whatever that opcode parked in
        // `ret`. The write-back that put it there runs ahead of the `break`.
        state.ret
    }

    /// [`run_mainloop_f`] from the top of `program` with freshly seeded banks —
    /// the shape every caller had before the entry door, and what a caller with
    /// no driver to reuse still wants.
    fn run_mainloop_from_start_f(
        driver: &mut majit_metainterp::JitDriver<VmStateF>,
        program: &Code,
        init_regs: &[i64],
        num_fregs: usize,
    ) -> i64 {
        run_mainloop_f(
            driver,
            program,
            &mut initial_state_f(init_regs, num_fregs),
            ENTRY_PC,
        )
    }

    /// A green key the pool computed once, in both the forms a probe needs.
    ///
    /// The hash names a BUCKET, and a bucket can hold more than one cell, so a
    /// probe that stops at the hash is asking about whichever cell heads the
    /// chain rather than about the one its greens own. The typed halves are
    /// what settles that, through `comparekey`. Carrying both is what lets
    /// every door in this file decide on the cell it means.
    struct PooledGreenKey {
        hash: u64,
        /// The `(i64 bits, GreenType)` pair per slot, as
        /// `__majit_green_key_run_mainloop_f` built them. Held rather than
        /// split into parallel arrays because that is the shape the generated
        /// builder hands back, and re-splitting it here would be a second place
        /// for a slot to move.
        slots: [(i64, majit_ir::GreenType); 3],
    }

    impl PooledGreenKey {
        /// The key of the cell these greens own.
        ///
        /// One bucket walk. The typed key is BUILT only where the bucket is
        /// chained — `resolve_cell_key` answers an empty or single-cell bucket
        /// from the walk alone — so the two allocations behind it are paid on
        /// the shape that needs them and not on every call. On every bucket
        /// holding one cell or none the answer is the raw hash, which is what
        /// the probes asked before this existed.
        fn resolve(&self, driver: &majit_metainterp::JitDriver<VmStateF>) -> u64 {
            driver.resolve_cell_key(self.hash, || {
                majit_ir::GreenKey::with_types(
                    self.slots.iter().map(|slot| slot.0).collect(),
                    self.slots.iter().map(|slot| slot.1).collect(),
                )
            })
        }
    }

    /// Which key form the yield probe in [`try_function_entry_jit_f`] asks on.
    ///
    /// A measurement probe, not a feature. The two variants are the door before
    /// and after the resolution landed, and they exist as two branches of ONE
    /// compiled function so that what the resolution costs is a difference
    /// taken inside a single binary. Two builds cannot answer that: they admit
    /// compile drift and stale binaries, and neither is visible in the numbers
    /// they produce.
    ///
    /// The two agree on every bucket that holds one cell or none, which is
    /// every bucket a real program has ever produced here —
    /// [`PooledGreenKey::resolve`] answers such a bucket from the walk alone
    /// and returns the raw hash. So this is a cost probe, not a behaviour
    /// switch: the arms differ in what they DO, not in what they decide.
    #[cfg(feature = "__loop-key-arm-probe")]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum LoopKeyArm {
        /// The bare bucket hash, which is what the probe asked before the
        /// resolution landed. No walk, no cell read — the hash is already in
        /// the pool.
        BareHash,
        /// What the door does without the probe: one bucket walk per loop key,
        /// through [`PooledGreenKey::resolve`].
        Resolved,
    }

    #[cfg(feature = "__loop-key-arm-probe")]
    std::thread_local! {
        /// The arm the next call on this thread takes at the yield probe.
        ///
        /// Read once per CALL — not once per loop key — and identically by both
        /// arms, so it is a constant that cancels out of any difference between
        /// them. It does mean no arm's ABSOLUTE figure is what the shipping
        /// door costs; only differences are claims.
        static LOOP_KEY_ARM: core::cell::Cell<LoopKeyArm> =
            const { core::cell::Cell::new(LoopKeyArm::Resolved) };
    }

    /// [`run_jit_persistent_f`] with the yield probe's key form chosen
    /// explicitly.
    ///
    /// Announced through a thread-local rather than threaded down as an
    /// argument, because the door is reached through the pool's `with` closure
    /// and a parameter would have to be spelled in the shipping signature too.
    /// Saved and restored around the one call, so two arms interleaved in one
    /// process never observe each other's setting.
    #[cfg(feature = "__loop-key-arm-probe")]
    pub fn run_jit_persistent_probe_f(
        program: &std::sync::Arc<Code>,
        init_regs: &[i64],
        num_fregs: usize,
        threshold: u32,
        arm: LoopKeyArm,
    ) -> i64 {
        let previous = LOOP_KEY_ARM.with(|slot| slot.replace(arm));
        let out = run_jit_persistent_f(program, init_regs, num_fregs, threshold);
        LOOP_KEY_ARM.with(|slot| slot.set(previous));
        out
    }

    /// How many loop keys the yield probe walks for `program` on every call.
    ///
    /// The divisor a per-key figure needs. Exposed for the probe only: a
    /// caller quoting a per-key cost has to divide by the count the door
    /// actually walks, and that count is [`loop_header_keys`]'s answer rather
    /// than the number of loops a reader would count in the source.
    #[cfg(feature = "__loop-key-arm-probe")]
    pub fn loop_key_count(program: &Code) -> usize {
        loop_header_keys(program).len()
    }

    /// The green key a door arming at `pc` in `program` files under.
    ///
    /// The slots are the ones `can_enter_jit!` builds for this mainloop — the
    /// marker's own position argument, then each declared green with the arming
    /// position substituted for `pc` — and they are no longer spelled twice.
    /// `#[jit_interp]` emits the builder from the same `greens` list it lowers
    /// the merge point from, so the two cannot drift; this used to write the
    /// seed and the per-slot fold out by hand next to a comment saying they had
    /// to match. A mismatch was undetectable from here: a wrong slot still
    /// hashes, just to a cell nothing files under, and the door would then
    /// probe forever for an artifact that exists.
    fn green_key_at(program: &Code, pc: usize) -> PooledGreenKey {
        let (hash, slots) = __majit_green_key_run_mainloop_f(pc, program);
        PooledGreenKey { hash, slots }
    }

    /// The green keys of `program`'s own loop headers — the target of every
    /// backward [`OP_JUMP_IF_ABOVE`], which is where the row loop and any inner
    /// element loop arm their `can_enter_jit!`.
    ///
    /// A word-wise scan, not a decode: an operand that happens to hold
    /// `OP_JUMP_IF_ABOVE`'s value contributes a position that is not an
    /// instruction. That is sound because the result is only ever asked whether
    /// a key is COMPILED, and a key is only ever filed at a position some door
    /// armed at — a real back-edge target, or [`ENTRY_PC`]. A spurious position
    /// answers no. [`OPERANDS`] would let this decode instead, and a decode
    /// would contribute no spurious position; the scan is kept because it is
    /// the cheaper of two answers this caller cannot tell apart.
    ///
    /// [`ENTRY_PC`] is excluded, and that exclusion is what makes the paragraph
    /// above true rather than nearly true. A spurious position CAN be
    /// [`ENTRY_PC`] — `[1, 2, 3, 4, 5].map(x, x * 2)` unrolls to a body whose
    /// `OP_MUL_OVF` reads register 16 and traps to register 0, so its `[.., 16,
    /// 17, 18, 0, ..]` words read as a back edge to 0 — and that position does
    /// NOT answer no, because it is the key the function-entry door itself
    /// files under. Left in, the door reads its own artifact as evidence that
    /// some other door exists and declines from the call after the one that
    /// minted it, permanently: `compiles=1`, `compiled_entries=0`. Excluding it
    /// costs nothing a real loop header at 0 would have provided, since a door
    /// there and this door are one key and the caller is that door.
    fn loop_header_keys(program: &Code) -> Vec<PooledGreenKey> {
        let mut targets: Vec<usize> = Vec::new();
        for pc in 0..program.len().saturating_sub(3) {
            if program[pc] != OP_JUMP_IF_ABOVE {
                continue;
            }
            let target = program[pc + 3];
            if target < 0 || target as usize >= pc {
                continue;
            }
            let target = target as usize;
            if target == ENTRY_PC {
                continue;
            }
            if !targets.contains(&target) {
                targets.push(target);
            }
        }
        targets
            .into_iter()
            .map(|target| green_key_at(program, target))
            .collect()
    }

    /// The function-entry door: consult the JIT once per CALL, before the first
    /// instruction runs.
    ///
    /// The row loop's `can_enter_jit!` is the only other door, and it is a back
    /// edge on a bottom-tested loop — an `n`-row batch takes `n - 1` of them and
    /// a ONE-row batch takes none. Warmup counted there counts rows, so a
    /// program evaluated a million times at one row per call never warms, and a
    /// program whose loop is already compiled never enters it at one row either.
    /// That is the ordinary way CEL is used: compile the expression once,
    /// evaluate it per record with fresh bindings. This door restores the
    /// counting a call-shaped workload needs, and it is the same driver call the
    /// back edge makes — only keyed on [`ENTRY_PC`] instead of the loop header.
    ///
    /// `Some(value)` means compiled code ran the whole call and FINISHed, so the
    /// value is this call's result and nothing else runs. `None` means the call
    /// is to be interpreted, and the state and pc it must start from come back
    /// through the out-parameters: the door may have already run compiled code
    /// that deopted, in which case the blackhole has recovered `state` to the
    /// guard's resume point and interpretation continues from there rather than
    /// from the top. Re-entering at the top instead would re-run the rows the
    /// compiled prefix already ran, and would record any bridge the deopt
    /// started against the wrong live-ins.
    ///
    /// The branches are ordered by what each costs, because this runs on EVERY
    /// call including the overwhelming majority that do nothing. Only the
    /// compiled-entry branch touches the driver's trace machinery — which builds
    /// the trace metadata and extracts the live values, two allocations, before
    /// it ever consults a counter. A call that is nowhere near the threshold
    /// must not pay for that and throw it away: warming through
    /// `back_edge_structured` alone cost ~10ns per row on a two-row batch,
    /// enough for `majit_shape_change`'s "never worse than the untraced VM"
    /// bound to fire. The counter is what decides here, and it is a hash lookup
    /// and an add.
    fn try_function_entry_jit_f(
        driver: &mut majit_metainterp::JitDriver<VmStateF>,
        program: &Code,
        pooled: &PooledProgram,
        state: &mut VmStateF,
        pc: &mut usize,
    ) -> Option<i64> {
        // A program whose own loop is compiled already has a door, at that
        // loop's back edge, and takes it on the first row. A second door in
        // front of it does not add a way in — it takes one AWAY. Measured on
        // `items.all(i, i.price > 10)` over 20 000-row calls: with an entry
        // artifact minted, every later call entered the ENTRY key once and the
        // loop header not at all, where before it entered the loop header once
        // and ran the whole batch there. Both answer correctly; they are not the
        // same code. The calls this door exists for are the ones the loop's door
        // cannot serve, and those are exactly the calls whose loop never gets
        // hot — a bottom-tested loop at one row takes no back edge at all.
        //
        // `has_runnable_compiled_loop`, not `has_compiled_loop`, because the
        // question this asks is whether the OTHER door can serve the call — a
        // cell carrying only a `compile_tmp_callback` token has a compiled body
        // and no frontend meta, so the generic entry runner cannot execute it
        // and the loop's back edge would keep counting rather than run. Yielding
        // to a door that cannot open would leave the call with no door at all.
        //
        // That this rule makes the two doors mutually exclusive for a program at
        // all is the review's §1-third / §2-first finding (the artifacts do not
        // coexist); resolving it needs majit-side artifact coexistence and is
        // not attempted here.
        //
        // The predicate and not the token form, unlike the entry decision
        // below: this probe asks about a cell some OTHER door runs, and the
        // only outcome here is to decline, so there is no run to hand a token
        // to. It costs one cell read per key either way — the predicate is that
        // read with its answer collapsed to a bool.
        //
        // On the RESOLVED key, like the entry decision below. A loop header's
        // greens are `pc` and the program address, the same shape
        // [`green_key_at`] builds, and the pool now keeps them
        // ([`PooledGreenKey`]) rather than the hash alone — so a chained bucket
        // no longer lets this answer for whichever cell heads the chain.
        //
        // It costs one bucket walk per loop key per call, on a path that runs
        // on every call. Proportionate rather than free: the entry decision
        // below already pays exactly this walk once per call, so the door was
        // already built to afford one, and a program with no backward jump has
        // no loop key to walk for. What it buys is smaller than the entry fix —
        // the outcome here is a yield-or-not decision about a door that is not
        // this one, not an entry into someone else's compiled code — but the
        // two probes now decide on the same footing, which is what stops the
        // weaker one from being read as evidence about the stronger.
        //
        // Under `__loop-key-arm-probe` the key form is selected at RUN time, so
        // this walk and the bare-hash probe that preceded it are two branches
        // of one compiled door. See [`LoopKeyArm`]; the default build has
        // neither the branch nor the read.
        #[cfg(feature = "__loop-key-arm-probe")]
        let arm = LOOP_KEY_ARM.with(core::cell::Cell::get);
        if pooled.loop_keys.iter().any(|key| {
            #[cfg(feature = "__loop-key-arm-probe")]
            let probe = match arm {
                LoopKeyArm::BareHash => key.hash,
                LoopKeyArm::Resolved => key.resolve(driver),
            };
            #[cfg(not(feature = "__loop-key-arm-probe"))]
            let probe = key.resolve(driver);
            driver.has_runnable_compiled_loop(probe)
        }) {
            return None;
        }
        // Read, not recomputed: the key is `green_key_at(program, ENTRY_PC)`,
        // and both of its inputs are fixed for as long as the pool pins this
        // address. See [`PooledProgram`].
        //
        // Resolved once, then carried. `entry_key` holds a bucket hash, and a
        // bucket can hold more than one cell; the typed key settles which one
        // this door's greens own, and every step below — the entry decision,
        // the token it decides on, and the run that token is handed to — uses
        // that one answer. The door used to ask on the bare hash and then hand
        // the bare hash on, which was two resolutions per warm call and, on a
        // chained bucket, a decision about one cell followed by a run keyed
        // through another.
        let entry_key = pooled.entry_key.resolve(driver);
        // The token IS the decision — `Some` is the runnable-compiled-loop
        // answer this branch used to ask for as a predicate, and it is the same
        // object the run below enters, so nothing between the two can make them
        // disagree.
        //
        // It is the runnable form, not the weaker code-present one:
        // code-present is true as soon as the cell's token has a body, which
        // includes a `compile_tmp_callback` stub that has NO frontend
        // `compiled_loops` meta; the dispatch inside the entry path unwraps
        // that meta unconditionally, so entering on the weaker answer panics
        // instead of falling through to interpretation. Upstream's
        // `maybe_compile_and_run` treats a temporary token by continuing to
        // count, which is what declining here does. cel mints no
        // CALL_ASSEMBLER token today — tmp callbacks come from recursive calls
        // this VM has no opcode for — so this is defense, not a live crash
        // being fixed.
        if let Some(procedure_token) = driver.runnable_procedure_token(entry_key) {
            // The same run the row loop's back edge reaches, so entry, guard
            // failure, blackhole resume and bridge start are all handled the one
            // way — only reached with the cell and its token already in hand
            // instead of re-derived from the hash.
            let resume = driver.back_edge_resolved(
                entry_key,
                procedure_token,
                ENTRY_PC,
                state,
                program,
                || {},
            );
            // Drained before `resume` is read, and for the same reason the back
            // edge's expansion drains it first: a run that FINISHed did not stop
            // at a resume point, and the pc offered alongside the finish is the
            // door's own arming position, so taking it would run the call a
            // second time.
            if let Some(value) = driver.take_back_edge_finish_int() {
                return Some(value);
            }
            *pc = resume.unwrap_or(ENTRY_PC);
            return None;
        }
        // A trace left live by an earlier call is the row loop's, recording from
        // wherever it started; arming a second one here would abandon it.
        if driver.is_tracing() {
            return None;
        }
        // On the resolved key as well, so the counter that decides to trace, the
        // trace that starts, and the entry decision above are all about the one
        // cell this door's greens own. Identical to the raw hash on every bucket
        // that holds a single cell, which includes every bucket that holds none
        // — so a cold program's first calls count exactly where they did.
        if driver
            .meta_interp_mut()
            .warm_state_mut()
            .should_trace_function_entry(entry_key)
        {
            // The counter above is the whole threshold decision, so the start
            // has to be the one that does not consult a counter of its own.
            driver.force_start_tracing(entry_key, ENTRY_PC, state, program);
        }
        None
    }

    /// Reference two-bank interpreter — correctness oracle for the float path.
    pub fn clean_interp_f(program: &Code, num_regs: usize, num_fregs: usize) -> i64 {
        clean_interp_seeded_f(program, &vec![0i64; num_regs], num_fregs)
    }

    /// Register banks owned across executes, so a caller that runs the same
    /// shape repeatedly stops paying for two heap allocations per run.
    ///
    /// A caller-owned pair rather than a `thread_local!` one on purpose. The
    /// cost being removed is ~8-9 ns of allocation, and a TLS read through
    /// `_tlv_get_addr` is about that much on its own — a pool behind a
    /// `thread_local!` would have spent the saving on reaching it.
    #[derive(Default)]
    pub struct Banks {
        regs: Vec<i64>,
        fregs: Vec<f64>,
    }

    /// [`clean_interp_f`] over a caller-supplied initial int register bank, for
    /// programs whose data (column bases, row count, trap-word address) arrives
    /// in registers instead of as immediates.
    ///
    /// Allocates its banks per call. A caller in a loop wants
    /// [`clean_interp_seeded_f_in`], which is this function over banks it keeps.
    pub fn clean_interp_seeded_f(program: &Code, init_regs: &[i64], num_fregs: usize) -> i64 {
        clean_interp_seeded_f_in(program, init_regs, num_fregs, &mut Banks::default())
    }

    /// [`clean_interp_seeded_f`] over banks the caller keeps between runs.
    ///
    /// The banks arrive in whatever state the previous run left them, so both
    /// are re-established here rather than assumed: the int bank is resized and
    /// overwritten from `init_regs`, and the float bank is refilled with zeroes,
    /// which is the state `vec![0.0; num_fregs]` used to hand over. Only the
    /// allocation is saved, not the initialization.
    pub fn clean_interp_seeded_f_in(
        program: &Code,
        init_regs: &[i64],
        num_fregs: usize,
        banks: &mut Banks,
    ) -> i64 {
        let regs = &mut banks.regs;
        regs.resize(init_regs.len(), 0);
        regs.copy_from_slice(init_regs);
        let fregs = &mut banks.fregs;
        fregs.clear();
        fregs.resize(num_fregs, 0.0);
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
                OP_MUL_IMM => {
                    regs[program[pc + 3] as usize] =
                        regs[program[pc + 1] as usize] * program[pc + 2];
                    pc += 4;
                }
                OP_ADD_IMM => {
                    regs[program[pc + 3] as usize] =
                        regs[program[pc + 1] as usize] + program[pc + 2];
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
                OP_COL_STORE => {
                    majit_raw_store_i64(
                        regs[program[pc + 1] as usize],
                        regs[program[pc + 2] as usize],
                        regs[program[pc + 3] as usize],
                    );
                    pc += 4;
                }
                OP_COL_STORE_F => {
                    majit_raw_store_i64(
                        regs[program[pc + 1] as usize],
                        regs[program[pc + 2] as usize],
                        majit_f64_to_bits(fregs[program[pc + 3] as usize]),
                    );
                    pc += 4;
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
                    regs[program[pc + 3] as usize] = majit_raw_load_i64(base, ea);
                    pc += 4;
                }
                OP_COL_LOAD_B => {
                    let base = regs[program[pc + 1] as usize];
                    let ea = regs[program[pc + 2] as usize];
                    regs[program[pc + 3] as usize] = majit_raw_load_u8(base, ea);
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
                OP_U2F => {
                    fregs[program[pc + 2] as usize] = regs[program[pc + 1] as usize] as u64 as f64;
                    pc += 3;
                }
                OP_F2I => {
                    regs[program[pc + 2] as usize] = fregs[program[pc + 1] as usize] as i64;
                    pc += 3;
                }
                OP_F2U => {
                    regs[program[pc + 2] as usize] = fregs[program[pc + 1] as usize] as u64 as i64;
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
    /// once.
    ///
    /// The install takes no program. `build_meta` is generated to ignore both
    /// of its arguments — it fills `__JitMeta` from the state's array lengths
    /// alone (`majit-macros` `codegen_state.rs` `build_meta_fields`) — and a
    /// driver here serves every program of its shape, so there is no program
    /// this could meaningfully be handed. Passing the batch that happened to
    /// arrive first would only make the driver look program-specific.
    fn new_driver_f(
        threshold: u32,
        num_regs: usize,
        num_fregs: usize,
    ) -> majit_metainterp::JitDriver<VmStateF> {
        // No quasi-immutable state exists here — a fixed batch program over plain
        // integer and float reds. The periodic loop-invalidation timer this used
        // to opt out of via `with_options(threshold, false)` no longer exists:
        // majit dropped it in favour of revalidating quasi-immutable fields at
        // the guard, so `new` is now the whole constructor.
        let mut driver: majit_metainterp::JitDriver<VmStateF> =
            majit_metainterp::JitDriver::new(threshold);
        // `JitDriver::new` sets only the back-edge threshold; the counter the
        // function-entry door consults is a second one, and it defaults to
        // upstream's `rlib/jit.py:589 function_threshold = 1619` — over 200x
        // this tier's own knob, so a door left on the default never opens under
        // any measurement this crate takes.
        //
        // Both are set to `threshold` rather than kept in upstream's 1619:1039
        // ratio because the tier has ONE knob and every caller spells its
        // intention with it: `threshold_for(Tier::Interpreter)` is `u32::MAX`,
        // meaning never, and a scaled value cannot say that. Same number, two
        // counters, keyed on two different greens.
        driver.set_param("function_threshold", threshold as i64);
        // Diagnostic override for the JIT's `retrace_limit`, unset by default.
        //
        // `unroll.py:213-220` branches on this limit to decide what a bridge
        // whose state matches no existing target does: below the limit it
        // retraces a new specialised loop, at or above it retries with forced
        // boxes and jumps to the preamble. Which arm ran is not otherwise
        // observable from outside the tier, and the two behave very differently
        // on a batch whose inner trip count differs from the traced one
        // (`tests/majit_shape_change.rs`, `examples/poison.rs`).
        //
        // The default stays majit's `DEFAULT_RETRACE_LIMIT`, which is upstream's
        // shipped `rlib/jit.py:595 retrace_limit: 0` — the 5 at
        // `warmspot.py:93` is `jittify_and_run`, the test harness. This is an
        // instrument, NOT a tuning parameter: raising it to move a benchmark
        // would be tuning away from upstream.
        if let Ok(v) = std::env::var("CEL_RETRACE_LIMIT") {
            if let Ok(n) = v.parse::<i64>() {
                driver.set_param("retrace_limit", n);
            }
        }
        // Every driver, not just the pooled ones: the counters are shared and
        // the callbacks are not, so a driver that skipped this would run
        // uncounted and read as a tier that never fired.
        Census::install(&mut driver);
        let seed = VmStateF {
            regs: majit_metainterp::virt_array::VirtArray::filled(0, num_regs),
            fregs: majit_metainterp::virt_array::VirtArray::filled(0.0, num_fregs),
            ret: 0,
        };
        {
            use majit_metainterp::JitState as _;
            seed.build_meta(0, &[])
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
        let mut driver = new_driver_f(threshold, init_regs.len(), num_fregs);
        // No entry door here, and the driver is the reason: it dies with the
        // call, so an entry artifact this call minted could never be entered by
        // another one. The door is on [`run_jit_persistent_f`], whose driver
        // outlives the call.
        let result = run_mainloop_from_start_f(&mut driver, program, init_regs, num_fregs);
        // This driver may have published the per-thread state-field store, and
        // it is about to die. Telling the pool who holds the store now is what
        // keeps its skip-the-republish stamp honest — see
        // [`DriverPool::published_epoch`].
        let epoch = driver.state_field_fvc_epoch();
        if epoch != 0 {
            DRIVERS.with(|d| d.borrow_mut().published_epoch = epoch);
        }
        // Nothing to drain before this driver dies: bridges and compile panics
        // reach the census through `set_on_compile_bridge` and
        // `set_on_internal_compile_panic`, which fired while it ran.
        result
    }

    /// How many distinct programs one pooled driver keeps alive before it is
    /// flushed whole.
    ///
    /// ⚠ This bound is **not** the upstream mechanism, and it should not be read
    /// as the port. Upstream needs no such cap because it bounds the population
    /// two other ways, and majit has neither working:
    ///
    /// * JitCells die, because the memory manager retires loops. majit builds it
    ///   with `max_age` 0 (`warmstate.rs:476 MemoryManager::new(0)`, against
    ///   `rpython/rlib/jit.py:594`'s default of 1000), which parks the ageing
    ///   gate where it can never fire. Everything in `memmgr.rs` is faithful;
    ///   the gate is correctly implementing "disabled".
    /// * A cell holds its token **weakly** (`warmstate.py:157-199`
    ///   `wref_procedure_token`), so a retired loop drops out from under it.
    ///   `BaseJitCell.token` is an `Arc`, not a `Weak`.
    ///
    /// So this is a bound cel has to supply itself, standing in for two absent
    /// invariants, and it stops being necessary the moment either is restored.
    pub const MAX_PROGRAMS_PER_DRIVER: usize = 256;

    /// One pooled driver together with the programs its compiled loops are
    /// keyed on.
    ///
    /// The pairing is the point. The green key is the program **pointer** plus
    /// pc, so a compiled loop that outlives the program its key names is a
    /// loaded gun: the next allocation at that address inherits the loop and
    /// answers an earlier expression's question. Holding the words here means
    /// the address cannot be recycled while a key can still name it, and a flush
    /// drops the loops and the words it keyed them on in one statement.
    ///
    /// This is the half that IS orthodox. Upstream's green key names a code
    /// object the language runtime keeps alive independently of the JIT; this is
    /// that arrangement at the granularity cel can reach, since majit's cell
    /// table lives inside `WarmEnterState` and is not reachable from here. It
    /// errs conservative: a driver holds every program of its register shape,
    /// where upstream would hold one per cell.
    struct PooledDriver {
        driver: majit_metainterp::JitDriver<VmStateF>,
        /// In first-sighting order, so a caller that has already resolved one
        /// reaches it by index. Appended to and never reordered or removed:
        /// the whole driver is flushed at once ([`MAX_PROGRAMS_PER_DRIVER`]),
        /// which is what keeps an index from outliving what it names.
        programs: Vec<PooledProgram>,
        /// Address to index into `programs`. Keyed by address, which is exactly
        /// the identity that must not be reused; two binds of one program
        /// collapse to one entry. Consulted on a first sighting only.
        by_addr: std::collections::HashMap<usize, usize>,
        /// The register banks a call runs on, kept between calls so seeding
        /// them is two memcpys rather than two allocations. See
        /// [`reseed_state_f`].
        ///
        /// Reached only through the checked-out driver, so a re-entrant call —
        /// which by construction gets a driver of its own — cannot alias it.
        state: VmStateF,
    }

    /// One program the pool holds alive, with what the entry door needs to know
    /// about it.
    ///
    /// Every field but `words` is a pure function of the words, and the words
    /// cannot change under an address the pool is pinning, so all of it is
    /// computed on the first sighting and read on every call after.
    struct PooledProgram {
        /// The words. Held for exactly as long as a compiled loop can be keyed
        /// on their address — see [`PooledDriver`].
        words: std::sync::Arc<Code>,
        /// [`loop_header_keys`] for those words — the keys of the doors that
        /// are NOT this one, which the entry door yields to.
        loop_keys: Vec<PooledGreenKey>,
        /// [`green_key_at`] at [`ENTRY_PC`] — the key the function-entry door
        /// files under. The hash folds the program POINTER, so it is a property
        /// of this pinned address and not of the words alone.
        entry_key: PooledGreenKey,
    }

    impl PooledProgram {
        fn new(program: &std::sync::Arc<Code>) -> Self {
            PooledProgram {
                words: std::sync::Arc::clone(program),
                loop_keys: loop_header_keys(program),
                entry_key: green_key_at(program, ENTRY_PC),
            }
        }
    }

    impl PooledDriver {
        fn new(threshold: u32, num_regs: usize, num_fregs: usize) -> Self {
            PooledDriver {
                driver: new_driver_f(threshold, num_regs, num_fregs),
                programs: Vec::new(),
                by_addr: std::collections::HashMap::new(),
                state: VmStateF {
                    regs: majit_metainterp::virt_array::VirtArray::filled(0, num_regs),
                    fregs: majit_metainterp::virt_array::VirtArray::filled(0.0, num_fregs),
                    ret: 0,
                },
            }
        }

        /// The index of `program` in [`PooledDriver::programs`], pinning the
        /// words on a first sighting.
        ///
        /// Pinning has to happen before the run, not after: the loop this call
        /// may compile is keyed on the address, so the address has to be held
        /// by the time the key is taken.
        fn intern(&mut self, program: &std::sync::Arc<Code>, addr: usize) -> usize {
            if let Some(&index) = self.by_addr.get(&addr) {
                return index;
            }
            self.programs.push(PooledProgram::new(program));
            let index = self.programs.len() - 1;
            self.by_addr.insert(addr, index);
            index
        }
    }

    /// Byte width of the value [`run_jit_persistent_f`] moves out of `DRIVERS`
    /// and back into it on every call.
    ///
    /// Exposed because that pair of moves is a candidate for the per-call cost
    /// of the persistent tier, and the candidate cannot be sized without this
    /// number. It is a `size_of`, so it counts only what is INLINE: every heavy
    /// component named in `JitDriver`/`MetaInterp` is a heap handle -- the
    /// counter's timetable is a `Vec`, the warm-enter state's cells are an
    /// index map, cranelift's module and builder context are headers -- so a
    /// small answer here refutes the memcpy reading rather than confirming it.
    ///
    /// It is now a pointer width, because the map holds the driver behind a
    /// `Box`. Compare against [`pooled_driver_inline_bytes`], which is what
    /// this used to report and what the move used to copy.
    pub fn pooled_driver_bytes() -> usize {
        core::mem::size_of::<Box<PooledDriver>>()
    }

    /// Inline width of the driver itself -- what the take-and-reinsert pair
    /// copied per call before the `Box`, twice.
    ///
    /// Kept so the diagnostic still reports the cost that was removed rather
    /// than only the one that remains; a bare "8 bytes moved" reads as though
    /// there had never been anything to remove.
    pub fn pooled_driver_inline_bytes() -> usize {
        core::mem::size_of::<PooledDriver>()
    }

    /// This thread's pooled drivers, plus what a call needs to reach one
    /// without hashing anything.
    #[derive(Default)]
    struct DriverPool {
        /// One per `(regs_len, num_fregs, threshold)` ever asked for. Appended
        /// to and never reordered, so an index names the same shape for as long
        /// as the pool lives; [`reset_persistent_state`] clears the whole thing
        /// and the memo with it.
        entries: Vec<PoolEntry>,
        /// Where the last call resolved to. Most callers run one expression on
        /// one tier over and over, and for them this answers the whole lookup:
        /// a shape scan and an address hash both become four word compares.
        /// A caller alternating between two expressions misses it every call
        /// and pays what every call used to pay.
        last: Option<Resolved>,
        /// Epoch of the state-field store this thread last published, or `0`
        /// before anything did.
        ///
        /// The store is one slot per thread ([`JitDriver::republish_state_field_fvc`]),
        /// so a driver has to re-aim it at itself before it can compile against
        /// another driver's jitcodes. This records whose it is, so the call is
        /// made only when the answer is "not ours". Sound because every route
        /// that writes the slot writes it for a driver a run of this pool is
        /// holding: `register_dispatch_jitcode` at the first trace start,
        /// `force_start_tracing` and `start_bridge_tracing` after that. Epochs
        /// come from a process-global `fetch_add` and are never reused, so a
        /// dropped driver's epoch can never be matched by a later one.
        published_epoch: u64,
    }

    /// One shape's slot in [`DriverPool`].
    struct PoolEntry {
        key: (usize, usize, u32),
        /// `None` while a call has the driver checked out, after a flush
        /// emptied the slot, and before the first call of this shape built one.
        /// [`DriverPool::check_out`] says why those are one state and not three.
        driver: Option<Box<PooledDriver>>,
    }

    /// A resolved `(shape, threshold, program address)`, as the two indices the
    /// per-call path actually walks.
    #[derive(Clone, Copy)]
    struct Resolved {
        key: (usize, usize, u32),
        addr: usize,
        /// Index into [`DriverPool::entries`].
        entry: usize,
        /// Index into that driver's [`PooledDriver::programs`].
        program: usize,
    }

    std::thread_local! {
        /// Drivers kept across calls, keyed by the state shape they were built
        /// for and the threshold they compile at.
        ///
        /// The compiled loop lives in the driver, so a driver per call is a
        /// recompile per call. RPython keeps it on the greens-keyed JitCell
        /// instead (`warmstate.py:157-199` `wref_procedure_token`, held for
        /// `max_age` generations by `memmgr.py:23-69`) and never recompiles per
        /// invocation. The threshold is part of the key so the interpreter tier
        /// (`u32::MAX`) can never pick up the JIT tier's compiled loop.
        /// Boxed because `run_jit_persistent_f` takes the entry OUT of the pool
        /// for the duration of the run and puts it back after, so the slot's
        /// value type is copied twice per call. Behind a `Box` that pair moves
        /// a pointer instead of the whole `JitDriver`, which is inline-large
        /// (`pooled_driver_inline_bytes`) even though its heavy components are
        /// heap handles. The take-and-reinsert shape is unchanged -- this is
        /// the same re-entrancy behaviour, not a different one.
        static DRIVERS: core::cell::RefCell<DriverPool> =
            core::cell::RefCell::new(DriverPool::default());
    }

    /// Drop this thread's persistent drivers, so the next batch traces and
    /// compiles from cold.
    ///
    /// One statement is enough because a driver owns both halves of the pairing:
    /// majit's side is reached by value (`JitDriver` holds `MetaInterp`,
    /// `jitdriver.rs:1184`, which holds `WarmEnterState` with its cell table and
    /// `compiled_loops`, `pyjitpl.rs:1091`), and the programs its keys name are
    /// held in the same [`PooledDriver`]. There is no side table to forget.
    ///
    /// That is the property to preserve, not an implementation detail. The
    /// `#[jit_interp]` green key is the program **pointer** plus pc
    /// (`trace_ctx.rs` `green_key_raw`), and a typed green key stores that
    /// pointer as a *number* (`pyjitpl.rs:4404` `with_typed_decision_key`), so
    /// two programs at one address produce identical keys and `comparekey` finds
    /// them **equal**. That is a true collision in the identity, not a hash
    /// collision a comparison can resolve, and it answers an earlier
    /// expression's question rather than crashing. Anything that frees program
    /// words without dropping the loops keyed on them reopens it.
    pub fn reset_persistent_state() {
        DRIVERS.with(|d| {
            let mut pool = d.borrow_mut();
            pool.entries.clear();
            // The memo names an entry and a program by index, and both indices
            // die with the entries. Dropping it is not an optimisation detail:
            // a surviving memo would name a program slot in a driver that no
            // longer exists.
            pool.last = None;
        });
    }

    /// How many EXTRA times each stage of [`run_jit_persistent_f`] runs before
    /// the call does its own.
    ///
    /// A measurement probe, never a shipping feature, and the third of its kind
    /// in this file after [`LoopKeyArm`]. What it itemises is the fixed cost of
    /// arriving in compiled code -- `batch::JIT_ENTRY_PS`, ~100 ns whole --
    /// across the named stages the call is made of, so the ranking of those
    /// stages is a difference taken inside ONE binary.
    ///
    /// REPETITION and not a clock, because `Instant::now()` costs 20-25 ns on
    /// the box these were taken on: six of them inside a 100 ns budget would
    /// report the clock. Time the call with every count at zero, time it again
    /// with one count at `k`, and the difference over `k` is that stage --
    /// everything the two arms share cancels out of it.
    ///
    /// Read ONCE per call rather than once per stage, so the read is a constant
    /// both arms pay. It does mean no arm's ABSOLUTE figure is what the
    /// shipping door costs: a build carrying this feature pays that read and
    /// five zero-trip loop tests per call that a default build does not have.
    /// Only differences are claims.
    #[cfg(feature = "__entry-stage-probe")]
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct EntryStageRepeats {
        /// Extra [`DRIVERS`] accesses.
        ///
        /// A LOWER bound on that stage rather than a measurement of it: a
        /// thread-local resolves to the same address on every access in a
        /// thread, so a compiler may compute it once for the whole loop. What
        /// repetition cannot elide is the initialization check `with` makes --
        /// [`DRIVERS`] has a non-const initializer -- nor the reload the
        /// barrier in each pass forces.
        pub tls: u32,
        /// Extra pool ROUND TRIPS: [`DriverPool::check_out`] and
        /// [`DriverPool::check_in`] as a pair.
        ///
        /// A pair, and not two counts, because the halves are not separately
        /// repeatable. `check_out` TAKES the driver out of its slot, so a
        /// second one in a row finds the slot empty, answers `pooled: None`,
        /// and the caller builds a fresh driver -- a cold start with a dispatch
        /// jitcode registration in it, which is not this stage and is orders
        /// larger than it. Checking the driver back in before the next take is
        /// what makes the pair idempotent.
        pub pool: u32,
        /// Extra [`reseed_state_f`] calls.
        pub reseed: u32,
        /// Extra yield scans -- `has_runnable_compiled_loop` on the resolved
        /// key of every loop header the program has, which is what the entry
        /// door opens with.
        ///
        /// Always the RESOLVED form, so under `__loop-key-arm-probe` this prices
        /// the shipping scan whichever arm the door itself is taking.
        pub loop_keys: u32,
        /// Extra passes of the yield scan's WALK half alone: `key.resolve`,
        /// with neither table lookup after it.
        ///
        /// `resolve_cell_key` answers an empty or single-cell bucket from the
        /// walk alone, so on every bucket a real program here has produced
        /// this is one celltable walk and nothing else -- no allocation, and
        /// the answer it returns is the raw hash.
        pub loop_walk: u32,
        /// Extra passes of the yield scan's TOKEN half: `has_compiled_loop`,
        /// which is `entry_procedure_token(..).is_some()` -- a celltable
        /// lookup, a `Weak::upgrade`, the `invalidated` and `has_compiled_code`
        /// loads, and the drop of the `Arc` the upgrade produced.
        ///
        /// Asked on `key.hash`, not on `key.resolve`, so that the walk this
        /// splits out is not paid twice and the arm prices the lookup alone.
        /// The two agree on every bucket holding one cell or none -- the same
        /// equivalence `LoopKeyArm` is built on.
        pub loop_token: u32,
        /// Extra passes of the yield scan's UPGRADE half:
        /// `probe_cell_token_upgrades` -- the celltable lookup, the
        /// `Weak::upgrade`, and the drop of the `Arc` it produced, with NO
        /// flag read after it.
        ///
        /// The one arm that is not expressible from majit's shipping API:
        /// `get_procedure_token` performs the refcount pair AND reads
        /// `invalidated` inside one function, so [`Self::loop_token`] fuses
        /// two separable costs. Differenced against it, this leaves the two
        /// flag loads -- and that difference is what decides whether the
        /// refcount pair or the atomic loads is the thing to attack.
        ///
        /// ⚠ Its answer is deliberately WEAKER than the door's: an
        /// invalidated token still upgrades. A cost probe, never a decision.
        pub loop_upgrade: u32,
        /// Extra passes of the yield scan's META half: `get_compiled_meta`,
        /// one `compiled_loops` hash and probe. Asked on `key.hash` for the
        /// same reason as [`Self::loop_token`].
        pub loop_meta: u32,
        /// Extra passes of an amplification loop with NO stage in it: the
        /// counter, and the one optimization barrier every other field's loop
        /// also carries. Subtracting it is what leaves a stage's own cost
        /// rather than its cost plus the machinery that made it repeat.
        pub barrier: u32,
    }

    #[cfg(feature = "__entry-stage-probe")]
    std::thread_local! {
        /// The counts the next call on this thread runs with.
        static ENTRY_STAGE_REPEATS: core::cell::Cell<EntryStageRepeats> = const {
            core::cell::Cell::new(EntryStageRepeats {
                tls: 0,
                pool: 0,
                reseed: 0,
                loop_keys: 0,
                loop_walk: 0,
                loop_token: 0,
                loop_upgrade: 0,
                loop_meta: 0,
                barrier: 0,
            })
        };
    }

    #[cfg(feature = "__entry-stage-probe")]
    std::thread_local! {
        /// How many loop keys the yield scan last walked.
        ///
        /// The divisor a per-key figure needs, published by the door rather
        /// than recounted outside it, where a caller would have to rediscover
        /// which backward jumps [`loop_header_keys`] admits. Written only while
        /// the scan is being amplified, so the arm the difference is taken
        /// against does not carry the store.
        static ENTRY_STAGE_LOOP_KEYS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };

        /// Passes run by the sub-arms of the yield scan, in the order
        /// `[walk, token, upgrade, meta]`, plus a fifth slot that is NOT a
        /// pass count -- see below.
        ///
        /// The first four are what prove an arm is REACHED rather than merely
        /// armed. A sub-arm that compiled to nothing would still leave its
        /// repeat count set and its difference indistinguishable from noise; a
        /// counter the arm's own loop increments cannot be satisfied that way.
        ///
        /// ⭐ Slot 4 is the CHAINED-BUCKET count: how many times
        /// `key.resolve(driver)` answered something other than `key.hash`.
        /// `PooledGreenKey::resolve` answers an empty or single-cell bucket
        /// from the walk alone and returns the raw hash, so on every bucket a
        /// real program here has produced this stays ZERO -- which is what
        /// licenses asking the `token`, `upgrade` and `meta` arms on
        /// `key.hash` instead of paying the walk again inside each of them.
        /// That licence is an assertion about this corpus, so it is COUNTED
        /// rather than asserted in prose: if a bucket ever chains, the number
        /// says so and the three arms stop being about the same cell the door
        /// asks about.
        static ENTRY_STAGE_SUB_PASSES: core::cell::Cell<[u64; 5]> =
            const { core::cell::Cell::new([0; 5]) };
    }

    /// Set the repeat counts for subsequent calls on this thread, answering
    /// what they were.
    ///
    /// A setter rather than a wrapper around one call, unlike
    /// [`run_jit_persistent_probe_f`]: the entry's whole-cost figure is read
    /// through `BoundBatch::collect_into_on`, so the stages have to be read
    /// through it too, or the parts and the whole are about different doors.
    #[cfg(feature = "__entry-stage-probe")]
    pub fn set_entry_stage_repeats(repeats: EntryStageRepeats) -> EntryStageRepeats {
        ENTRY_STAGE_REPEATS.with(|slot| slot.replace(repeats))
    }

    /// The loop-key count the last amplified scan walked. Zero until one has
    /// run.
    #[cfg(feature = "__entry-stage-probe")]
    pub fn entry_stage_loop_keys() -> usize {
        ENTRY_STAGE_LOOP_KEYS.with(core::cell::Cell::get)
    }

    /// Passes run by the yield scan's `[walk, token, upgrade, meta]` sub-arms
    /// since the last [`reset_entry_stage_sub_passes`], and in slot 4 the
    /// number of loop keys whose `resolve` disagreed with their raw hash --
    /// zero on every bucket holding one cell or none.
    #[cfg(feature = "__entry-stage-probe")]
    pub fn entry_stage_sub_passes() -> [u64; 5] {
        ENTRY_STAGE_SUB_PASSES.with(core::cell::Cell::get)
    }

    /// Zero the sub-arm pass counters, so one arm's reachability is read
    /// against the calls made under it and not against every call so far.
    #[cfg(feature = "__entry-stage-probe")]
    pub fn reset_entry_stage_sub_passes() {
        ENTRY_STAGE_SUB_PASSES.with(|slot| slot.set([0; 5]));
    }

    /// [`run_jit_seeded_f`] on a driver that outlives the call, so a program
    /// already compiled by an earlier call runs compiled from its first row.
    ///
    /// Takes the program by `Arc` rather than by slice because outliving the
    /// call is exactly the problem: the driver files a compiled loop under this
    /// program's address, so it must hold the words for as long as it holds the
    /// loop. A `&Code` here would let the caller drop the words while the key
    /// naming them stayed live. See [`PooledDriver`].
    pub fn run_jit_persistent_f(
        program: &std::sync::Arc<Code>,
        init_regs: &[i64],
        num_fregs: usize,
        threshold: u32,
    ) -> i64 {
        let key = (init_regs.len(), num_fregs, threshold);
        let addr = program.as_ptr() as usize;
        // Read once per call, ahead of every stage, so no stage's difference
        // carries it. See [`EntryStageRepeats`]; a default build has neither
        // this read nor the five loops it feeds.
        #[cfg(feature = "__entry-stage-probe")]
        let repeats = ENTRY_STAGE_REPEATS.with(core::cell::Cell::get);
        #[cfg(feature = "__entry-stage-probe")]
        for _ in 0..repeats.tls {
            DRIVERS.with(|cell| {
                std::hint::black_box(cell as *const core::cell::RefCell<DriverPool>);
            });
        }
        // ONE thread-local access for the whole call. The borrow inside it is
        // taken twice and held across neither run: a re-entrant call has to
        // find the pool readable rather than panicking on the `RefCell`.
        DRIVERS.with(|cell| {
            // The pool round trip, amplified as a PAIR. See
            // [`EntryStageRepeats::pool`] for why the halves cannot be repeated
            // apart -- a bare second `check_out` prices a cold start.
            #[cfg(feature = "__entry-stage-probe")]
            for _ in 0..repeats.pool {
                let Checkout {
                    pooled,
                    home,
                    program_index,
                    ..
                } = cell.borrow_mut().check_out(key, program);
                // An empty slot is a first sighting of this shape, and building
                // a driver here would price that rather than the round trip.
                // Leave it to the call below, where a first sighting belongs.
                let Some(pooled) = pooled else { break };
                let keep = pooled.programs.len() <= MAX_PROGRAMS_PER_DRIVER;
                if let Some(dropped) =
                    cell.borrow_mut()
                        .check_in(key, addr, home, program_index, pooled, keep)
                {
                    // The pool declined the driver it had just handed out, so
                    // the next pass would price a rebuild. Its tallies are
                    // already counted — every one of them arrived by callback
                    // while it ran — so there is nothing to drain here.
                    let _ = dropped;
                    break;
                }
            }
            // ── check out ──────────────────────────────────────────────────
            let Checkout {
                pooled,
                home,
                mut program_index,
                published_epoch,
            } = cell.borrow_mut().check_out(key, program);
            // Outside the borrow, because building a driver registers a
            // dispatch jitcode, which writes the per-thread store this pool
            // also reads. Only a first call of a shape, or one the slot could
            // not serve, reaches it.
            let mut pooled = match pooled {
                Some(pooled) => pooled,
                None => {
                    let mut fresh = Box::new(PooledDriver::new(threshold, key.0, key.1));
                    program_index = fresh.intern(program, addr);
                    fresh
                }
            };
            // Split so the entry door can read this program's cached keys while
            // it drives the driver; the three are separate fields and their
            // borrows do not overlap.
            let PooledDriver {
                driver,
                programs,
                state,
                ..
            } = &mut *pooled;
            let pooled_program = &programs[program_index];
            // The store majit decodes guard and resume metadata through is one
            // slot per thread, written when a driver registers its dispatch
            // jitcode. We keep a driver per shape, so aim it back at this one
            // before it can compile anything: another shape's store decodes at
            // the same pcs and returns a mistyped frame count rather than
            // failing. Skipped when this driver is the one that wrote it — see
            // [`DriverPool::published_epoch`] for why that is decidable here.
            if driver.state_field_fvc_epoch() != published_epoch {
                driver.republish_state_field_fvc();
            }
            #[cfg(feature = "__entry-stage-probe")]
            for _ in 0..repeats.reseed {
                reseed_state_f(state, init_regs, num_fregs);
                // Every pass writes the same bytes over the same banks, so
                // without a barrier the passes after the first are dead stores
                // and the loop prices nothing.
                std::hint::black_box(&mut *state);
            }
            // The same loop and the same barrier with no stage in them, so what
            // the amplification itself costs is subtracted rather than reported
            // as a stage.
            #[cfg(feature = "__entry-stage-probe")]
            for _ in 0..repeats.barrier {
                std::hint::black_box(&mut *state);
            }
            reseed_state_f(state, init_regs, num_fregs);
            // The yield scan the entry door opens with. Amplified here rather
            // than inside [`try_function_entry_jit_f`] because both of its
            // inputs are already in hand at this point, which leaves that
            // door's signature the shipping one.
            #[cfg(feature = "__entry-stage-probe")]
            if repeats.loop_keys != 0 {
                ENTRY_STAGE_LOOP_KEYS.with(|slot| slot.set(pooled_program.loop_keys.len()));
                for _ in 0..repeats.loop_keys {
                    let yielded = pooled_program.loop_keys.iter().any(|key| {
                        let probe = key.resolve(driver);
                        driver.has_runnable_compiled_loop(probe)
                    });
                    // Reading the answer is what keeps the walk, and the
                    // barrier inside that read is what stops the next pass
                    // reusing this one's bucket loads.
                    std::hint::black_box(yielded);
                }
            }
            // The three halves the scan above is made of, each amplified on its
            // own so which of them carries the stage is a measurement rather
            // than a reading of the code.
            //
            // All three are pure reads and answer the same thing every pass, so
            // all three amplify honestly -- none of the single-shot treatment
            // `E3c` needs applies here.
            //
            // ⚠ These are three INDEPENDENT arms, not a cumulative ladder, so
            // `walk + token + meta` against the whole scan is a CHECK on the
            // split rather than an assumption behind it. It is expected to be
            // close, not exact: the shipping scan short-circuits, asking
            // `get_compiled_meta` only where a token was found, whereas the
            // `meta` arm asks unconditionally. Warm, where a token is always
            // present, the short-circuit never fires and the two agree.
            #[cfg(feature = "__entry-stage-probe")]
            if repeats.loop_walk != 0 {
                ENTRY_STAGE_LOOP_KEYS.with(|slot| slot.set(pooled_program.loop_keys.len()));
                ENTRY_STAGE_SUB_PASSES.with(|slot| {
                    let mut p = slot.get();
                    p[0] += u64::from(repeats.loop_walk);
                    slot.set(p);
                });
                for _ in 0..repeats.loop_walk {
                    for key in &pooled_program.loop_keys {
                        // The driver goes through the barrier too, not just the
                        // answer. Both inputs are loop-invariant, so barriering
                        // only the result would let the walk be computed once
                        // and the same value handed to the barrier every pass --
                        // an arm that counts its passes and prices nothing. All
                        // four sub-arms carry it, so it cancels between them.
                        let resolved = key.resolve(std::hint::black_box(&*driver));
                        // The licence for the other three arms to ask on
                        // `key.hash`, counted rather than assumed. Inside the
                        // arm that already pays the walk, so no other arm and
                        // no barrier arm carries the comparison.
                        if resolved != key.hash {
                            ENTRY_STAGE_SUB_PASSES.with(|slot| {
                                let mut p = slot.get();
                                p[4] += 1;
                                slot.set(p);
                            });
                        }
                        std::hint::black_box(resolved);
                    }
                }
            }
            #[cfg(feature = "__entry-stage-probe")]
            if repeats.loop_token != 0 {
                ENTRY_STAGE_LOOP_KEYS.with(|slot| slot.set(pooled_program.loop_keys.len()));
                ENTRY_STAGE_SUB_PASSES.with(|slot| {
                    let mut p = slot.get();
                    p[1] += u64::from(repeats.loop_token);
                    slot.set(p);
                });
                for _ in 0..repeats.loop_token {
                    for key in &pooled_program.loop_keys {
                        std::hint::black_box(
                            std::hint::black_box(&*driver).has_compiled_loop(key.hash),
                        );
                    }
                }
            }
            #[cfg(feature = "__entry-stage-probe")]
            if repeats.loop_upgrade != 0 {
                ENTRY_STAGE_LOOP_KEYS.with(|slot| slot.set(pooled_program.loop_keys.len()));
                ENTRY_STAGE_SUB_PASSES.with(|slot| {
                    let mut p = slot.get();
                    p[2] += u64::from(repeats.loop_upgrade);
                    slot.set(p);
                });
                for _ in 0..repeats.loop_upgrade {
                    for key in &pooled_program.loop_keys {
                        std::hint::black_box(
                            std::hint::black_box(&*driver).probe_cell_token_upgrades(key.hash),
                        );
                    }
                }
            }
            #[cfg(feature = "__entry-stage-probe")]
            if repeats.loop_meta != 0 {
                ENTRY_STAGE_LOOP_KEYS.with(|slot| slot.set(pooled_program.loop_keys.len()));
                ENTRY_STAGE_SUB_PASSES.with(|slot| {
                    let mut p = slot.get();
                    p[3] += u64::from(repeats.loop_meta);
                    slot.set(p);
                });
                for _ in 0..repeats.loop_meta {
                    for key in &pooled_program.loop_keys {
                        std::hint::black_box(
                            std::hint::black_box(&*driver)
                                .get_compiled_meta(key.hash)
                                .is_some(),
                        );
                    }
                }
            }
            // The call-counted door, ahead of the first instruction. It answers
            // the whole call when the entry key already has compiled code;
            // otherwise it hands back where interpretation is to pick up.
            let mut pc = ENTRY_PC;
            let result =
                match try_function_entry_jit_f(driver, program, pooled_program, state, &mut pc) {
                    Some(value) => value,
                    None => run_mainloop_f(driver, program, state, pc),
                };
            // ── check in ───────────────────────────────────────────────────
            // Wholesale over the bound, and that is the whole point: the loops
            // and the words they are keyed on go together, so no key can
            // survive the address it names. Dropping either alone is the defect
            // ([`MAX_PROGRAMS_PER_DRIVER`] says why cel has to bound this at
            // all).
            let keep = pooled.programs.len() <= MAX_PROGRAMS_PER_DRIVER;
            let dropped = cell
                .borrow_mut()
                .check_in(key, addr, home, program_index, pooled, keep);
            drop(dropped);
            result
        })
    }

    /// What [`DriverPool::check_out`] hands one call.
    struct Checkout {
        /// The pooled driver, or `None` when the slot held none and the caller
        /// has to build one.
        pooled: Option<Box<PooledDriver>>,
        /// The slot the driver goes back to.
        home: usize,
        /// Index into the driver's programs. Meaningless without `pooled`.
        program_index: usize,
        /// [`DriverPool::published_epoch`] as of the checkout.
        published_epoch: u64,
    }

    impl DriverPool {
        /// Take this shape's driver out of the pool for the duration of a run.
        ///
        /// An empty slot is the only signal there is, and it means two things at
        /// once: nobody has built a driver of this shape yet, or one is checked
        /// out. That conflation is deliberate — both answers are "build one" —
        /// and it is what makes the pool survive a run that panics: an
        /// unwinding call simply never checks its driver back in, and the next
        /// call of that shape rebuilds. A busy FLAG would instead survive the
        /// unwind and mark the slot occupied by a run that no longer exists.
        fn check_out(
            &mut self,
            key: (usize, usize, u32),
            program: &std::sync::Arc<Code>,
        ) -> Checkout {
            let addr = program.as_ptr() as usize;
            let published_epoch = self.published_epoch;
            let memo = self
                .last
                .filter(|last| last.key == key && last.addr == addr);
            let home = match memo {
                Some(last) => last.entry,
                None => match self.entries.iter().position(|e| e.key == key) {
                    Some(entry) => entry,
                    None => {
                        self.entries.push(PoolEntry { key, driver: None });
                        self.entries.len() - 1
                    }
                },
            };
            let mut pooled = self.entries[home].driver.take();
            // Resolving the program is a hash only on a first sighting; the
            // memo answers every call after, and holds because `programs` is
            // append-only for as long as the driver lives.
            let program_index = match (&memo, pooled.as_mut()) {
                (Some(last), Some(_)) => last.program,
                (None, Some(pooled)) => pooled.intern(program, addr),
                // No driver to index into. The caller builds one and interns
                // the program itself.
                _ => usize::MAX,
            };
            Checkout {
                pooled,
                home,
                program_index,
                published_epoch,
            }
        }

        /// Put a run's driver back. `Some` means the pool declined it and the
        /// caller now owns it — and its unreported tallies.
        fn check_in(
            &mut self,
            key: (usize, usize, u32),
            addr: usize,
            home: usize,
            program_index: usize,
            pooled: Box<PooledDriver>,
            keep: bool,
        ) -> Option<Box<PooledDriver>> {
            // A driver that published the thread's store is the one holding it
            // now. A driver that never registered published nothing, so the
            // slot still belongs to whoever wrote it last and the stamp stands.
            let epoch = pooled.driver.state_field_fvc_epoch();
            if epoch != 0 {
                self.published_epoch = epoch;
            }
            // Bounds-checked, and the key re-read rather than trusted: a
            // `reset_persistent_state` from inside a run would have emptied the
            // pool under this slot, and refilling a slot that now names another
            // shape is worse than dropping the driver.
            if !keep || !self.entries.get(home).is_some_and(|slot| slot.key == key) {
                // Either way the memo names a program index inside a driver
                // that is about to go.
                self.last = None;
                return Some(pooled);
            }
            self.entries[home].driver = Some(pooled);
            self.last = Some(Resolved {
                key,
                addr,
                entry: home,
                program: program_index,
            });
            None
        }
    }

    /// One reading of the tier's trace census.
    ///
    /// [`majit_metainterp::embed::CensusCounts`] under this crate's own name:
    /// the fields were the same eight, and a second struct restating them is a
    /// place for the two to drift. It carries two more —
    /// `last_ops_after` and `last_loop_body_shape` — which say whether the last
    /// compiled body is a loop at all. `loops_compiled > 0` does not: an empty
    /// dispatch still compiles a trace, one whose whole optimized body is
    /// `Finish()`.
    ///
    /// Every field now shares ONE window. That is the change: `bridges_compiled`
    /// and `internal_compile_panics` used to be read off the drivers instead of
    /// a callback, so their window was "since the last
    /// [`reset_persistent_state`]" while the rest were "since the last
    /// [`reset_jit_stats`]", and a caller that reset only one of the two was
    /// comparing counts taken over different spans.
    pub use majit_metainterp::embed::CensusCounts as JitStats;

    /// Read the tier's counters.
    ///
    /// A total, not a window — the same reading this module always returned.
    /// [`majit_metainterp::embed::Census::begin`] is the windowed form, and it
    /// is deliberately not used here: it holds a process-global mutex for the
    /// window's whole life and is not reentrant, so it cannot be what a
    /// free-standing `jit_stats()` call opens.
    pub fn jit_stats() -> JitStats {
        Census::totals()
    }

    /// Zero every counter, opening a fresh measurement window.
    ///
    /// It no longer matters whether [`reset_persistent_state`] is called with
    /// it. Every counter is now fed by a callback, so none of them lives on a
    /// driver and dropping the drivers neither zeroes nor preserves anything
    /// this reads — which is what used to make a reset of one without the other
    /// return two different windows in one struct.
    pub fn reset_jit_stats() {
        Census::reset();
    }

    /// Snapshot majit's abort-reason tallies as `(label, count)` pairs.
    ///
    /// [`JitStats::loops_aborted`] counts aborts without saying WHY, and the
    /// reason is not carried on the `on_trace_abort` hook — it survives only in
    /// majit's `MC_DIAG` slots, keyed by `Counters::ABORT_*`. Without this a
    /// test can assert `aborts == 0`, watch it fail, and have no way to learn
    /// which of seven reasons fired.
    ///
    /// `majit-metainterp` is an optional dependency of this crate and is not a
    /// dependency of the test binaries, so the reader has to come through here.
    ///
    /// Selection is by the `abrt_` label prefix, deliberately NOT by the slot
    /// range those labels currently occupy: a hard-coded range names the wrong
    /// counters the moment a slot is added, and it does so silently.
    ///
    /// ⛔ THIS CENSUS HAS TWO VALUES, NOT SEVEN. `AbortReason` defines only
    /// `TooLong` and `Generic`, and `Generic.as_int()` is `ABORT_BRIDGE`, so
    /// every abort that is not a length/tag overflow lands in `abrt_bridge`
    /// regardless of cause. `abrt_bridge` is the UNCLASSIFIED bucket wearing a
    /// specific-sounding name; `jitprof.rs` says so outright — "reading it as
    /// 'a bridge aborted' sends the next reader looking for bridge activity
    /// that is not there". `stage_abort_reason` would let a caller name a real
    /// reason and currently has zero callers, which is why nothing is ever
    /// classified. So [`abort_reasons_since`] relabels that slot in its output
    /// rather than repeating a name that misdescribes itself.
    ///
    /// ⚠ `MC_DIAG` is process-global, cumulative, and has no reset. Diff two
    /// snapshots with [`abort_reasons_since`] rather than reading one directly.
    pub use majit_metainterp::embed::abort_reasons;

    /// Render the abort reasons that fired since `before`, as `label=delta`.
    ///
    /// `abrt_bridge` is printed as `unclassified(abrt_bridge)` because that is
    /// what it measures — see [`abort_reasons`]. The original label is kept in
    /// the parentheses so the output still names the slot it came from.
    ///
    /// Empty string when nothing aborted, so a caller can print it
    /// unconditionally and a quiet window stays quiet.
    pub fn abort_reasons_since(before: &[(&'static str, u64)]) -> String {
        majit_metainterp::embed::render_abort_delta(before, &abort_reasons())
    }

    /// Render majit's per-guard deopt census as `distinct=N total=M top=…`.
    ///
    /// `guard_failures` is a total, and a total cannot separate the two things
    /// that produce a large one: ONE guard that keeps returning to the runtime
    /// because no bridge was ever attached to it, versus a WIDE SPREAD of cold
    /// guards that each fail below `trace_eagerness`. Those want opposite fixes
    /// — the first is a bridging defect, the second is a warmup cost that a
    /// budget mis-sized. `distinct` is what tells them apart.
    ///
    /// ⚠ Off unless `MAJIT_GUARD_CENSUS` is set, because the map write sits on
    /// the deopt path, i.e. exactly the path under study. Returns
    /// `guard_census=off` when it was never enabled — a third value distinct
    /// from an empty census, so "no rows" cannot be misread as "no deopts".
    ///
    /// ⚠ Cumulative, process-global, and there is no reset. Read the INCREMENT
    /// across a window rather than the absolute, the way a sweep over several
    /// shapes has to.
    pub fn guard_census_summary(top: usize) -> String {
        majit_metainterp::guard_census_summary(top)
    }

    /// Print one `[jit-stats]` line, tagged with `label`, in the format the pyre
    /// runner uses. `label` names the shape being measured, so a run that sweeps
    /// several of them stays readable.
    pub fn print_jit_stats(label: &str) {
        eprintln!("[jit-stats] {label} {}", jit_stats());
    }
}
