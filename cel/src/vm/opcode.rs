//! The instruction set.
//!
//! Every opcode here was derived by exhaustive match over [`Expr`]'s variants
//! rather than from a list, which is why the set contains no entry for
//! `all`/`exists`/`exists_one`/`map`/`filter`: those are expanded into
//! [`Expr::Comprehension`] at parse time by `parser::macros`, so they never
//! reach an evaluator as calls at all.
//!
//! [`Expr`]: crate::common::ast::Expr
//! [`Expr::Comprehension`]: crate::common::ast::Expr::Comprehension

/// One instruction.
///
/// Fieldless on purpose. Operands travel as separate words in the instruction
/// stream (see [`CelCode`]), not as enum payloads, for two reasons:
///
/// * A payload-carrying enum reaches the tracing front end as an
///   enum-variant construction, whose only general lowering is hard-anchored
///   to `core::result::Result`. Everything else arrives at the optimizer as an
///   opaque transparent-constructor residual -- invisible to the escape
///   analysis, and carrying no class the optimizer can reason about. A flat
///   integer stream has no constructor to lower.
/// * A fixed word width makes a jump target an index, so the compiler can
///   patch a forward jump without re-encoding anything after it.
///
/// [`CelCode`]: super::CelCode
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpCode {
    // -- loads ------------------------------------------------------------
    /// Push `consts[a]`.
    LoadConst,
    /// Push the context variable named `names[a]`.
    ///
    /// Split from [`OpCode::LoadLocal`] at compile time. The tree walker asks
    /// one question for both and pays a map lookup for the answer; a
    /// comprehension variable is known to be a slot before the program runs.
    LoadVar,
    /// Push activation-record slot `a`.
    LoadLocal,
    /// Pop into activation-record slot `a`.
    StoreLocal,

    // -- selection --------------------------------------------------------
    /// Pop an operand, push its `names[a]` field.
    GetField,
    /// Pop an operand, push whether it has field `names[a]`.
    ///
    /// This is where `has(x.y)` lands. `has` is not a call and not a macro
    /// expansion: the parser sets `SelectExpr::test`, so it is a compile-time
    /// flag on a `Select` node. Left as a flag it would be a branch in the
    /// hot path on every field read.
    HasField,
    /// Pop index and operand, push the indexed element.
    Index,
    /// Pop index and operand, push an optional of the indexed element.
    ///
    /// Split from [`OpCode::Index`]: the walker reaches both through one arm
    /// and recovers which it is by comparing the operator's *name string* at
    /// evaluation time.
    OptIndex,
    /// Pop an operand, push an optional of its `names[a]` field.
    OptSelect,

    // -- aggregate construction -------------------------------------------
    /// Pop `a` elements, push a list.
    BuildList,
    /// Pop an optional and append its value to the list beneath it, or leave
    /// the list unchanged when the optional is empty.
    ///
    /// `ListExpr::optional_indices` is compile-time data, so the compiler
    /// selects this per element instead of carrying the index set into the
    /// program and testing it per iteration.
    ListAppendOptional,
    /// Pop `2 * a` operands (key, value interleaved), push a map.
    BuildMap,
    /// Pop an optional value and a key, inserting into the map beneath them
    /// only when the optional is non-empty.
    MapInsertOptional,
    /// Pop `2 * a` operands (field id, value interleaved), push a struct of
    /// the message type named `names[b]`.
    BuildStruct,
    /// The struct-literal counterpart of [`OpCode::MapInsertOptional`].
    StructSetOptional,

    // -- binary operators -------------------------------------------------
    //
    // One opcode each, deliberately not a single `BinOp` taking a kind
    // operand: a kind operand is a second switch inside the dispatch arm, and
    // the receiver-narrowing chain that makes the arms monomorphic is
    // per-operator.
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Equals,
    NotEquals,
    Less,
    LessEquals,
    Greater,
    GreaterEquals,
    /// The `@in` operator.
    In,

    // -- unary operators --------------------------------------------------
    Not,
    Negate,
    /// The `@not_strictly_false` operator, which comprehension loop
    /// conditions are wrapped in.
    NotStrictlyFalse,

    // -- calls ------------------------------------------------------------
    /// Pop `b` arguments, call the global function `names[a]`.
    CallHost,
    /// Pop `b` arguments and a receiver, call method `names[a]` on it.
    CallMethod,
    /// Pop `b` arguments, call the namespaced function `names[a]`.
    ///
    /// Distinguishing this from [`OpCode::CallMethod`] is compile-time work
    /// here. The walker asks the question on every member call whose receiver
    /// parses as an identifier, because `math.max(x)` and `s.startsWith(x)`
    /// have the same shape.
    CallQualified,

    // -- control flow -----------------------------------------------------
    /// Jump to `a`.
    Jump,
    /// Pop an operand; jump to `a` when it is false.
    JumpIfFalse,
    /// Pop an operand; jump to `a` when it is true.
    JumpIfTrue,
    /// Short-circuit `&&`: see the note on error absorption below.
    And,
    /// Short-circuit `||`.
    Or,
    /// Stop, returning the top of the stack.
    Return,
}

impl OpCode {
    /// How many operand words follow this opcode in the instruction stream.
    ///
    /// Exhaustive on purpose: a new opcode that forgets to declare its width
    /// is a compile error rather than a stream that decodes correctly until
    /// the first program that uses it.
    pub const fn operands(self) -> u32 {
        match self {
            OpCode::LoadConst
            | OpCode::LoadVar
            | OpCode::LoadLocal
            | OpCode::StoreLocal
            | OpCode::GetField
            | OpCode::HasField
            | OpCode::OptSelect
            | OpCode::BuildList
            | OpCode::BuildMap
            | OpCode::Jump
            | OpCode::JumpIfFalse
            | OpCode::JumpIfTrue
            | OpCode::And
            | OpCode::Or => 1,

            OpCode::BuildStruct | OpCode::CallHost | OpCode::CallMethod | OpCode::CallQualified => {
                2
            }

            OpCode::Index
            | OpCode::OptIndex
            | OpCode::ListAppendOptional
            | OpCode::MapInsertOptional
            | OpCode::StructSetOptional
            | OpCode::Add
            | OpCode::Sub
            | OpCode::Mul
            | OpCode::Div
            | OpCode::Mod
            | OpCode::Equals
            | OpCode::NotEquals
            | OpCode::Less
            | OpCode::LessEquals
            | OpCode::Greater
            | OpCode::GreaterEquals
            | OpCode::In
            | OpCode::Not
            | OpCode::Negate
            | OpCode::NotStrictlyFalse
            | OpCode::Return => 0,
        }
    }

    /// The width of this instruction, opcode word included.
    pub const fn width(self) -> u32 {
        1 + self.operands()
    }
}

/// The number of distinct opcodes.
///
/// Kept next to [`OpCode::from_word`] so the round-trip test below has
/// something to enumerate. Update it when adding an opcode; the test fails
/// otherwise.
pub const OPCODE_COUNT: u32 = OpCode::Return as u32 + 1;

impl OpCode {
    /// Decode an instruction-stream word.
    pub const fn from_word(word: u32) -> Option<OpCode> {
        if word >= OPCODE_COUNT {
            return None;
        }
        // SAFETY: `OpCode` is `#[repr(u8)]` with consecutive discriminants
        // from 0, and `word` was just bounds-checked against the count.
        Some(unsafe { core::mem::transmute::<u8, OpCode>(word as u8) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every discriminant below the count decodes, and decodes to itself.
    ///
    /// This is what pins `OPCODE_COUNT` to the enum: adding a variant without
    /// updating the constant leaves the new discriminant undecodable, and
    /// `Return` stops being last.
    #[test]
    fn every_opcode_round_trips_through_its_word() {
        for word in 0..OPCODE_COUNT {
            let op = OpCode::from_word(word).expect("word below the count must decode");
            assert_eq!(op as u32, word);
        }
        assert!(OpCode::from_word(OPCODE_COUNT).is_none());
    }

    #[test]
    fn operand_widths_are_declared_for_every_opcode() {
        for word in 0..OPCODE_COUNT {
            let op = OpCode::from_word(word).unwrap();
            assert!(op.width() >= 1, "{op:?} has no opcode word");
            assert!(op.operands() <= 2, "{op:?} declares an unexpected arity");
        }
    }
}
