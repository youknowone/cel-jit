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
    //
    // Uniformly incremental: allocate empty, then one append per element.
    // The alternative -- a single `BuildList n` that pops its elements at
    // once -- needs a second, different encoding as soon as one element is
    // optional, because whether an element contributes is only known once it
    // has been evaluated. One shape that always works beats two shapes and a
    // rule for choosing between them.
    /// Push an empty list.
    NewList,
    /// Pop a value and append it to the list beneath it.
    ListAppend,
    /// Pop an optional; append its value to the list beneath it, or leave the
    /// list unchanged when the optional is empty.
    ///
    /// `ListExpr::optional_indices` is compile-time data, so the compiler
    /// selects between this and [`OpCode::ListAppend`] per element instead of
    /// carrying the index set into the program and testing it per iteration.
    ListAppendOptional,
    /// Push an empty map.
    NewMap,
    /// Pop a value and a key, inserting them into the map beneath them.
    MapInsert,
    /// Pop an optional value and a key, inserting only when the optional is
    /// non-empty.
    MapInsertOptional,
    /// Push an empty struct of the message type named `names[a]`.
    NewStruct,
    /// Pop a value into field `names[a]` of the struct beneath it.
    StructSet,
    /// Pop an optional into field `names[a]` of the struct beneath it,
    /// leaving the field unset when the optional is empty.
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
    /// Try the namespaced function `names[a]` over `b` arguments: on a hit,
    /// pop them, push the result and jump to `c`; on a miss, fall through to
    /// the receiver-call path the compiler emitted after it.
    ///
    /// Either way the `b` arguments come off the stack here. A miss hands them
    /// straight to that [`OpCode::CallMethod`] rather than pushing them back
    /// for it to pop again, so the receiver it loads in between is the only
    /// thing that instruction finds on the stack. The compiler's depth model
    /// still counts them as pushed, which leaves `max_stack` an upper bound
    /// and nothing else.
    ///
    /// `names[a]` is the *joined* name, built once at compile time. The
    /// walker asks this question on every member call whose receiver parses
    /// as an identifier -- `math.max(x)` and `s.startsWith(x)` have the same
    /// shape -- and joining the two names per evaluation is an allocation.
    ///
    /// The miss path must not have evaluated the receiver yet, because
    /// `optional.of(1)` names no variable `optional`.
    CallQualified,

    // -- iteration --------------------------------------------------------
    //
    // A comprehension is the only construct that loops, and these are what
    // let it do so without a host call per iteration: `size(x)` reached
    // through `CallHost` would be a residual call inside the loop body.
    /// Pop a value, push the sequence a one-variable comprehension iterates:
    /// a list's elements, or a map's keys.
    IterElems,
    /// Pop a value, push the sequence a two-variable comprehension iterates
    /// as its first variable: a list's *indices*, or a map's keys.
    ///
    /// Distinct from [`OpCode::IterElems`] only for lists, where
    /// `xs.all(i, v, ...)` binds `i` to the index and `v` to the element.
    IterKeys,
    /// Push the length of the list in slot `a`. Nothing is popped.
    ///
    /// The sequence is named by SLOT rather than taken from the stack because
    /// the loop that emits this runs it once per element, and reaching a slot
    /// through [`OpCode::LoadLocal`] copies what is in it: for a list that is
    /// a `ListRef` clone, whose `Arc` refcount is an atomic increment matched
    /// by a decrement when this instruction drops the copy again. Reading the
    /// slot in place costs neither, and the `LoadLocal` itself stops being
    /// emitted at all.
    IterLen,
    /// Push the element of the sequence in slot `a` at the index in slot `b`.
    /// Nothing is popped.
    ///
    /// Both inputs are named by slot for the reason [`OpCode::IterLen`] gives;
    /// between them the two accounted for four atomic refcount operations per
    /// element, on one shared count.
    ///
    /// The index is produced by the compiler's own counter, so it is in range
    /// by construction rather than by a check.
    IterAt,

    // -- control flow -----------------------------------------------------
    /// Jump to `a`.
    Jump,
    /// Pop an operand; jump to `a` when it is false.
    JumpIfFalse,
    /// Jump to `a` when the top operand is an empty optional, leaving it on
    /// the stack.
    ///
    /// An optional container answers a plain index on its own -- `opt_none[k]`
    /// is `optional.none` -- and it answers *before* the key is evaluated, so
    /// `opt_none[1 / 0]` does not divide. That order is unobservable in a tree
    /// walker, which reaches the container first because it recurses; a flat
    /// stream has to say it.
    ///
    /// The operand is left in place rather than replaced with a fresh
    /// `optional.none`, because an empty optional is what the result is.
    JumpIfOptNone,
    /// Pop an operand; jump to `a` when it is true.
    JumpIfTrue,
    /// The left half of `&&`: pop the left operand into logic slot `a`. When
    /// it is exactly `false`, push `false` and jump to `b`; otherwise fall
    /// through to the right operand.
    ///
    /// A non-bool operand is not a short circuit -- it is recorded as an
    /// overload failure and decided at the merge, because `true && 1` and
    /// `1 && true` are both errors while `error && false` is `false`.
    And,
    /// The left half of `||`, mirroring [`OpCode::And`]: short-circuits on
    /// `true`.
    Or,
    /// The merge of `&&`: pop the right operand, combine it with logic slot
    /// `a`, push the result or raise.
    ///
    /// This is where CEL's commutativity over errors lives. A left-hand error
    /// is *discarded* when the right operand decides the result, so the merge
    /// has to be able to turn a recorded error into a successful `false` --
    /// which is why the left half records rather than raises.
    AndMerge,
    /// The merge of `||`, mirroring [`OpCode::AndMerge`].
    OrMerge,
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
            | OpCode::NewStruct
            | OpCode::StructSet
            | OpCode::StructSetOptional
            | OpCode::Jump
            | OpCode::JumpIfFalse
            | OpCode::JumpIfOptNone
            | OpCode::JumpIfTrue
            | OpCode::AndMerge
            | OpCode::OrMerge
            | OpCode::IterLen => 1,

            OpCode::CallHost | OpCode::CallMethod | OpCode::And | OpCode::Or | OpCode::IterAt => 2,

            OpCode::CallQualified => 3,

            OpCode::Index
            | OpCode::OptIndex
            | OpCode::NewList
            | OpCode::ListAppend
            | OpCode::ListAppendOptional
            | OpCode::NewMap
            | OpCode::MapInsert
            | OpCode::MapInsertOptional
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
            | OpCode::IterElems
            | OpCode::IterKeys
            | OpCode::Return => 0,
        }
    }

    /// The width of this instruction, opcode word included.
    pub const fn width(self) -> u32 {
        1 + self.operands()
    }

    /// How many operands this instruction pops and pushes, on the path that
    /// falls through to the next instruction.
    ///
    /// Exhaustive for the same reason [`OpCode::operands`] is: the compiler's
    /// `max_stack` walk is only as trustworthy as this table, and an opcode
    /// added without a declared effect would silently under-size the operand
    /// stack rather than fail to build.
    ///
    /// `operands` is the instruction's operand words, because a call's arity
    /// is one of them.
    pub fn stack_effect(self, operands: &[u32]) -> (u32, u32) {
        // Argument counts live in the second operand word of the call forms.
        let arity = |index: usize| operands.get(index).copied().unwrap_or(0);
        match self {
            OpCode::LoadConst | OpCode::LoadVar | OpCode::LoadLocal => (0, 1),
            // Both name their inputs by slot, so neither pops anything.
            OpCode::IterLen | OpCode::IterAt => (0, 1),
            OpCode::NewList | OpCode::NewMap | OpCode::NewStruct => (0, 1),

            OpCode::StoreLocal | OpCode::Return => (1, 0),
            OpCode::ListAppend | OpCode::ListAppendOptional => (1, 0),
            OpCode::StructSet | OpCode::StructSetOptional => (1, 0),
            OpCode::MapInsert | OpCode::MapInsertOptional => (2, 0),

            OpCode::GetField | OpCode::HasField | OpCode::OptSelect => (1, 1),
            OpCode::AndMerge | OpCode::OrMerge => (1, 1),
            OpCode::Not | OpCode::Negate | OpCode::NotStrictlyFalse => (1, 1),
            OpCode::IterElems | OpCode::IterKeys => (1, 1),

            OpCode::Index | OpCode::OptIndex => (2, 1),
            OpCode::Add
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
            | OpCode::In => (2, 1),

            OpCode::CallHost => (arity(1), 1),
            // The receiver is pushed last, above the arguments, because it is
            // evaluated only after the namespaced lookup has missed.
            OpCode::CallMethod => (arity(1) + 1, 1),
            // Declared for the path that *jumps*, which is the one that
            // reaches the merge. The falling-through path leaves the stack
            // untouched, and the compiler restores the depth itself.
            OpCode::CallQualified => (arity(1), 1),

            // Both paths leave the operand where it is: the falling-through
            // one because the index still needs it, the jumping one because
            // an empty optional is the answer.
            OpCode::Jump | OpCode::JumpIfOptNone => (0, 0),
            // Declared for the falling-through path, where the operand moves
            // into the logic slot. The short-circuiting path pushes the
            // answer instead, so both paths reach the merge one deep.
            OpCode::JumpIfFalse | OpCode::JumpIfTrue | OpCode::And | OpCode::Or => (1, 0),
        }
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
            assert!(op.operands() <= 3, "{op:?} declares an unexpected arity");
        }
    }

    /// A call's declared pop count follows its arity operand, which is what
    /// makes the `max_stack` walk correct for calls at all.
    #[test]
    fn call_stack_effects_follow_the_arity_operand() {
        assert_eq!(OpCode::CallHost.stack_effect(&[0, 3]), (3, 1));
        assert_eq!(OpCode::CallQualified.stack_effect(&[0, 0]), (0, 1));
        assert_eq!(OpCode::CallMethod.stack_effect(&[0, 2]), (3, 1));
    }
}
