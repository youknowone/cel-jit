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
    /// Add one to the integer in activation-record slot `a`, in place. Nothing
    /// is pushed and nothing is popped.
    ///
    /// This is a comprehension's loop counter, and the comprehension lowering
    /// now emits it folded into [`OpCode::IterAdvance`] together with the back
    /// edge that follows it, so no lowering emits it on its own. It stays the
    /// unfolded half of that instruction: the arm is what `IterAdvance` does
    /// to the counter, and both are reachable from a hand-built stream.
    ///
    /// Spelled out, the counter was `LoadLocal a ; LoadConst 1 ; Add ;
    /// StoreLocal a` -- four dispatches, an operand-stack round trip and a
    /// constant-pool entry, to add one to a number the program already owns.
    /// The slot is named directly for the reason [`OpCode::IterLen`] gives,
    /// and the increment is applied where the value already is.
    ///
    /// Unlike [`OpCode::Add`] this removes no reference counting: a counter
    /// and the literal one are both `Value::Int`, which owns nothing. What it
    /// removes is three instructions per element of every comprehension,
    /// unconditionally.
    ///
    /// A slot holding anything but an integer is a malformed program rather
    /// than a typing error the language can express -- the compiler writes
    /// this slot with a zero and then only through this instruction -- so it
    /// is refused the way [`OpCode::IterLen`] refuses a non-list, and overflow
    /// still raises what the `Add` it replaces raised.
    IncLocal,

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
    /// Push field `names[b]` of the value in activation-record slot `a`.
    /// Nothing is popped.
    ///
    /// Spelled out, this was `LoadLocal a ; GetField b`. The container went
    /// onto the operand stack only to be popped off again by the very next
    /// instruction, which then read a field through a reference to it.
    ///
    /// The container is read in place for the reason [`OpCode::IterLen`]
    /// gives, and here that reason is worth more than a dispatch: a record or
    /// a map is an `Arc`, so the copy `LoadLocal` made was an atomic increment
    /// matched by a decrement when this instruction dropped it -- per field
    /// read, on the shared count. Nothing between the two ever needed a
    /// container of its own.
    ///
    /// A comprehension variable is a slot, which is what makes this the shape
    /// of every field read in a loop body.
    GetFieldLocal,
    /// [`OpCode::GetFieldLocal`] for `has(x.y)`: push whether the value in
    /// slot `a` has field `names[b]`.
    ///
    /// `has` is a compile-time flag on the same AST node, so it selects
    /// between these two exactly as it selects between [`OpCode::GetField`]
    /// and [`OpCode::HasField`]; the operand it reads and the way it reads it
    /// are the same.
    HasFieldLocal,

    // -- aggregate construction -------------------------------------------
    //
    // Uniformly incremental: allocate empty, then one append per element.
    // The alternative -- a single `BuildList n` that pops its elements at
    // once -- needs a second, different encoding as soon as one element is
    // optional, because whether an element contributes is only known once it
    // has been evaluated. One shape that always works beats two shapes and a
    // rule for choosing between them.
    /// Push an empty list with room for `a` elements, `BUILD_LIST n` with
    /// the count spent on capacity rather than on popping.
    NewList,
    /// Push an empty list with room for as many elements as the list in slot
    /// `a` holds.
    ///
    /// The comprehension accumulator's opener: `BUILD_LIST_FROM_ARG`, which
    /// sizes the list by the `length_hint` of the range it is about to walk.
    /// A `map` then fills it without regrowing, and a `filter` at worst
    /// leaves some of the capacity unused. `NewList` plus one append per
    /// element regrew the buffer three times for a ten-element range.
    NewListFromArg,
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

    // -- binary operators against a literal ---------------------------------
    //
    // `<lhs> ; LoadConst k ; <BinOp>` is what a predicate against a constant
    // compiles to, and the middle instruction exists only to put the constant
    // where the operator will pop it from. Each of these is that pair, with
    // the constant named by pool index instead.
    //
    // One opcode per operator, and not a single `BinOpConst` carrying a kind
    // operand, for the reason the operators above give: a kind operand is a
    // second switch inside the dispatch arm.
    //
    // The set is deliberately narrow -- the operators a predicate actually
    // writes against a literal. Everything else keeps the pair, which is
    // correct and merely unfused; `compile::const_operator` is the list.
    /// Pop the left operand, push it plus `consts[a]`.
    AddConst,
    /// Pop the left operand, push it times `consts[a]`.
    MulConst,
    /// Pop the left operand, push it modulo `consts[a]`.
    ModConst,
    /// Pop the left operand, push whether it equals `consts[a]`.
    ///
    /// Of the eight forms here, this pair and its twin are the only ones that
    /// also remove the constant's `Value::clone`: equality is decided through
    /// `PartialEq`, which reads both sides through references, where the
    /// arithmetic and ordering forms hand their operands to helpers that take
    /// them by value. A string constant is an `Arc`, so what that saves is an
    /// atomic pair per evaluation -- and a string constant is exactly what an
    /// equality predicate is written against.
    EqualsConst,
    /// Pop the left operand, push whether it differs from `consts[a]`.
    NotEqualsConst,
    /// Pop the left operand, push whether it orders below `consts[a]`.
    LessConst,
    /// Pop the left operand, push whether it orders above `consts[a]`.
    GreaterConst,
    /// Pop the left operand, push whether it orders at or above `consts[a]`.
    GreaterEqualsConst,

    // -- binary operators over a slot and a literal --------------------------
    //
    // The operators above with their LEFT operand named by slot as well, which
    // is what a predicate over a comprehension variable is: `LoadLocal a ;
    // <op>Const b` is the whole body of `xs.map(x, x * 2)`. Nothing but the
    // operator ever looked at the copy the load pushed.
    //
    // Both halves are named, so the load's dispatch and the operand-stack round
    // trip it existed for are gone and the instruction pushes its answer
    // directly. That is what all eight remove, and all of it: what each one
    // then does about COPYING its operands is the individual arm's business.
    //
    // One per operator rather than a kind operand, for the reason the operators
    // above give; the set is exactly the one `compile::const_operator` maps,
    // because a fused form needs the constant folded first.
    /// Push slot `a` plus `consts[b]`. Nothing is popped.
    AddLocalConst,
    /// Push slot `a` times `consts[b]`. Nothing is popped.
    MulLocalConst,
    /// Push slot `a` modulo `consts[b]`. Nothing is popped.
    ModLocalConst,
    /// Push whether slot `a` equals `consts[b]`. Nothing is popped.
    EqualsLocalConst,
    /// Push whether slot `a` differs from `consts[b]`. Nothing is popped.
    NotEqualsLocalConst,
    /// Push whether slot `a` orders below `consts[b]`. Nothing is popped.
    LessLocalConst,
    /// Push whether slot `a` orders above `consts[b]`. Nothing is popped.
    GreaterLocalConst,
    /// Push whether slot `a` orders at or above `consts[b]`. Nothing is
    /// popped.
    GreaterEqualsLocalConst,

    // -- producing straight into a list ------------------------------------
    //
    // `<producer> ; ListAppend` is what appending an element compiles to, and
    // the value travels through the operand stack between them: the producer
    // pushes it and the append takes it straight off again. Each opcode here
    // is that pair, with the value handed to the list builder directly.
    //
    // The builder is NOT an operand of these. It is the [`Operand::List`] the
    // enclosing `NewList` left on the stack and it stays there for the whole
    // aggregate, so every one of these is stack-neutral: it pops nothing,
    // pushes nothing, and mutates what is already on top. That is the same
    // arrangement `ListAppend` itself has, minus the pop.
    //
    // The set is exactly the `Local` producers -- the instructions that name a
    // SLOT and push a finished value -- and `compile::appending_producer` is
    // the list. Naming a slot is what makes a producer worth fusing here: its
    // answer is a function of the element, which is what an append inside a
    // comprehension consumes once per iteration. A `LoadConst` or a `LoadVar`
    // element appends the same value on every iteration, so what such a
    // program wants is not an opcode but a hoist, and it keeps the pair.
    //
    // One opcode per producer rather than a producer operand, for the reason
    // the binary operators give: a kind operand is a second switch inside the
    // dispatch arm. Each name is its producer's plus `Append`, so the twin
    // relation is readable off the two names.
    /// Append slot `a` to the list beneath. Nothing is pushed or popped.
    LoadLocalAppend,
    /// Append field `names[b]` of slot `a` to the list beneath.
    GetFieldLocalAppend,
    /// Append whether slot `a` has field `names[b]` to the list beneath.
    HasFieldLocalAppend,
    /// Append slot `a` plus `consts[b]` to the list beneath.
    AddLocalConstAppend,
    /// Append slot `a` times `consts[b]` to the list beneath.
    MulLocalConstAppend,
    /// Append slot `a` modulo `consts[b]` to the list beneath.
    ModLocalConstAppend,
    /// Append whether slot `a` equals `consts[b]` to the list beneath.
    EqualsLocalConstAppend,
    /// Append whether slot `a` differs from `consts[b]` to the list beneath.
    NotEqualsLocalConstAppend,
    /// Append whether slot `a` orders below `consts[b]` to the list beneath.
    LessLocalConstAppend,
    /// Append whether slot `a` orders above `consts[b]` to the list beneath.
    GreaterLocalConstAppend,
    /// Append whether slot `a` orders at or above `consts[b]` to the list
    /// beneath.
    GreaterEqualsLocalConstAppend,

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
    ///
    /// A list's elements ARE that list, so the popped value is pushed back
    /// unchanged and the loop reads the caller's buffer rather than a private
    /// copy of it. What that costs and why it is safe is at the arm itself.
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
    /// the loop runs this once per element, and reaching a slot through
    /// [`OpCode::LoadLocal`] copies what is in it: for a list that is a
    /// `ListRef` clone, whose `Arc` refcount is an atomic increment matched by
    /// a decrement when this instruction drops the copy again. Reading the
    /// slot in place costs neither, and the `LoadLocal` itself stops being
    /// emitted at all.
    ///
    /// Folded into [`OpCode::IterGuard`], which is the whole comparison this
    /// length was pushed for, so no lowering emits it on its own.
    IterLen,
    /// Push the element of the sequence in slot `a` at the index in slot `b`.
    /// Nothing is popped.
    ///
    /// Both inputs are named by slot for the reason [`OpCode::IterLen`] gives;
    /// between them the two accounted for four atomic refcount operations per
    /// element, on one shared count.
    ///
    /// Folded into [`OpCode::IterBind`], which is this read and the store that
    /// always followed it, so no lowering emits it on its own.
    ///
    /// The compiler emits no bounds test of its own, because the loop guard
    /// that runs immediately before this has already compared the index
    /// against the length. The arm still checks: an instruction stream is
    /// public data, and the buffer underneath is indexed directly. What the
    /// arm does not do is reach the element through the general indexing path,
    /// which would decide the container's kind and the key's kind first.
    IterAt,

    // -- the fused loop -----------------------------------------------------
    //
    // The three groups of a comprehension's per-element block that touch the
    // operand stack without ever needing to: each one puts a value on the
    // stack and takes it off again inside the same group, so the stack is at
    // the same depth before and after. What was travelling through it is a
    // slot's number, a list's length and an element -- all of which the
    // instruction can name directly.
    //
    // The comprehension lowering emits these directly; nothing folds an
    // emitted stream afterwards, for the reason `Compiler::comprehension`
    // gives.
    /// The loop guard: fall through while the counter in slot `a` is below the
    /// length of the list in slot `b`, and jump to `c` when it is not. Nothing
    /// is pushed and nothing is popped.
    ///
    /// Spelled out, this was `LoadLocal a ; IterLen b ; Less ; JumpIfFalse c`
    /// -- four dispatches and three operand-stack round trips to compare two
    /// integers the program already holds.
    ///
    /// Both are integers by construction, which is what lets the comparison be
    /// an `i64` one rather than `compare_values`: the counter slot is written
    /// once, with a zero, before the loop, and thereafter only by
    /// [`OpCode::IterAdvance`], and the length is a list's. Neither fact is
    /// true of the INSTRUCTION STREAM, which is public data anyone can build,
    /// so a counter slot holding anything else is refused the way
    /// [`OpCode::IncLocal`] refuses one and a source slot holding anything else
    /// the way [`OpCode::IterLen`] refuses one.
    IterGuard,
    /// Bind one element: write the element of the list in slot `a` at the
    /// index in slot `b` into slot `c`. Nothing is pushed and nothing is
    /// popped.
    ///
    /// Spelled out, this was `IterAt a b ; StoreLocal c`, which moved the
    /// element onto the operand stack only to take it off again and put it
    /// where it was always going.
    IterBind,
    /// The back edge: add one to the integer in slot `a`, then jump to `b`.
    /// Nothing is pushed and nothing is popped.
    ///
    /// Spelled out, this was `IncLocal a ; Jump b`. Neither half touched the
    /// operand stack, so this is the one group of the three whose whole saving
    /// is a dispatch: one of the two the pair cost.
    IterAdvance,

    // -- the fused accumulator ----------------------------------------------
    //
    // The scaffolding `all` and `exists` carry, which `parser::macros`
    // synthesises around their predicate: a loop condition that reads the
    // accumulator, and a short-circuit operator whose left operand is that
    // same accumulator. Both put a slot's value on the operand stack and take
    // it off again without anything else looking at it.
    //
    // These are what the tree walker does not pay at all -- it reaches the
    // same accumulator as a `Context` binding through its own recursion -- so
    // they are an asymmetry against it rather than shared work.
    /// The loop condition `all` carries: fall through while the accumulator in
    /// slot `a` is not strictly false, and jump to `b` when it is. Nothing is
    /// pushed and nothing is popped.
    ///
    /// Spelled out, this was `LoadLocal a ; NotStrictlyFalse ; JumpIfFalse b`.
    /// The accumulator went onto the operand stack, was replaced there by the
    /// bool it answers to, and came off again -- three dispatches and two
    /// round trips to read one slot.
    ///
    /// The slot is read in place for the reason [`OpCode::IterLen`] gives: a
    /// `LoadLocal` copies what is in it, and the copy existed only because the
    /// value had to travel.
    ///
    /// `@not_strictly_false` answers `true` for a non-bool rather than
    /// failing, so this instruction cannot raise.
    AccuLoopCond,
    /// [`OpCode::AccuLoopCond`] over the NEGATED accumulator, which is the
    /// condition `exists` carries: `LoadLocal a ; Not ; NotStrictlyFalse ;
    /// JumpIfFalse b`.
    ///
    /// Its own opcode rather than a flag operand on [`OpCode::AccuLoopCond`],
    /// for the reason the binary operators give: a kind operand is a second
    /// switch inside the dispatch arm.
    ///
    /// Unlike its twin this CAN raise. The negation is INSIDE the
    /// `@not_strictly_false`, so a non-bool accumulator reaches `!` first and
    /// is an overload failure there rather than a `true` the test would have
    /// answered.
    AccuLoopCondNot,
    /// The left half of a `&&` whose left operand is slot `a`: record that
    /// slot's bool -- or its overload failure -- in logic slot `b`, and when
    /// it is exactly `false`, push `false` and jump to `c`.
    ///
    /// Spelled out, this was `LoadLocal a ; And b c`. It is the shape `all`'s
    /// step has on every element, because the accumulator is a slot; nothing
    /// about it is special to `all`, so the compiler emits it for any `&&`
    /// whose left operand resolves to a slot.
    ///
    /// A non-bool slot is not a short circuit, and is recorded rather than
    /// raised, for the reason [`OpCode::And`] gives.
    AndLocal,
    /// The `||` twin of [`OpCode::AndLocal`], mirroring [`OpCode::Or`]:
    /// short-circuits on `true`.
    OrLocal,

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
            | OpCode::IncLocal
            | OpCode::IterLen
            | OpCode::NewList
            | OpCode::NewListFromArg
            | OpCode::NewMap
            | OpCode::AddConst
            | OpCode::MulConst
            | OpCode::ModConst
            | OpCode::EqualsConst
            | OpCode::NotEqualsConst
            | OpCode::LessConst
            | OpCode::GreaterConst
            | OpCode::GreaterEqualsConst
            | OpCode::LoadLocalAppend => 1,

            OpCode::CallHost
            | OpCode::CallMethod
            | OpCode::And
            | OpCode::Or
            | OpCode::IterAt
            | OpCode::IterAdvance
            | OpCode::AccuLoopCond
            | OpCode::AccuLoopCondNot
            | OpCode::GetFieldLocal
            | OpCode::HasFieldLocal
            | OpCode::AddLocalConst
            | OpCode::MulLocalConst
            | OpCode::ModLocalConst
            | OpCode::EqualsLocalConst
            | OpCode::NotEqualsLocalConst
            | OpCode::LessLocalConst
            | OpCode::GreaterLocalConst
            | OpCode::GreaterEqualsLocalConst
            // The appending twins carry exactly what their producers carry,
            // because `ListAppend` names nothing: the list it appends to is
            // the operand on top of the stack.
            | OpCode::GetFieldLocalAppend
            | OpCode::HasFieldLocalAppend
            | OpCode::AddLocalConstAppend
            | OpCode::MulLocalConstAppend
            | OpCode::ModLocalConstAppend
            | OpCode::EqualsLocalConstAppend
            | OpCode::NotEqualsLocalConstAppend
            | OpCode::LessLocalConstAppend
            | OpCode::GreaterLocalConstAppend
            | OpCode::GreaterEqualsLocalConstAppend => 2,

            OpCode::CallQualified
            | OpCode::IterGuard
            | OpCode::IterBind
            | OpCode::AndLocal
            | OpCode::OrLocal => 3,

            OpCode::Index
            | OpCode::OptIndex
            | OpCode::ListAppend
            | OpCode::ListAppendOptional
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
            // Reads and writes slots; the operand stack is not involved at
            // all, which is the whole reason these opcodes exist. Each fused
            // form is stack-neutral because the group it replaces was: what
            // travelled through the stack was put there and taken off again
            // inside the same group.
            OpCode::IncLocal | OpCode::IterGuard | OpCode::IterBind | OpCode::IterAdvance => (0, 0),
            // Reads one slot and branches on it; neither path touches the
            // operand stack, because the value the three instructions moved
            // through it never left the slot.
            OpCode::AccuLoopCond | OpCode::AccuLoopCondNot => (0, 0),
            OpCode::NewList | OpCode::NewListFromArg | OpCode::NewMap | OpCode::NewStruct => (0, 1),

            OpCode::StoreLocal | OpCode::Return => (1, 0),
            OpCode::ListAppend | OpCode::ListAppendOptional => (1, 0),
            OpCode::StructSet | OpCode::StructSetOptional => (1, 0),
            OpCode::MapInsert | OpCode::MapInsertOptional => (2, 0),

            OpCode::GetField | OpCode::HasField | OpCode::OptSelect => (1, 1),
            // The same read with its operand named by slot instead of taken
            // off the stack, so the pop is gone and the push is not.
            OpCode::GetFieldLocal | OpCode::HasFieldLocal => (0, 1),
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
            // The same operators with the right operand named by pool index
            // instead of taken off the stack, so one of the two pops is gone.
            OpCode::AddConst
            | OpCode::MulConst
            | OpCode::ModConst
            | OpCode::EqualsConst
            | OpCode::NotEqualsConst
            | OpCode::LessConst
            | OpCode::GreaterConst
            | OpCode::GreaterEqualsConst => (1, 1),
            // Both operands named, so the remaining pop is gone too: this is
            // the composition of a `LoadLocal`'s `(0, 1)` with the `(1, 1)`
            // above, and the value that used to travel between them never
            // reaches the stack.
            OpCode::AddLocalConst
            | OpCode::MulLocalConst
            | OpCode::ModLocalConst
            | OpCode::EqualsLocalConst
            | OpCode::NotEqualsLocalConst
            | OpCode::LessLocalConst
            | OpCode::GreaterLocalConst
            | OpCode::GreaterEqualsLocalConst => (0, 1),
            // The composition of one of the producers above with
            // `ListAppend`'s `(1, 0)`: the producer's push and the append's
            // pop were each other's, so the pair's net is what the one
            // instruction declares. The list it appends to is the operand
            // underneath, which it neither pops nor replaces.
            OpCode::LoadLocalAppend
            | OpCode::GetFieldLocalAppend
            | OpCode::HasFieldLocalAppend
            | OpCode::AddLocalConstAppend
            | OpCode::MulLocalConstAppend
            | OpCode::ModLocalConstAppend
            | OpCode::EqualsLocalConstAppend
            | OpCode::NotEqualsLocalConstAppend
            | OpCode::LessLocalConstAppend
            | OpCode::GreaterLocalConstAppend
            | OpCode::GreaterEqualsLocalConstAppend => (0, 0),

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
            // The same declaration one instruction earlier: the operand the
            // falling-through path moved into the logic slot is now read out
            // of an activation-record slot instead, so the falling-through
            // path is stack-neutral and the short-circuiting one still pushes
            // the answer. Both paths reach the merge one deep, as above.
            OpCode::AndLocal | OpCode::OrLocal => (0, 0),
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

    /// Advancing a slot in place is what [`OpCode::IncLocal`] is for, so it
    /// must cost no operand-stack traffic. Declared here rather than inferred
    /// from an emitted program, because it is a property of the instruction.
    #[test]
    fn inc_local_names_a_slot_and_leaves_the_operand_stack_alone() {
        assert_eq!(OpCode::IncLocal.operands(), 1);
        assert_eq!(OpCode::IncLocal.stack_effect(&[0]), (0, 0));
    }

    /// The fused loop instructions name everything they touch, so none of them
    /// costs an operand-stack round trip. Declared here rather than inferred
    /// from an emitted program, because it is the property the fusion is for:
    /// a fused form that still pushed or popped would have removed dispatches
    /// only, and the stack traffic is what the block was measured to be made
    /// of.
    #[test]
    fn the_fused_loop_instructions_leave_the_operand_stack_alone() {
        for (op, operands) in [
            (OpCode::IterGuard, 3),
            (OpCode::IterBind, 3),
            (OpCode::IterAdvance, 2),
        ] {
            assert_eq!(op.operands(), operands, "{op:?}");
            assert_eq!(op.stack_effect(&[0, 0, 0]), (0, 0), "{op:?}");
        }
    }

    /// The slot-and-literal operators name a slot and a pool index, so each
    /// takes two operand words and pushes its answer onto an operand stack it
    /// never reads. Declared here rather than inferred from an emitted
    /// program, for the reason above -- and this is the one of the three where
    /// a wrong declaration is silent in release: `Vm::new` only `reserve`s
    /// `max_stack` and both depth guards are `debug_assert`, so a net effect
    /// declared one low surfaces as `Vm::unwind` truncating below a live
    /// `Operand::List` accumulator rather than as a failure here.
    #[test]
    fn the_slot_and_literal_operators_name_both_operands_and_only_push() {
        for op in [
            OpCode::AddLocalConst,
            OpCode::MulLocalConst,
            OpCode::ModLocalConst,
            OpCode::EqualsLocalConst,
            OpCode::NotEqualsLocalConst,
            OpCode::LessLocalConst,
            OpCode::GreaterLocalConst,
            OpCode::GreaterEqualsLocalConst,
        ] {
            assert_eq!(op.operands(), 2, "{op:?}");
            assert_eq!(op.stack_effect(&[]), (0, 1), "{op:?}");
        }
    }
}
