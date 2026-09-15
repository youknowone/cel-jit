//! The code object: a compiled expression.

use super::error::NameId;
use super::opcode::OpCode;
use crate::Value;

/// One instruction, already decoded.
///
/// The operands are a fixed three-wide array because that is what the
/// dispatch arms take: the widest opcode declares three, and a record that
/// already holds them hands them over without deciding how many there are.
/// Words the opcode does not declare are zero and no arm reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Insn {
    pub op: OpCode,
    pub ops: [u32; 3],
}

/// A compiled CEL expression.
///
/// Sized entirely at compile time. `n_slots` and `max_stack` are what let the
/// activation record be one flat array allocated once per execution rather
/// than a growable stack: CEL has no functions of its own, so there is no
/// frame stack and no depth that depends on the input.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CelCode {
    /// The instructions, one record each. A `pc` -- here, in a jump operand
    /// and in the handler table -- is an index into this vector, so it counts
    /// instructions rather than words and advancing costs no arithmetic over
    /// the opcode's width.
    pub insns: Vec<Insn>,
    /// Literal values, indexed by a `LoadConst` operand.
    pub consts: Vec<Value>,
    /// Identifiers, field names and function names, indexed by [`NameId`].
    pub names: Vec<Box<str>>,
    /// Size of the activation record -- the high-water mark of the compiler's
    /// scope stack, so sibling comprehensions share slots and only nested
    /// ones occupy distinct ranges.
    pub n_slots: u32,
    /// Depth the operand stack reaches, from the abstract-interpretation walk.
    pub max_stack: u32,
    /// How many logic slots the program uses, one per `&&`/`||`.
    pub n_logic: u32,
    /// Where an error raised inside a short-circuit operator's *left* operand
    /// is caught. Innermost match wins.
    pub handlers: Vec<Handler>,
}

/// One entry of the handler table.
///
/// CEL's `&&` and `||` absorb an error on either side when the other side
/// decides the result, so an error raised anywhere inside the left operand
/// has to reach the merge rather than unwind past it. A tree walker gets this
/// by not applying `?` to its recursive call; a flat instruction stream needs
/// to say which instructions the merge is willing to catch for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handler {
    /// First instruction covered.
    pub start: u32,
    /// One past the last instruction covered -- the merge's own left half.
    pub end: u32,
    /// Where to resume: the first instruction of the right operand.
    pub land: u32,
    /// Logic slot the caught error is recorded in.
    pub logic: u32,
    /// Operand-stack depth to restore, which is the depth before the left
    /// operand was evaluated.
    pub depth: u32,
}

impl CelCode {
    /// The name behind a [`NameId`].
    pub fn name(&self, id: NameId) -> Option<&str> {
        self.names.get(id.0 as usize).map(|n| &**n)
    }

    /// The literal behind a `LoadConst` operand.
    pub fn konst(&self, index: u32) -> Option<&Value> {
        self.consts.get(index as usize)
    }

    /// Walk the whole program, yielding `(pc, opcode, operands)`.
    ///
    /// Tooling and tests only. The operands are the words the opcode declares,
    /// so a caller sees the same slice a variable-width stream would have
    /// carried after it.
    pub fn instructions(&self) -> impl Iterator<Item = (u32, OpCode, &[u32])> {
        self.insns
            .iter()
            .enumerate()
            .map(|(pc, insn)| (pc as u32, insn.op, &insn.ops[..insn.op.operands() as usize]))
    }

    /// The innermost handler covering `pc`, if any.
    ///
    /// Innermost wins so that a nested `&&` inside another's left operand
    /// absorbs into its own merge rather than the outer one.
    pub fn handler_for(&self, pc: u32) -> Option<&Handler> {
        self.handlers
            .iter()
            .filter(|h| h.start <= pc && pc < h.end)
            .min_by_key(|h| h.end - h.start)
    }

    /// A one-instruction-per-line rendering, for test assertions and
    /// debugging.
    pub fn disassemble(&self) -> String {
        let mut out = String::new();
        for (pc, op, operands) in self.instructions() {
            out.push_str(&format!("{pc:4}  {op:?}"));
            for operand in operands {
                out.push_str(&format!(" {operand}"));
            }
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two malformed-program cases were tested here and are gone: a word that
    // is not an opcode, and an instruction whose operands the stream ended
    // before. Neither state can be built out of decoded records -- an `Insn`
    // holds an `OpCode` and three operand words -- so there is nothing left to
    // refuse.

    /// A hand-built program walks back as the instructions it holds, with `pc`
    /// counting instructions and each one carrying only the operands its
    /// opcode declares.
    #[test]
    fn a_program_walks_back_as_the_instructions_it_holds() {
        let code = CelCode {
            insns: vec![
                Insn {
                    op: OpCode::LoadConst,
                    ops: [0, 0, 0],
                },
                Insn {
                    op: OpCode::LoadVar,
                    ops: [1, 0, 0],
                },
                Insn {
                    op: OpCode::Add,
                    ops: [0, 0, 0],
                },
                Insn {
                    op: OpCode::CallHost,
                    ops: [2, 3, 0],
                },
                Insn {
                    op: OpCode::Return,
                    ops: [0, 0, 0],
                },
            ],
            consts: vec![Value::Int(1)],
            names: vec!["a".into(), "b".into(), "size".into()],
            max_stack: 2,
            ..CelCode::default()
        };

        let walked: Vec<_> = code
            .instructions()
            .map(|(pc, op, operands)| (pc, op, operands.to_vec()))
            .collect();

        assert_eq!(
            walked,
            vec![
                (0, OpCode::LoadConst, vec![0]),
                (1, OpCode::LoadVar, vec![1]),
                (2, OpCode::Add, vec![]),
                (3, OpCode::CallHost, vec![2, 3]),
                (4, OpCode::Return, vec![]),
            ]
        );
    }

    #[test]
    fn names_and_consts_resolve_by_index() {
        let code = CelCode {
            consts: vec![Value::Int(7)],
            names: vec!["x".into()],
            ..CelCode::default()
        };
        assert_eq!(code.name(NameId(0)), Some("x"));
        assert_eq!(code.name(NameId(1)), None);
        assert_eq!(code.konst(0), Some(&Value::Int(7)));
        assert_eq!(code.konst(1), None);
    }
}
