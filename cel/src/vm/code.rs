//! The code object: a compiled expression.

use super::error::NameId;
use super::opcode::OpCode;
use crate::Value;

/// A compiled CEL expression.
///
/// Sized entirely at compile time. `n_slots` and `max_stack` are what let the
/// activation record be one flat array allocated once per execution rather
/// than a growable stack: CEL has no functions of its own, so there is no
/// frame stack and no depth that depends on the input.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CelCode {
    /// The instruction stream: an opcode word followed by that opcode's
    /// operands, repeated. Jump operands are indices into this vector.
    pub code: Vec<u32>,
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

    /// Decode the instruction at `pc`, returning it with its operands.
    ///
    /// `None` when `pc` is out of range, when the word is not an opcode, or
    /// when the stream is too short to hold the operands the opcode declares
    /// -- a truncated stream is a malformed program, not a shorter one.
    pub fn decode(&self, pc: u32) -> Option<(OpCode, &[u32])> {
        let word = *self.code.get(pc as usize)?;
        let op = OpCode::from_word(word)?;
        let first = pc as usize + 1;
        let end = first + op.operands() as usize;
        if end > self.code.len() {
            return None;
        }
        Some((op, &self.code[first..end]))
    }

    /// Walk the whole stream, yielding `(pc, opcode, operands)`.
    ///
    /// Tooling and tests only. Decoding linearly is valid because every
    /// instruction has a fixed, opcode-determined width, so no operand can be
    /// mistaken for an opcode.
    pub fn instructions(&self) -> impl Iterator<Item = (u32, OpCode, &[u32])> {
        let mut pc = 0u32;
        std::iter::from_fn(move || {
            let (op, operands) = self.decode(pc)?;
            let here = pc;
            pc += op.width();
            Some((here, op, operands))
        })
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

    /// A hand-built stream decodes back to the instructions it encodes,
    /// including the two-operand form.
    #[test]
    fn a_stream_decodes_to_the_instructions_it_encodes() {
        let code = CelCode {
            code: vec![
                OpCode::LoadConst as u32,
                0,
                OpCode::LoadVar as u32,
                1,
                OpCode::Add as u32,
                OpCode::CallHost as u32,
                2,
                3,
                OpCode::Return as u32,
            ],
            consts: vec![Value::Int(1)],
            names: vec!["a".into(), "b".into(), "size".into()],
            n_slots: 0,
            max_stack: 2,
        };

        let decoded: Vec<_> = code
            .instructions()
            .map(|(pc, op, operands)| (pc, op, operands.to_vec()))
            .collect();

        assert_eq!(
            decoded,
            vec![
                (0, OpCode::LoadConst, vec![0]),
                (2, OpCode::LoadVar, vec![1]),
                (4, OpCode::Add, vec![]),
                (5, OpCode::CallHost, vec![2, 3]),
                (8, OpCode::Return, vec![]),
            ]
        );
    }

    /// A stream that ends mid-instruction is malformed, and decoding says so
    /// rather than reading a shorter instruction.
    #[test]
    fn a_truncated_operand_does_not_decode() {
        let code = CelCode {
            // `CallHost` declares two operands and only one follows.
            code: vec![OpCode::CallHost as u32, 0],
            ..CelCode::default()
        };
        assert!(code.decode(0).is_none());
        assert_eq!(code.instructions().count(), 0);
    }

    #[test]
    fn a_word_that_is_not_an_opcode_does_not_decode() {
        let code = CelCode {
            code: vec![u32::MAX],
            ..CelCode::default()
        };
        assert!(code.decode(0).is_none());
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
