//! The JIT portal for [`super::interp::cel_eval_loop`].
//!
//! `interp_jit.py` `PyPyJitDriver`: greens are `(next_instr, code)`, the
//! red virtualizable is `frame`. `jit_merge_point` is the first statement
//! of the loop; `can_enter_jit` is only on a backward jump.
//!
//! Opcode bodies stay in [`super::interp::Vm::dispatch_one`] as a residual
//! call so the portal can attach without inlining `Result` into the
//! traced graph. Inlining interned arms is the next step, not a
//! prerequisite for the doors.

use majit_metainterp::JitDriver;

use super::code::CelCode;
use super::interp::{Step, Vm};
#[allow(unused_imports)] // named in `virtualizable_fields`
use crate::runtime::object::{
    W_CelFrame, CELFRAME_LAST_INSTR_OFFSET, CELFRAME_LOCALS_STACK_OFFSET,
    CELFRAME_VABLE_TOKEN_OFFSET, CELFRAME_VALUESTACKDEPTH_OFFSET,
};
#[allow(unused_imports)]
use crate::runtime::object_array::{CEL_ITEMS_BLOCK_ITEMS_OFFSET, CEL_ITEMS_BLOCK_LEN_OFFSET};
use crate::{ExecutionError, Value};

/// `dispatch_one` finished the program.
const PORTAL_DONE: i64 = -1;
/// `dispatch_one` failed with no handler.
const PORTAL_FAIL: i64 = -2;

struct PortalState {
    frame: usize,
    vm: i64,
    ret: i64,
}

/// One opcode as an integer, residual so the portal does not index a `Vec`.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn insn_op(program: &CelCode, pc: usize) -> i64 {
    program
        .insns
        .get(pc)
        .map(|insn| insn.op as u8 as i64)
        .unwrap_or(-1)
}

/// One instruction of the existing evaluator, residual.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn residual_dispatch(vm_bits: i64, pc: i64) -> i64 {
    let vm = unsafe { &mut *(vm_bits as usize as *mut Vm<'_>) };
    match vm.dispatch_one(pc as u32) {
        Ok(Step::Next) => pc + 1,
        Ok(Step::Jump(target)) => i64::from(target),
        Ok(Step::Return(value)) => {
            vm.portal_ret = Some(Ok(value));
            PORTAL_DONE
        }
        Err(err) => {
            vm.portal_ret = Some(Err(err));
            PORTAL_FAIL
        }
    }
}

/// Evaluate `code` through the portal loop.
pub(crate) fn eval_through_portal(
    vm: &mut Vm<'_>,
    code: &CelCode,
) -> Result<Value, ExecutionError> {
    let mut state = PortalState {
        frame: vm.cel_frame as usize,
        vm: vm as *mut Vm<'_> as i64,
        ret: 0,
    };
    // High threshold: the doors are attached; a compile is not required
    // for the native portal loop to answer. Census is not installed here
    // — that hook is process-global and would clobber other machines.
    let mut driver: JitDriver<PortalState> = JitDriver::new(1_000_000);
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, code)
            .install_canonical_liveness(&mut driver);
    }
    let bits = run_cel_portal(&mut driver, code, &mut state, 0);
    match bits {
        PORTAL_DONE => match vm.portal_ret.take() {
            Some(Ok(value)) => Ok(value),
            Some(Err(err)) => Err(vm.public_error(err)),
            None => Err(ExecutionError::InternalError(
                "portal done without a result".into(),
            )),
        },
        PORTAL_FAIL => match vm.portal_ret.take() {
            Some(Err(err)) => Err(vm.public_error(err)),
            other => Err(ExecutionError::InternalError(format!(
                "portal fail without an error: {other:?}"
            ))),
        },
        _ => Err(ExecutionError::InternalError(format!(
            "portal returned unexpected pc {bits}"
        ))),
    }
}

#[majit_macros::jit_interp(
    state = PortalState,
    env = CelCode,
    greens = [pc, program],
    state_fields = {
        frame: ref(W_CelFrame),
        vm: int,
        ret: int,
    },
    virtualizable_fields = {
        var: frame,
        token_offset: CELFRAME_VABLE_TOKEN_OFFSET,
        fields: {
            last_instr: int @ CELFRAME_LAST_INSTR_OFFSET,
            valuestackdepth: int @ CELFRAME_VALUESTACKDEPTH_OFFSET,
        },
        arrays: {
            locals_stack_w: ref @ CELFRAME_LOCALS_STACK_OFFSET {
                ptr_offset: 0,
                length_offset: CEL_ITEMS_BLOCK_LEN_OFFSET,
                items_offset: CEL_ITEMS_BLOCK_ITEMS_OFFSET,
            },
        },
    },
    auto_calls = true,
    calls = {
        insn_op => residual_int,
        residual_dispatch => residual_int,
    },
)]
fn run_cel_portal(
    mut driver: &mut JitDriver<PortalState>,
    program: &CelCode,
    state: &mut PortalState,
    mut pc: usize,
) -> i64 {
    loop {
        jit_merge_point!(driver, program, pc; *state);
        let opcode = insn_op(program, pc);
        let vm = state.vm as i64;
        let here = pc as i64;
        let next = match opcode {
            0 => residual_dispatch(vm, here),
            1 => residual_dispatch(vm, here),
            2 => residual_dispatch(vm, here),
            3 => residual_dispatch(vm, here),
            4 => residual_dispatch(vm, here),
            5 => residual_dispatch(vm, here),
            6 => residual_dispatch(vm, here),
            7 => residual_dispatch(vm, here),
            8 => residual_dispatch(vm, here),
            9 => residual_dispatch(vm, here),
            10 => residual_dispatch(vm, here),
            11 => residual_dispatch(vm, here),
            12 => residual_dispatch(vm, here),
            13 => residual_dispatch(vm, here),
            14 => residual_dispatch(vm, here),
            15 => residual_dispatch(vm, here),
            16 => residual_dispatch(vm, here),
            17 => residual_dispatch(vm, here),
            18 => residual_dispatch(vm, here),
            19 => residual_dispatch(vm, here),
            20 => residual_dispatch(vm, here),
            21 => residual_dispatch(vm, here),
            22 => residual_dispatch(vm, here),
            23 => residual_dispatch(vm, here),
            24 => residual_dispatch(vm, here),
            25 => residual_dispatch(vm, here),
            26 => residual_dispatch(vm, here),
            27 => residual_dispatch(vm, here),
            28 => residual_dispatch(vm, here),
            29 => residual_dispatch(vm, here),
            30 => residual_dispatch(vm, here),
            31 => residual_dispatch(vm, here),
            32 => residual_dispatch(vm, here),
            33 => residual_dispatch(vm, here),
            34 => residual_dispatch(vm, here),
            35 => residual_dispatch(vm, here),
            36 => residual_dispatch(vm, here),
            37 => residual_dispatch(vm, here),
            38 => residual_dispatch(vm, here),
            39 => residual_dispatch(vm, here),
            40 => residual_dispatch(vm, here),
            41 => residual_dispatch(vm, here),
            42 => residual_dispatch(vm, here),
            43 => residual_dispatch(vm, here),
            44 => residual_dispatch(vm, here),
            45 => residual_dispatch(vm, here),
            46 => residual_dispatch(vm, here),
            47 => residual_dispatch(vm, here),
            48 => residual_dispatch(vm, here),
            49 => residual_dispatch(vm, here),
            50 => residual_dispatch(vm, here),
            51 => residual_dispatch(vm, here),
            52 => residual_dispatch(vm, here),
            53 => residual_dispatch(vm, here),
            54 => residual_dispatch(vm, here),
            55 => residual_dispatch(vm, here),
            56 => residual_dispatch(vm, here),
            57 => residual_dispatch(vm, here),
            58 => residual_dispatch(vm, here),
            59 => residual_dispatch(vm, here),
            60 => residual_dispatch(vm, here),
            61 => residual_dispatch(vm, here),
            62 => residual_dispatch(vm, here),
            63 => residual_dispatch(vm, here),
            64 => residual_dispatch(vm, here),
            65 => residual_dispatch(vm, here),
            66 => residual_dispatch(vm, here),
            67 => residual_dispatch(vm, here),
            68 => residual_dispatch(vm, here),
            69 => residual_dispatch(vm, here),
            70 => residual_dispatch(vm, here),
            71 => residual_dispatch(vm, here),
            72 => residual_dispatch(vm, here),
            73 => residual_dispatch(vm, here),
            74 => residual_dispatch(vm, here),
            75 => residual_dispatch(vm, here),
            76 => residual_dispatch(vm, here),
            77 => residual_dispatch(vm, here),
            78 => residual_dispatch(vm, here),
            79 => residual_dispatch(vm, here),
            80 => residual_dispatch(vm, here),
            81 => residual_dispatch(vm, here),
            82 => residual_dispatch(vm, here),
            83 => residual_dispatch(vm, here),
            84 => residual_dispatch(vm, here),
            85 => residual_dispatch(vm, here),
            86 => residual_dispatch(vm, here),
            87 => residual_dispatch(vm, here),
            88 => residual_dispatch(vm, here),
            89 => residual_dispatch(vm, here),
            _ => residual_dispatch(vm, here),
        };
        if next < 0 {
            state.ret = next;
            break;
        }
        let tgt = next as usize;
        if tgt < pc {
            can_enter_jit!(driver, tgt, &mut *state, program, || {});
        }
        pc = tgt;
    }
    state.ret
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::parser::Parser;
    use crate::vm::compile::compile;
    use crate::vm::interp::cel_eval_loop;

    #[test]
    fn the_portal_evaluates_the_same_as_the_public_door() {
        let expr = Parser::default().parse("1 + 2").unwrap();
        let code = compile(&expr).expect("compile");
        let ctx = Context::default();
        let via_loop = cel_eval_loop(&code, &ctx).expect("eval");
        assert_eq!(via_loop, Value::Int(3));
    }
}
