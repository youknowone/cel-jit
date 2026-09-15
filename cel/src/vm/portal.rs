//! The JIT portal for [`super::interp::cel_eval_loop`].
//!
//! `interp_jit.py` `PyPyJitDriver`: greens are `(next_instr, code)`, the
//! red virtualizable is `frame`. `jit_merge_point` is the first statement
//! of the loop; `can_enter_jit` is only on a backward jump.
//!
//! Interned arithmetic, comparison, local load/store and return run
//! on `frame.locals_stack_w[i]` — the `getarrayitem_vable_*` shape —
//! and call `cel_add` / `cel_equals` with no `Result`. Everything
//! else is residual [`Vm::dispatch_one`].

use majit_metainterp::JitDriver;

use super::code::CelCode;
use super::interp::{Step, Vm};
use super::opcode::OpCode;
use crate::runtime::binop::{
    cel_add, cel_div, cel_equals, cel_greater, cel_greater_equals, cel_less, cel_less_equals,
    cel_mul, cel_not_equals, cel_rem, cel_sub,
};
use crate::runtime::error::ERROR_SENTINEL;
use crate::runtime::object::CelRef;
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

const OP_LOAD_LOCAL: i64 = OpCode::LoadLocal as i64;
const OP_STORE_LOCAL: i64 = OpCode::StoreLocal as i64;
const OP_ADD: i64 = OpCode::Add as i64;
const OP_SUB: i64 = OpCode::Sub as i64;
const OP_MUL: i64 = OpCode::Mul as i64;
const OP_DIV: i64 = OpCode::Div as i64;
const OP_MOD: i64 = OpCode::Mod as i64;
const OP_EQ: i64 = OpCode::Equals as i64;
const OP_NE: i64 = OpCode::NotEquals as i64;
const OP_LT: i64 = OpCode::Less as i64;
const OP_LE: i64 = OpCode::LessEquals as i64;
const OP_GT: i64 = OpCode::Greater as i64;
const OP_GE: i64 = OpCode::GreaterEquals as i64;
const OP_RETURN: i64 = OpCode::Return as i64;

macro_rules! interned_binop {
    ($frame:ident, $vm:ident, $here:ident, $op:expr) => {{
        let depth = $frame.valuestackdepth;
        let b = $frame.locals_stack_w[depth - 1];
        let a = $frame.locals_stack_w[depth - 2];
        if a.is_null() || b.is_null() {
            residual_dispatch($vm, $here)
        } else {
            let r = unsafe { $op(a, b) };
            if r == ERROR_SENTINEL {
                residual_dispatch($vm, $here)
            } else {
                $frame.locals_stack_w[depth - 2] = r;
                $frame.valuestackdepth = depth - 1;
                vm_sync_binop($vm, r as i64);
                $here + 1
            }
        }
    }};
}

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

/// Operand `a` of the instruction at `pc`.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn insn_a(program: &CelCode, pc: usize) -> i64 {
    program
        .insns
        .get(pc)
        .map(|insn| i64::from(insn.ops[0]))
        .unwrap_or(0)
}

fn vm_of<'a>(vm_bits: i64) -> &'a mut Vm<'a> {
    unsafe { &mut *(vm_bits as usize as *mut Vm<'a>) }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn vm_sync_binop(vm_bits: i64, w: i64) {
    vm_of(vm_bits).sync_pop_push_interned(2, w as usize as CelRef);
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn vm_sync_push(vm_bits: i64, w: i64) {
    vm_of(vm_bits).sync_push_interned(w as usize as CelRef);
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn vm_sync_store(vm_bits: i64, slot: i64, w: i64) {
    vm_of(vm_bits).sync_store_interned(slot as u32, w as usize as CelRef);
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn vm_park_return(vm_bits: i64, w: i64) {
    vm_of(vm_bits).park_return(w as usize as CelRef);
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
        insn_a => residual_int,
        residual_dispatch => residual_int,
        vm_sync_binop => residual_int,
        vm_sync_push => residual_int,
        vm_sync_store => residual_int,
        vm_park_return => residual_int,
        cel_add => inline_ref,
        cel_sub => inline_ref,
        cel_mul => inline_ref,
        cel_div => inline_ref,
        cel_rem => inline_ref,
        cel_equals => inline_ref,
        cel_not_equals => inline_ref,
        cel_less => inline_ref,
        cel_less_equals => inline_ref,
        cel_greater => inline_ref,
        cel_greater_equals => inline_ref,
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
        let frame = unsafe { &mut *(state.frame as *mut W_CelFrame) };
        frame.last_instr = pc as i64;
        let opcode = insn_op(program, pc);
        let vm = state.vm;
        let here = pc as i64;
        let next = match opcode {
            OP_LOAD_LOCAL => {
                let slot = insn_a(program, pc);
                let w = frame.locals_stack_w[slot];
                if w.is_null() {
                    residual_dispatch(vm, here)
                } else {
                    let depth = frame.valuestackdepth;
                    frame.locals_stack_w[depth] = w;
                    frame.valuestackdepth = depth + 1;
                    vm_sync_push(vm, w as i64);
                    here + 1
                }
            }
            OP_STORE_LOCAL => {
                let depth = frame.valuestackdepth;
                let w = frame.locals_stack_w[depth - 1];
                if w.is_null() {
                    residual_dispatch(vm, here)
                } else {
                    let slot = insn_a(program, pc);
                    frame.locals_stack_w[slot] = w;
                    frame.valuestackdepth = depth - 1;
                    vm_sync_store(vm, slot, w as i64);
                    here + 1
                }
            }
            OP_ADD => interned_binop!(frame, vm, here, cel_add),
            OP_SUB => interned_binop!(frame, vm, here, cel_sub),
            OP_MUL => interned_binop!(frame, vm, here, cel_mul),
            OP_DIV => interned_binop!(frame, vm, here, cel_div),
            OP_MOD => interned_binop!(frame, vm, here, cel_rem),
            OP_EQ => interned_binop!(frame, vm, here, cel_equals),
            OP_NE => interned_binop!(frame, vm, here, cel_not_equals),
            OP_LT => interned_binop!(frame, vm, here, cel_less),
            OP_LE => interned_binop!(frame, vm, here, cel_less_equals),
            OP_GT => interned_binop!(frame, vm, here, cel_greater),
            OP_GE => interned_binop!(frame, vm, here, cel_greater_equals),
            OP_RETURN => {
                let depth = frame.valuestackdepth;
                let w = frame.locals_stack_w[depth - 1];
                if w.is_null() {
                    residual_dispatch(vm, here)
                } else {
                    vm_park_return(vm, w as i64);
                    PORTAL_DONE
                }
            }
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
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("1 == 1").unwrap()).unwrap(),
                &ctx
            )
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("3 * 4 - 2").unwrap()).unwrap(),
                &ctx
            )
            .unwrap(),
            Value::Int(10)
        );
    }

    #[test]
    fn interned_opcode_numbers_match_the_enum() {
        assert_eq!(OP_LOAD_LOCAL, OpCode::LoadLocal as i64);
        assert_eq!(OP_ADD, OpCode::Add as i64);
        assert_eq!(OP_EQ, OpCode::Equals as i64);
        assert_eq!(OP_RETURN, OpCode::Return as i64);
    }
}
