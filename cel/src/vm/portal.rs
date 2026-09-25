//! The JIT portal for [`super::interp::cel_eval_loop`].
//!
//! Greens are `(pc, program)`. The frame is the virtualizable
//! (`virtualizable_fields`): `last_instr`, `valuestackdepth`, and
//! `locals_stack_w[*]`. Its token is 0 outside the JIT (`TOKEN_NONE`,
//! `virtualizable.py`). `jit_merge_point` is the first statement of the
//! loop; `can_enter_jit` is only on a backward jump.
//!
//! Interned arithmetic, comparison, local load/store, context load,
//! field/index and return run on `frame.locals_stack_w[i]` — the
//! `getarrayitem_vable_*` shape — and call `cel_add` / `cel_equals`
//! with no `Result`. Everything else is residual [`Vm::dispatch_one`].

use majit_metainterp::JitDriver;

use super::code::CelCode;
use super::error::{CelErr, NameId};
use super::interp::{interned_optional_is_none, Step, Vm};
use super::opcode::OpCode;
use crate::runtime::binop::{
    cel_add, cel_div, cel_equals, cel_greater, cel_greater_equals, cel_less, cel_less_equals,
    cel_mul, cel_negate, cel_not_equals, cel_rem, cel_sub,
};
use crate::runtime::convert::{
    intern_leaf, interned_as_keyref, interned_list_get, interned_map_get,
    interned_map_lookup_string,
};
use crate::runtime::error::ERROR_SENTINEL;
use crate::runtime::heap::CelHeap;
use crate::runtime::object::{
    bytes_len, interned_list_eq, list_int_at, list_ints_slice, list_len,
    list_promote_empty_to_ints, list_resize_ge, list_store_int, list_try_append, map_len,
    map_try_insert, new_bool, new_double_in, new_int, new_int_in, new_list_with_capacity_in,
    new_map_with_capacity_in, new_string, string_as_str, string_byte_len, w_kind, w_type, CelKind,
    CelObject, CelRef, ListStrategy, W_BoolObject, W_DoubleObject, W_IntObject, W_OptionalObject,
    CEL_DOUBLE_CLASS, CEL_INT_CLASS,
};
use crate::runtime::object::{force_virtualizable_if_necessary, W_CelFrame};
use crate::runtime::optional::{
    cel_optional_has_value, cel_optional_none, cel_optional_of, cel_optional_of_non_zero_value,
    cel_optional_or, cel_optional_or_value, cel_optional_value,
};
use crate::{ExecutionError, Value};

/// `dispatch_one` finished the program.
const PORTAL_DONE: i64 = -1;
/// `dispatch_one` failed with no handler.
const PORTAL_FAIL: i64 = -2;

/// A published interned leaf, or `None` when the cell is null.
fn slot_leaf(w: i64) -> Option<CelRef> {
    let w = w as usize as CelRef;
    if w.is_null() {
        None
    } else {
        Some(w)
    }
}

/// Operand `from_top` down the value stack (1 = top).
///
/// Locals occupy `0..n_slots`. A `CallQualified` miss parks its arguments and
/// drops `valuestackdepth` to the locals, so a later `CallMethod` must not
/// treat a local — or the items-block header at index `-1` — as an operand.
fn operand_cell(frame: &W_CelFrame, from_top: i64) -> Option<CelRef> {
    let i = frame.valuestackdepth - from_top;
    if i < frame.n_slots {
        return None;
    }
    let w = frame.locals_stack_w[i];
    #[cfg(debug_assertions)]
    crate::runtime::heap::assert_frame_cell(w);
    Some(w)
}

fn read_cell(frame: &W_CelFrame, i: i64) -> CelRef {
    let w = frame.locals_stack_w[i];
    #[cfg(debug_assertions)]
    crate::runtime::heap::assert_frame_cell(w);
    w
}

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
const OP_LOAD_CONST: i64 = OpCode::LoadConst as i64;
const OP_INC_LOCAL: i64 = OpCode::IncLocal as i64;
const OP_ADD_K: i64 = OpCode::AddConst as i64;
const OP_MUL_K: i64 = OpCode::MulConst as i64;
const OP_MOD_K: i64 = OpCode::ModConst as i64;
const OP_EQ_K: i64 = OpCode::EqualsConst as i64;
const OP_NE_K: i64 = OpCode::NotEqualsConst as i64;
const OP_LT_K: i64 = OpCode::LessConst as i64;
const OP_GT_K: i64 = OpCode::GreaterConst as i64;
const OP_GE_K: i64 = OpCode::GreaterEqualsConst as i64;
const OP_ADD_LOCAL_K: i64 = OpCode::AddLocalConst as i64;
const OP_MUL_LOCAL_K: i64 = OpCode::MulLocalConst as i64;
const OP_MOD_LOCAL_K: i64 = OpCode::ModLocalConst as i64;
const OP_EQ_LOCAL_K: i64 = OpCode::EqualsLocalConst as i64;
const OP_NE_LOCAL_K: i64 = OpCode::NotEqualsLocalConst as i64;
const OP_LT_LOCAL_K: i64 = OpCode::LessLocalConst as i64;
const OP_GT_LOCAL_K: i64 = OpCode::GreaterLocalConst as i64;
const OP_GE_LOCAL_K: i64 = OpCode::GreaterEqualsLocalConst as i64;
const OP_AND: i64 = OpCode::And as i64;
const OP_OR: i64 = OpCode::Or as i64;
const OP_AND_LOCAL: i64 = OpCode::AndLocal as i64;
const OP_OR_LOCAL: i64 = OpCode::OrLocal as i64;
const OP_AND_MERGE: i64 = OpCode::AndMerge as i64;
const OP_OR_MERGE: i64 = OpCode::OrMerge as i64;
const OP_NEW_MAP: i64 = OpCode::NewMap as i64;
const OP_MAP_INSERT: i64 = OpCode::MapInsert as i64;
const OP_MAP_INSERT_OPTIONAL: i64 = OpCode::MapInsertOptional as i64;
const OP_JUMP: i64 = OpCode::Jump as i64;
const OP_JUMP_IF_FALSE: i64 = OpCode::JumpIfFalse as i64;
const OP_JUMP_IF_TRUE: i64 = OpCode::JumpIfTrue as i64;
const OP_ITER_ADVANCE: i64 = OpCode::IterAdvance as i64;
const OP_NEW_LIST: i64 = OpCode::NewList as i64;
const OP_NEW_LIST_FROM_ARG: i64 = OpCode::NewListFromArg as i64;
const OP_LIST_APPEND: i64 = OpCode::ListAppend as i64;
const OP_ITER_ELEMS: i64 = OpCode::IterElems as i64;
const OP_ITER_LEN: i64 = OpCode::IterLen as i64;
const OP_ITER_AT: i64 = OpCode::IterAt as i64;
const OP_ITER_GUARD: i64 = OpCode::IterGuard as i64;
const OP_ITER_BIND: i64 = OpCode::IterBind as i64;
const OP_LOAD_VAR: i64 = OpCode::LoadVar as i64;
const OP_GET_FIELD: i64 = OpCode::GetField as i64;
const OP_HAS_FIELD: i64 = OpCode::HasField as i64;
const OP_GET_FIELD_LOCAL: i64 = OpCode::GetFieldLocal as i64;
const OP_HAS_FIELD_LOCAL: i64 = OpCode::HasFieldLocal as i64;
const OP_GET_FIELD_LOCAL_APPEND: i64 = OpCode::GetFieldLocalAppend as i64;
const OP_HAS_FIELD_LOCAL_APPEND: i64 = OpCode::HasFieldLocalAppend as i64;
const OP_INDEX: i64 = OpCode::Index as i64;
const OP_NOT: i64 = OpCode::Not as i64;
const OP_NEGATE: i64 = OpCode::Negate as i64;
const OP_IN: i64 = OpCode::In as i64;
const OP_LOAD_LOCAL_APPEND: i64 = OpCode::LoadLocalAppend as i64;
const OP_CALL_HOST: i64 = OpCode::CallHost as i64;
const OP_CALL_METHOD: i64 = OpCode::CallMethod as i64;
const OP_CALL_QUALIFIED: i64 = OpCode::CallQualified as i64;
const OP_OPT_INDEX: i64 = OpCode::OptIndex as i64;
const OP_OPT_SELECT: i64 = OpCode::OptSelect as i64;
const OP_ITER_KEYS: i64 = OpCode::IterKeys as i64;
const OP_JUMP_IF_OPT_NONE: i64 = OpCode::JumpIfOptNone as i64;
const OP_LIST_APPEND_OPTIONAL: i64 = OpCode::ListAppendOptional as i64;
const OP_NOT_STRICTLY_FALSE: i64 = OpCode::NotStrictlyFalse as i64;
const OP_ACCU_LOOP_COND: i64 = OpCode::AccuLoopCond as i64;
const OP_ACCU_LOOP_COND_NOT: i64 = OpCode::AccuLoopCondNot as i64;
const OP_ADD_LOCAL_K_APPEND: i64 = OpCode::AddLocalConstAppend as i64;
const OP_MUL_LOCAL_K_APPEND: i64 = OpCode::MulLocalConstAppend as i64;
const OP_MOD_LOCAL_K_APPEND: i64 = OpCode::ModLocalConstAppend as i64;
const OP_EQ_LOCAL_K_APPEND: i64 = OpCode::EqualsLocalConstAppend as i64;
const OP_NE_LOCAL_K_APPEND: i64 = OpCode::NotEqualsLocalConstAppend as i64;
const OP_LT_LOCAL_K_APPEND: i64 = OpCode::LessLocalConstAppend as i64;
const OP_GT_LOCAL_K_APPEND: i64 = OpCode::GreaterLocalConstAppend as i64;
const OP_GE_LOCAL_K_APPEND: i64 = OpCode::GreaterEqualsLocalConstAppend as i64;

macro_rules! interned_binop {
    ($frame:ident, $vm:ident, $here:ident, $op:expr) => {{
        match (operand_cell($frame, 2), operand_cell($frame, 1)) {
            (Some(a), Some(b)) if !a.is_null() && !b.is_null() => {
                let r = unsafe { $op(a, b) };
                if r == ERROR_SENTINEL {
                    residual_dispatch($vm, $here)
                } else {
                    let depth = $frame.valuestackdepth;
                    $frame.locals_stack_w[depth - 2] = r;
                    $frame.valuestackdepth = depth - 1;
                    $here + 1
                }
            }
            _ => residual_dispatch($vm, $here),
        }
    }};
}

macro_rules! interned_arith {
    ($frame:ident, $vm:ident, $here:ident, $op:expr) => {{
        match (operand_cell($frame, 2), operand_cell($frame, 1)) {
            (Some(a), Some(b)) if !a.is_null() && !b.is_null() => {
                let r = unsafe { $op($vm, a, b) };
                if r == ERROR_SENTINEL {
                    residual_dispatch($vm, $here)
                } else {
                    let depth = $frame.valuestackdepth;
                    $frame.locals_stack_w[depth - 2] = r;
                    $frame.valuestackdepth = depth - 1;
                    $here + 1
                }
            }
            _ => residual_dispatch($vm, $here),
        }
    }};
}

macro_rules! interned_binop_k {
    ($frame:ident, $vm:ident, $program:ident, $pc:ident, $here:ident, $op:expr) => {{
        let k = intern_const($program, insn_a($program, $pc));
        match operand_cell($frame, 1) {
            Some(a) if !a.is_null() && !k.is_null() => {
                let r = unsafe { $op(a, k) };
                if r == ERROR_SENTINEL {
                    residual_dispatch($vm, $here)
                } else {
                    let depth = $frame.valuestackdepth;
                    $frame.locals_stack_w[depth - 1] = r;
                    $here + 1
                }
            }
            _ => residual_dispatch($vm, $here),
        }
    }};
}

macro_rules! interned_arith_k {
    ($frame:ident, $vm:ident, $program:ident, $pc:ident, $here:ident, $op:expr) => {{
        let k = intern_const($program, insn_a($program, $pc));
        match operand_cell($frame, 1) {
            Some(a) if !a.is_null() && !k.is_null() => {
                let r = unsafe { $op($vm, a, k) };
                if r == ERROR_SENTINEL {
                    residual_dispatch($vm, $here)
                } else {
                    let depth = $frame.valuestackdepth;
                    $frame.locals_stack_w[depth - 1] = r;
                    $here + 1
                }
            }
            _ => residual_dispatch($vm, $here),
        }
    }};
}

macro_rules! interned_local_k {
    ($frame:ident, $vm:ident, $program:ident, $pc:ident, $here:ident, $op:expr) => {{
        let slot = insn_a($program, $pc);
        let a = read_cell($frame, slot);
        let k = intern_const($program, insn_b($program, $pc));
        if a.is_null() || k.is_null() {
            residual_dispatch($vm, $here)
        } else {
            let r = unsafe { $op(a, k) };
            if r == ERROR_SENTINEL {
                residual_dispatch($vm, $here)
            } else {
                let depth = $frame.valuestackdepth;
                $frame.locals_stack_w[depth] = r;
                $frame.valuestackdepth = depth + 1;
                $here + 1
            }
        }
    }};
}

macro_rules! interned_arith_local_k {
    ($frame:ident, $vm:ident, $program:ident, $pc:ident, $here:ident, $op:expr) => {{
        let slot = insn_a($program, $pc);
        let a = read_cell($frame, slot);
        let k = intern_const($program, insn_b($program, $pc));
        if a.is_null() || k.is_null() {
            residual_dispatch($vm, $here)
        } else {
            let r = unsafe { $op($vm, a, k) };
            if r == ERROR_SENTINEL {
                residual_dispatch($vm, $here)
            } else {
                let depth = $frame.valuestackdepth;
                $frame.locals_stack_w[depth] = r;
                $frame.valuestackdepth = depth + 1;
                $here + 1
            }
        }
    }};
}

macro_rules! interned_local_k_append {
    ($frame:ident, $vm:ident, $program:ident, $pc:ident, $here:ident, $op:expr) => {{
        let a = read_cell($frame, insn_a($program, $pc));
        let k = intern_const($program, insn_b($program, $pc));
        match operand_cell($frame, 1) {
            Some(list) if !a.is_null() && !k.is_null() && !list.is_null() => {
                let r = unsafe { $op(a, k) };
                if r == ERROR_SENTINEL || try_append(list as i64, r as i64) == 0 {
                    residual_dispatch($vm, $here)
                } else {
                    $here + 1
                }
            }
            _ => residual_dispatch($vm, $here),
        }
    }};
}

macro_rules! interned_arith_local_k_append {
    ($frame:ident, $vm:ident, $program:ident, $pc:ident, $here:ident, $op:expr) => {{
        let a = read_cell($frame, insn_a($program, $pc));
        let k = intern_const($program, insn_b($program, $pc));
        match operand_cell($frame, 1) {
            Some(list) if !a.is_null() && !k.is_null() && !list.is_null() => {
                let r = unsafe { $op($vm, a, k) };
                if r == ERROR_SENTINEL || try_append(list as i64, r as i64) == 0 {
                    residual_dispatch($vm, $here)
                } else {
                    $here + 1
                }
            }
            _ => residual_dispatch($vm, $here),
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

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn insn_b(program: &CelCode, pc: usize) -> i64 {
    program
        .insns
        .get(pc)
        .map(|insn| i64::from(insn.ops[1]))
        .unwrap_or(0)
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn insn_c(program: &CelCode, pc: usize) -> i64 {
    program
        .insns
        .get(pc)
        .map(|insn| i64::from(insn.ops[2]))
        .unwrap_or(0)
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
unsafe fn interned_equals(a: CelRef, b: CelRef) -> CelRef {
    if let Some(eq) = interned_list_eq(a, b) {
        new_bool(eq) as CelRef
    } else {
        cel_equals(a, b)
    }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
unsafe fn interned_not_equals(a: CelRef, b: CelRef) -> CelRef {
    if let Some(eq) = interned_list_eq(a, b) {
        new_bool(!eq) as CelRef
    } else {
        cel_not_equals(a, b)
    }
}

/// Item of an interned list. Ints wrap as a young `int` leaf.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_item(vm_bits: i64, list: i64, index: i64) -> i64 {
    let Some(list) = slot_leaf(list) else {
        return 0;
    };
    if unsafe { w_kind(list) } != CelKind::List {
        return 0;
    }
    if let Some(n) = unsafe { list_int_at(list, index) } {
        return new_int_in(vm_heap(vm_bits), n) as i64;
    }
    match unsafe { interned_list_get(list, index) } {
        Some(w) => w as usize as i64,
        None => 0,
    }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn try_append(list: i64, item: i64) -> i64 {
    let Some(list) = slot_leaf(list) else {
        return 0;
    };
    let Some(item) = slot_leaf(item) else {
        return 0;
    };
    unsafe { i64::from(list_try_append(list, item)) }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn try_map_insert(map: i64, key: i64, value: i64) -> i64 {
    let Some(map) = slot_leaf(map) else {
        return 0;
    };
    let Some(key) = slot_leaf(key) else {
        return 0;
    };
    let Some(value) = slot_leaf(value) else {
        return 0;
    };
    unsafe { i64::from(map_try_insert(map, key, value)) }
}

/// Constant pool leaf. Null means the index misses.
///
/// Pure in `(program, idx)`: both are green at a trace, so the call folds.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn intern_const(program: &CelCode, idx: i64) -> *mut crate::runtime::object::CelObject {
    if let Some(w) = program.const_leaf(idx as u32) {
        return w;
    }
    let Some(value) = program.konst(idx as u32) else {
        return core::ptr::null_mut();
    };
    intern_leaf(value).unwrap_or(core::ptr::null_mut())
}

/// Field `names[name_idx]` of interned map/struct `w`. 0 means residual.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_field(w: i64, program: &CelCode, name_idx: i64) -> i64 {
    let Some(w) = slot_leaf(w) else {
        return 0;
    };
    let Some(field) = program.name(NameId(name_idx as u32)) else {
        return 0;
    };
    let found = match unsafe { w_kind(w) } {
        CelKind::Map => unsafe { interned_map_lookup_string(w, field) },
        #[cfg(feature = "structs")]
        CelKind::Struct => unsafe { crate::runtime::object::struct_lookup_field(w, field) },
        _ => None,
    };
    found.map(|r| r as usize as i64).unwrap_or(0)
}

/// Whether interned map/struct `w` has `names[name_idx]`.
///
/// `0` residual, `1` false, `2` true.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_has_field(w: i64, program: &CelCode, name_idx: i64) -> i64 {
    let Some(w) = slot_leaf(w) else {
        return 0;
    };
    let Some(field) = program.name(NameId(name_idx as u32)) else {
        return 0;
    };
    match unsafe { w_kind(w) } {
        CelKind::Map => {
            if unsafe { interned_map_lookup_string(w, field) }.is_some() {
                2
            } else {
                1
            }
        }
        #[cfg(feature = "structs")]
        CelKind::Struct => {
            if unsafe { crate::runtime::object::struct_lookup_field(w, field) }.is_some() {
                2
            } else {
                1
            }
        }
        _ => 0,
    }
}

/// Interned `container[key]`. 0 means residual (including a miss).
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_index(container: i64, key: i64) -> i64 {
    let Some(w) = slot_leaf(container) else {
        return 0;
    };
    let Some(k) = slot_leaf(key) else {
        return 0;
    };
    let found = match unsafe { w_kind(w) } {
        CelKind::List => {
            if unsafe { w_kind(k) } != CelKind::Int {
                return 0;
            }
            let index = unsafe { (*k.cast::<W_IntObject>()).intval };
            if let Some(n) = unsafe { list_int_at(w, index) } {
                return new_int(n) as i64;
            }
            unsafe { interned_list_get(w, index) }
        }
        CelKind::Map => {
            let Some(needle) = (unsafe { interned_as_keyref(k) }) else {
                return 0;
            };
            unsafe { interned_map_get(w, needle) }
        }
        #[cfg(feature = "structs")]
        CelKind::Struct => match unsafe { string_as_str(k) } {
            Some(field) => unsafe { crate::runtime::object::struct_lookup_field(w, field) },
            None => None,
        },
        _ => None,
    };
    found.map(|r| r as usize as i64).unwrap_or(0)
}

/// Interned `needle in container`. `0` residual, `1` false, `2` true.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_contains(container: i64, needle: i64) -> i64 {
    let Some(c) = slot_leaf(container) else {
        return 0;
    };
    let Some(n) = slot_leaf(needle) else {
        return 0;
    };
    match crate::objects::interned_contains(c, n) {
        Ok(true) => 2,
        Ok(false) => 1,
        Err(_) => 0,
    }
}

/// Length of an interned list/map/string/bytes. `-1` means residual.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_len_cell(w: CelRef) -> i64 {
    interned_len(w as i64)
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_len(w: i64) -> i64 {
    let Some(w) = slot_leaf(w) else {
        return -1;
    };
    match unsafe { w_kind(w) } {
        CelKind::List => unsafe { list_len(w) },
        CelKind::Map => unsafe { map_len(w) },
        CelKind::Str => unsafe { string_byte_len(w) },
        CelKind::Bytes => unsafe { bytes_len(w) },
        _ => -1,
    }
}

/// `1` if `names[idx]` is `size`.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_is_size(program: &CelCode, idx: i64) -> i64 {
    i64::from(program.name(NameId(idx as u32)) == Some("size"))
}

/// Unary optional method. 0 residual, else the result (may be `ERROR_SENTINEL`).
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_optional_unary(program: &CelCode, name_idx: i64, w: i64) -> i64 {
    let Some(w) = slot_leaf(w) else {
        return 0;
    };
    match program.name(NameId(name_idx as u32)) {
        Some("value") => unsafe { cel_optional_value(w) as i64 },
        Some("hasValue") => unsafe { cel_optional_has_value(w) as i64 },
        _ => 0,
    }
}

/// Method with one argument. 0 residual.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_method1(program: &CelCode, name_idx: i64, recv: i64, arg: i64) -> i64 {
    let Some(name) = program.name(NameId(name_idx as u32)) else {
        return 0;
    };
    let Some(r) = slot_leaf(recv) else {
        return 0;
    };
    let Some(a) = slot_leaf(arg) else {
        return 0;
    };
    match name {
        "contains" => {
            let found = interned_contains(recv, arg);
            if found == 0 {
                0
            } else {
                new_bool(found == 2) as i64
            }
        }
        "startsWith" => match (unsafe { string_as_str(r) }, unsafe { string_as_str(a) }) {
            (Some(rs), Some(ns)) => new_bool(rs.starts_with(ns)) as i64,
            _ => 0,
        },
        "endsWith" => match (unsafe { string_as_str(r) }, unsafe { string_as_str(a) }) {
            (Some(rs), Some(ns)) => new_bool(rs.ends_with(ns)) as i64,
            _ => 0,
        },
        #[cfg(feature = "regex")]
        "matches" => match (unsafe { string_as_str(r) }, unsafe { string_as_str(a) }) {
            (Some(rs), Some(ns)) => match crate::runtime::regex_intern::intern_regex(ns) {
                Ok(re) => new_bool(re.is_match(rs)) as i64,
                Err(_) => 0,
            },
            _ => 0,
        },
        "or" => unsafe { cel_optional_or(r, a) as i64 },
        "orValue" => unsafe { cel_optional_or_value(r, a) as i64 },
        _ => 0,
    }
}

/// `0` unknown, `1` optional.none, `2` optional.of, `3` optional.ofNonZeroValue.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_qualified_kind(program: &CelCode, name_idx: i64) -> i64 {
    match program.name(NameId(name_idx as u32)) {
        Some("optional.none") => 1,
        Some("optional.of") => 2,
        Some("optional.ofNonZeroValue") => 3,
        _ => 0,
    }
}

/// `0` not optional, `1` some, `2` none.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_optional_state(w: i64) -> i64 {
    let Some(w) = slot_leaf(w) else {
        return 0;
    };
    if interned_optional_is_none(w) {
        2
    } else if unsafe { w_kind(w) } != CelKind::Optional {
        0
    } else {
        1
    }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_optional_inner(w: i64) -> i64 {
    let Some(w) = slot_leaf(w) else {
        return 0;
    };
    if unsafe { w_kind(w) } != CelKind::Optional {
        return 0;
    }
    let inner = unsafe { (*w.cast::<W_OptionalObject>()).w_value };
    if inner.is_null() {
        0
    } else {
        inner as i64
    }
}

/// `0` residual, `1` false, `2` true, `3` non-bool.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_as_bool(w: i64) -> i64 {
    let Some(w) = slot_leaf(w) else {
        return 0;
    };
    if unsafe { w_kind(w) } != CelKind::Bool {
        return 3;
    }
    if unsafe { (*w.cast::<W_BoolObject>()).boolval } != 0 {
        2
    } else {
        1
    }
}

/// Optional index. `0` residual, else an optional leaf.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_opt_index(container: i64, key: i64) -> i64 {
    let Some(mut w) = slot_leaf(container) else {
        return 0;
    };
    let Some(k) = slot_leaf(key) else {
        return 0;
    };
    if unsafe { w_kind(w) } == CelKind::Optional {
        let inner = unsafe { (*w.cast::<W_OptionalObject>()).w_value };
        if inner.is_null() {
            return cel_optional_none() as i64;
        }
        w = inner;
    }
    let found = match unsafe { w_kind(w) } {
        CelKind::List => {
            if unsafe { w_kind(k) } != CelKind::Int {
                return 0;
            }
            let index = unsafe { (*k.cast::<W_IntObject>()).intval };
            if let Some(n) = unsafe { list_int_at(w, index) } {
                return unsafe { cel_optional_of(new_int(n) as CelRef) as i64 };
            }
            if unsafe { list_ints_slice(w) }.is_some() {
                return cel_optional_none() as i64;
            }
            unsafe { interned_list_get(w, index) }
        }
        CelKind::Map => {
            let Some(needle) = (unsafe { interned_as_keyref(k) }) else {
                return 0;
            };
            unsafe { interned_map_get(w, needle) }
        }
        #[cfg(feature = "structs")]
        CelKind::Struct => match unsafe { string_as_str(k) } {
            Some(field) => unsafe { crate::runtime::object::struct_lookup_field(w, field) },
            None => None,
        },
        _ => return 0,
    };
    match found {
        Some(item) => unsafe { cel_optional_of(item) as i64 },
        None => cel_optional_none() as i64,
    }
}

/// Optional field. `0` residual, else an optional leaf.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_opt_select(w: i64, program: &CelCode, name_idx: i64) -> i64 {
    let Some(w) = slot_leaf(w) else {
        return 0;
    };
    let Some(field) = program.name(NameId(name_idx as u32)) else {
        return 0;
    };
    match crate::objects::interned_opt_select(w, field) {
        Ok(r) => r as usize as i64,
        Err(_) => 0,
    }
}

fn vm_of<'a>(vm_bits: i64) -> &'a mut Vm<'a> {
    unsafe { &mut *(vm_bits as usize as *mut Vm<'a>) }
}

#[inline]
fn vm_heap<'a>(vm_bits: i64) -> &'a CelHeap {
    unsafe { &*(*(vm_bits as *mut Vm<'a>)).heap }
}

/// Interned left of `&&` / `||`. `1` short-circuits (the result is `is_or`),
/// `2` falls through to the right, `0` declines to residual.
fn interned_bool_short(w: CelRef, is_or: bool) -> i64 {
    if w.is_null() || unsafe { w_kind(w) } != CelKind::Bool {
        0
    } else if unsafe { (*w.cast::<W_BoolObject>()).boolval != 0 } == is_or {
        1
    } else {
        2
    }
}

/// `1` if a merge can keep the interned right-hand bool: the left was the
/// non-short-circuiting bool, or the interned arm never wrote the slot.
fn keep_right_merge(vm_bits: i64, slot: i64, is_or: bool) -> i64 {
    match vm_of(vm_bits).logic_copy(slot as u32) {
        Some(Err(CelErr::InternalError)) => 1,
        Some(Ok(true)) => i64::from(!is_or),
        Some(Ok(false)) => i64::from(is_or),
        _ => 0,
    }
}

/// Field/has/optional/`in`/qualified/map/`&&`/`||` arms.
#[inline(never)]
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn portal_rare(
    frame: &mut W_CelFrame,
    vm: i64,
    program: &CelCode,
    pc: usize,
    here: i64,
    opcode: i64,
) -> i64 {
    unsafe { force_virtualizable_if_necessary(frame) };
    match opcode {
        OP_HAS_FIELD => match operand_cell(frame, 1) {
            Some(recv) if !recv.is_null() => {
                let found = interned_has_field(recv as i64, program, insn_a(program, pc));
                if found == 0 {
                    residual_dispatch(vm, here)
                } else {
                    let depth = frame.valuestackdepth;
                    let r = new_bool(found == 2) as CelRef;
                    frame.locals_stack_w[depth - 1] = r;
                    here + 1
                }
            }
            _ => residual_dispatch(vm, here),
        },
        OP_HAS_FIELD_LOCAL => {
            let recv = read_cell(frame, insn_a(program, pc));
            let found = interned_has_field(recv as i64, program, insn_b(program, pc));
            if recv.is_null() || found == 0 {
                residual_dispatch(vm, here)
            } else {
                let r = new_bool(found == 2) as CelRef;
                let depth = frame.valuestackdepth;
                frame.locals_stack_w[depth] = r;
                frame.valuestackdepth = depth + 1;
                here + 1
            }
        }
        OP_HAS_FIELD_LOCAL_APPEND => {
            let recv = read_cell(frame, insn_a(program, pc));
            let found = interned_has_field(recv as i64, program, insn_b(program, pc));
            let depth = frame.valuestackdepth;
            let list = frame.locals_stack_w[depth - 1];
            if recv.is_null() || found == 0 || list.is_null() {
                residual_dispatch(vm, here)
            } else {
                let r = new_bool(found == 2) as CelRef;
                if try_append(list as i64, r as i64) == 0 {
                    residual_dispatch(vm, here)
                } else {
                    here + 1
                }
            }
        }
        OP_OPT_INDEX => match (operand_cell(frame, 2), operand_cell(frame, 1)) {
            (Some(container), Some(key)) if !container.is_null() && !key.is_null() => {
                let item = interned_opt_index(container as i64, key as i64);
                if item == 0 {
                    residual_dispatch(vm, here)
                } else {
                    let depth = frame.valuestackdepth;
                    let r = item as usize as CelRef;
                    frame.locals_stack_w[depth - 2] = r;
                    frame.valuestackdepth = depth - 1;
                    here + 1
                }
            }
            _ => residual_dispatch(vm, here),
        },
        OP_OPT_SELECT => match operand_cell(frame, 1) {
            Some(recv) if !recv.is_null() => {
                let found = interned_opt_select(recv as i64, program, insn_a(program, pc));
                if found == 0 {
                    residual_dispatch(vm, here)
                } else {
                    let depth = frame.valuestackdepth;
                    let r = found as usize as CelRef;
                    frame.locals_stack_w[depth - 1] = r;
                    here + 1
                }
            }
            _ => residual_dispatch(vm, here),
        },
        OP_JUMP_IF_OPT_NONE => {
            let depth = frame.valuestackdepth;
            let w = frame.locals_stack_w[depth - 1];
            // Residual jumps only when interned_optional_is_none; a plain
            // value, a some, a missing cell, and a builder all fall through.
            if !w.is_null() && interned_optional_is_none(w) {
                insn_a(program, pc)
            } else {
                here + 1
            }
        }
        OP_LIST_APPEND_OPTIONAL => {
            let depth = frame.valuestackdepth;
            let item = frame.locals_stack_w[depth - 1];
            let list = frame.locals_stack_w[depth - 2];
            let state = interned_optional_state(item as i64);
            if list.is_null() || state == 0 {
                residual_dispatch(vm, here)
            } else if state == 2 {
                frame.valuestackdepth = depth - 1;
                here + 1
            } else {
                let inner = interned_optional_inner(item as i64);
                if inner == 0 || try_append(list as i64, inner) == 0 {
                    residual_dispatch(vm, here)
                } else {
                    frame.valuestackdepth = depth - 1;
                    here + 1
                }
            }
        }
        OP_NOT_STRICTLY_FALSE => {
            let depth = frame.valuestackdepth;
            let w = frame.locals_stack_w[depth - 1];
            match interned_as_bool(w as i64) {
                0 => residual_dispatch(vm, here),
                found => {
                    let r = new_bool(found != 1) as CelRef;
                    frame.locals_stack_w[depth - 1] = r;
                    here + 1
                }
            }
        }
        OP_IN => match (operand_cell(frame, 2), operand_cell(frame, 1)) {
            (Some(needle), Some(container)) if !needle.is_null() && !container.is_null() => {
                let found = interned_contains(container as i64, needle as i64);
                if found == 0 {
                    residual_dispatch(vm, here)
                } else {
                    let depth = frame.valuestackdepth;
                    let r = new_bool(found == 2) as CelRef;
                    frame.locals_stack_w[depth - 2] = r;
                    frame.valuestackdepth = depth - 1;
                    here + 1
                }
            }
            _ => residual_dispatch(vm, here),
        },
        OP_CALL_QUALIFIED => {
            let kind = interned_qualified_kind(program, insn_a(program, pc));
            let arity = insn_b(program, pc);
            let skip = insn_c(program, pc);
            if kind == 1 && arity == 0 {
                let w = cel_optional_none();
                let depth = frame.valuestackdepth;
                frame.locals_stack_w[depth] = w;
                frame.valuestackdepth = depth + 1;
                skip
            } else if (kind == 2 || kind == 3) && arity == 1 {
                let depth = frame.valuestackdepth;
                let w = frame.locals_stack_w[depth - 1];
                if w.is_null() {
                    residual_dispatch(vm, here)
                } else {
                    let r = if kind == 2 {
                        unsafe { cel_optional_of(w) }
                    } else {
                        unsafe { cel_optional_of_non_zero_value(w) }
                    };
                    if r == ERROR_SENTINEL {
                        residual_dispatch(vm, here)
                    } else {
                        frame.locals_stack_w[depth - 1] = r;
                        skip
                    }
                }
            } else {
                residual_dispatch(vm, here)
            }
        }
        OP_AND | OP_OR => match operand_cell(frame, 1) {
            Some(w) => {
                let is_or = opcode == OP_OR;
                match interned_bool_short(w, is_or) {
                    1 => {
                        let depth = frame.valuestackdepth;
                        frame.locals_stack_w[depth - 1] = new_bool(is_or) as CelRef;
                        insn_b(program, pc)
                    }
                    2 => {
                        frame.valuestackdepth -= 1;
                        here + 1
                    }
                    _ => residual_hydrate(vm, here),
                }
            }
            _ => residual_hydrate(vm, here),
        },
        OP_AND_LOCAL | OP_OR_LOCAL => {
            let slot = insn_a(program, pc);
            let w = frame.locals_stack_w[slot];
            let is_or = opcode == OP_OR_LOCAL;
            match interned_bool_short(w, is_or) {
                1 => {
                    let r = new_bool(is_or) as CelRef;
                    let depth = frame.valuestackdepth;
                    frame.locals_stack_w[depth] = r;
                    frame.valuestackdepth = depth + 1;
                    insn_c(program, pc)
                }
                2 => here + 1,
                _ => residual_dispatch(vm, here),
            }
        }
        OP_AND_MERGE | OP_OR_MERGE => match operand_cell(frame, 1) {
            Some(w)
                if !w.is_null()
                    && unsafe { w_kind(w) } == CelKind::Bool
                    && keep_right_merge(vm, insn_a(program, pc), opcode == OP_OR_MERGE) != 0 =>
            {
                here + 1
            }
            _ => residual_dispatch(vm, here),
        },
        OP_NEW_MAP => {
            let w = new_map_with_capacity_in(vm_heap(vm), insn_a(program, pc)) as CelRef;
            let depth = frame.valuestackdepth;
            frame.locals_stack_w[depth] = w;
            frame.valuestackdepth = depth + 1;
            here + 1
        }
        OP_MAP_INSERT => {
            let depth = frame.valuestackdepth;
            let value = frame.locals_stack_w[depth - 1];
            let key = frame.locals_stack_w[depth - 2];
            let map = frame.locals_stack_w[depth - 3];
            if value.is_null()
                || key.is_null()
                || map.is_null()
                || try_map_insert(map as i64, key as i64, value as i64) == 0
            {
                residual_dispatch(vm, here)
            } else {
                frame.valuestackdepth = depth - 2;
                here + 1
            }
        }
        OP_MAP_INSERT_OPTIONAL => {
            let depth = frame.valuestackdepth;
            let value = frame.locals_stack_w[depth - 1];
            let key = frame.locals_stack_w[depth - 2];
            let map = frame.locals_stack_w[depth - 3];
            let state = interned_optional_state(value as i64);
            if map.is_null() || key.is_null() || state == 0 {
                residual_dispatch(vm, here)
            } else if state == 2 {
                frame.valuestackdepth = depth - 2;
                here + 1
            } else {
                let inner = interned_optional_inner(value as i64);
                if inner == 0 || try_map_insert(map as i64, key as i64, inner) == 0 {
                    residual_dispatch(vm, here)
                } else {
                    frame.valuestackdepth = depth - 2;
                    here + 1
                }
            }
        }
        _ => residual_dispatch(vm, here),
    }
}

macro_rules! interned_int_arith {
    ($name:ident, $checked:ident, $fallback:ident) => {
        unsafe fn $name(vm: i64, a: CelRef, b: CelRef) -> CelRef {
            if w_type(a) == (&CEL_INT_CLASS as *const _)
                && w_type(b) == (&CEL_INT_CLASS as *const _)
            {
                let l = (*a.cast::<W_IntObject>()).intval;
                let r = (*b.cast::<W_IntObject>()).intval;
                match l.$checked(r) {
                    Some(v) => new_int_in(vm_heap(vm), v) as CelRef,
                    None => $fallback(a, b),
                }
            } else {
                $fallback(a, b)
            }
        }
    };
}

interned_int_arith!(interned_add, checked_add, cel_add);
interned_int_arith!(interned_sub, checked_sub, cel_sub);
interned_int_arith!(interned_mul, checked_mul, cel_mul);
interned_int_arith!(interned_div, checked_div, cel_div);
interned_int_arith!(interned_rem, checked_rem, cel_rem);

/// Intern the context variable named `names[idx]`. Null means miss.
///
/// Residual: a [`crate::context::VariableResolver`] on the chain may
/// return a different value on every call.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn intern_var(vm_bits: i64, program: &CelCode, idx: i64) -> CelRef {
    let Some(name) = program.name(NameId(idx as u32)) else {
        return core::ptr::null_mut();
    };
    vm_of(vm_bits)
        .intern_context_var(name)
        .unwrap_or(core::ptr::null_mut())
}

/// [`intern_var`] when [`context_lookup_pure`] is true.
///
/// `effectinfo.py` `EF_ELIDABLE_CANNOT_RAISE` (`jtransform.py`
/// `_do_builtin_call` / `call.py` `EF_ELIDABLE_CANNOT_RAISE`).
/// Loop-invariant arguments reuse the preamble result (`pure.py` `OptPure`),
/// the same shape as `celldict.py` `_getdictvalue_no_unwrapping_pure`.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn intern_var_pure(
    vm_bits: i64,
    program: &CelCode,
    idx: i64,
) -> *mut crate::runtime::object::CelObject {
    let Some(name) = program.name(NameId(idx as u32)) else {
        return core::ptr::null_mut();
    };
    vm_of(vm_bits)
        .intern_context_var_pure(name)
        .unwrap_or(core::ptr::null_mut())
}

/// `1` when no resolver sits on the context chain, else `0`.
///
/// Elidable (`EF_ELIDABLE_CANNOT_RAISE`), not a residual call per iteration:
/// the context is borrowed immutably for the evaluation, so the bit is
/// loop-invariant and `OptPure` keeps it in the preamble.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn context_lookup_pure(vm_bits: i64) -> i64 {
    if vm_of(vm_bits).context_lookup_pure() {
        1
    } else {
        0
    }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_unary(vm_bits: i64, program: &CelCode, name_idx: i64, w: i64) -> i64 {
    let Some(w) = slot_leaf(w) else {
        return 0;
    };
    let Some(name) = program.name(NameId(name_idx as u32)) else {
        return 0;
    };
    vm_of(vm_bits).interned_unary_bits(name, w)
}

/// [`interned_unary`] as a reference, so the portal stores it without an
/// int-to-ref cast. Null declines.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_unary_cell(vm_bits: i64, program: &CelCode, name_idx: i64, w: CelRef) -> CelRef {
    let out = interned_unary(vm_bits, program, name_idx, w as i64);
    if out == 0 || out == ERROR_SENTINEL as i64 {
        core::ptr::null_mut()
    } else {
        out as usize as CelRef
    }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_temporal(vm_bits: i64, program: &CelCode, name_idx: i64, w: i64) -> i64 {
    let Some(w) = slot_leaf(w) else {
        return 0;
    };
    let Some(name) = program.name(NameId(name_idx as u32)) else {
        return 0;
    };
    match vm_of(vm_bits).interned_temporal_int(name, w) {
        Some(n) => new_int(n) as i64,
        None => 0,
    }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_map_keys(vm_bits: i64, w: i64) -> i64 {
    let Some(w) = slot_leaf(w) else {
        return 0;
    };
    if unsafe { w_kind(w) } != CelKind::Map {
        return 0;
    }
    vm_of(vm_bits).interned_map_key_list(w) as i64
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_list_indices(vm_bits: i64, w: i64) -> i64 {
    let Some(w) = slot_leaf(w) else {
        return 0;
    };
    if unsafe { w_kind(w) } != CelKind::List {
        return 0;
    }
    vm_of(vm_bits).interned_list_index_list(w) as i64
}

#[allow(dead_code)]
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn vm_sync_binop(vm_bits: i64, w: i64) {
    vm_of(vm_bits).sync_pop_push_interned(2, w as usize as CelRef);
}

#[allow(dead_code)]
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn vm_sync_replace(vm_bits: i64, w: i64) {
    vm_of(vm_bits).sync_pop_push_interned(1, w as usize as CelRef);
}

#[allow(dead_code)]
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn vm_sync_push(vm_bits: i64, w: i64) {
    vm_of(vm_bits).sync_push_interned(w as usize as CelRef);
}

#[allow(dead_code)]
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn vm_sync_store(vm_bits: i64, slot: i64, w: i64) {
    vm_of(vm_bits).sync_store_interned(slot as u32, w as usize as CelRef);
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn vm_park_return(vm_bits: i64, w: CelRef) {
    vm_of(vm_bits).park_return(w);
}

#[allow(dead_code)]
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn vm_sync_write_local(vm_bits: i64, slot: i64, w: i64) {
    vm_of(vm_bits).sync_write_local(slot as u32, w as usize as CelRef);
}

#[allow(dead_code)]
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn vm_sync_pop(vm_bits: i64) {
    vm_of(vm_bits).sync_pop();
}

/// One instruction of the existing evaluator, residual.
///
/// `And`/`Or` are not hot-match arms (that grouping cost a fixed ~1 ns on
/// `LoadVar`). They land here with every unmatched opcode; interned `&&`/`||`
/// divert to [`portal_rare`] before hydrate.
///
/// `#[inline(never)]` keeps this a single default-arm callee in the
/// interpreter (same jump-table group as s15). The tracer still walks it
/// (`inline_ref`) so compiled `&&`/`||` call [`portal_rare`] directly.
#[inline(never)]
fn residual_dispatch(vm_bits: i64, pc: i64) -> i64 {
    let vm = unsafe { &mut *(vm_bits as usize as *mut Vm<'_>) };
    // `force_virtualizable_if_necessary`: clear `TOKEN_TRACING_RESCALL`
    // so `vable_after_residual_call` reloads the boxes this call writes.
    unsafe { force_virtualizable_if_necessary(vm.cel_frame) };
    let opcode = insn_op(vm.code, pc as usize);
    if opcode == OP_AND || opcode == OP_OR {
        let frame = unsafe { &mut *vm.cel_frame };
        return portal_rare(frame, vm_bits, vm.code, pc as usize, pc, opcode);
    }
    residual_hydrate(vm_bits, pc)
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn residual_hydrate(vm_bits: i64, pc: i64) -> i64 {
    let vm = unsafe { &mut *(vm_bits as usize as *mut Vm<'_>) };
    unsafe { force_virtualizable_if_necessary(vm.cel_frame) };
    vm.hydrate_from_cells();
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

/// Back-edge / function-entry threshold for this process.
///
/// Default `1_000_000` keeps unit tests on the native portal loop.
/// `CEL_PORTAL_THRESHOLD` overrides both counters, the same single-knob
/// shape `new_driver_f` uses.
fn portal_threshold() -> u32 {
    std::env::var("CEL_PORTAL_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000)
}

struct DriverEntry {
    id: u64,
    live: std::sync::Weak<super::code::CodeLive>,
    driver: JitDriver<PortalState>,
}

struct PortalTable {
    /// Id of the last driver this thread ran. Zero means none.
    last_id: u64,
    last_idx: usize,
    entries: Vec<DriverEntry>,
}

impl PortalTable {
    const fn new() -> Self {
        PortalTable {
            last_id: 0,
            last_idx: 0,
            entries: Vec::new(),
        }
    }

    fn sweep_dead(&mut self) {
        self.entries.retain(|e| e.live.strong_count() > 0);
        self.last_id = 0;
        self.last_idx = 0;
    }

    fn index_for(
        &mut self,
        id: u64,
        live: &std::sync::Arc<super::code::CodeLive>,
        state: &mut PortalState,
        code: &CelCode,
    ) -> usize {
        if self.last_id == id {
            return self.last_idx;
        }
        if let Some(i) = self.entries.iter().position(|e| e.id == id) {
            self.last_id = id;
            self.last_idx = i;
            return i;
        }
        self.sweep_dead();
        self.entries.push(DriverEntry {
            id,
            live: std::sync::Arc::downgrade(live),
            driver: fresh_portal_driver(state, code),
        });
        self.last_id = id;
        self.last_idx = self.entries.len() - 1;
        self.last_idx
    }
}

thread_local! {
    /// Vm pointer for [`step_hot`]. The dispatch arm only passes the
    /// portal greens `program` and `pc`.
    static PORTAL_VM: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
    /// The `JitDriver` [`run_cel_portal`] is running. `step_hot` forces a
    /// compiled token through this pointer: the driver is already borrowed
    /// by the portal loop, so the force cannot take a second `RefCell` borrow.
    static ACTIVE_PORTAL_DRIVER: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Address of this byte is this thread's owner token. Const-initialised,
    /// so the first `with` is a TLS lookup, not a constructor.
    static OWNER_TOKEN: u8 = const { 0u8 };
    /// Drivers for programs this thread did not claim. Keyed by code id.
    /// `JitDriver` is not `Send`; a program another thread owns is reached
    /// through this table instead of the code-owned pointer.
    static PORTAL_DRIVER: std::cell::RefCell<PortalTable> =
        const { std::cell::RefCell::new(PortalTable::new()) };
}

#[inline(always)]
fn thread_owner_token() -> usize {
    OWNER_TOKEN.with(|b| b as *const u8 as usize)
}

pub(crate) fn driver_table_len() -> usize {
    PORTAL_DRIVER.with(|cell| cell.borrow().entries.len())
}

fn fresh_portal_driver(state: &mut PortalState, code: &CelCode) -> JitDriver<PortalState> {
    let threshold = portal_threshold();
    let mut driver = JitDriver::new(threshold);
    driver.set_param("function_threshold", i64::from(threshold));
    // `GcLLDescr_boehm`: vtable at offset 0, `malloc_fixedsize` into CelHeap.
    // No collector — `collector_installed` stays false and the off-GC
    // jitframe token path is unchanged.
    driver.set_vtable_offset(Some(0));
    majit_gc::set_malloc_fixedsize(Some(crate::runtime::heap::cel_malloc_fixedsize));
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, code)
            .install_canonical_liveness(&mut driver);
    }
    driver
}

unsafe fn drop_portal_driver(ptr: usize) {
    drop(Box::from_raw(ptr as *mut JitDriver<PortalState>));
}

/// The driver stored on `code`, created on first use by the owner thread.
///
/// # Safety
///
/// The caller is the owner thread and `jit.in_use` is true for the
/// duration of the returned borrow, so no other call holds a mutable
/// reference to the same driver.
#[allow(clippy::mut_from_ref)]
unsafe fn driver_on_code<'a>(
    jit: &'a super::code::CodeJit,
    state: &mut PortalState,
    code: &CelCode,
) -> &'a mut JitDriver<PortalState> {
    let slot = &mut *jit.driver.get();
    if *slot == 0 {
        let boxed = Box::new(fresh_portal_driver(state, code));
        jit.drop_fn.set(Some(drop_portal_driver));
        *slot = Box::into_raw(boxed) as usize;
    }
    &mut *(*slot as *mut JitDriver<PortalState>)
}

/// `compile.py ResumeGuardForcedDescr.force_now` on the driver the portal
/// loop is running. The token is the address compiled code stored in
/// `vable_token` before a `may_force` call.
///
/// # Safety
///
/// `token` is a live force token, and [`run_cel_portal`] has published its
/// driver for this thread.
pub(crate) unsafe fn force_portal_driver_token(token: u64) {
    let ptr = ACTIVE_PORTAL_DRIVER.with(|cell| cell.get());
    assert!(ptr != 0, "may_force residual with no portal driver");
    unsafe {
        (*(ptr as *mut JitDriver<PortalState>)).force_virtualizable_token(token);
    }
}

struct ActiveDriverGuard(usize);

impl Drop for ActiveDriverGuard {
    fn drop(&mut self) {
        ACTIVE_PORTAL_DRIVER.with(|cell| cell.set(self.0));
    }
}

fn call_portal(
    driver: &mut JitDriver<PortalState>,
    code: &CelCode,
    state: &mut PortalState,
) -> i64 {
    let prev = ACTIVE_PORTAL_DRIVER
        .with(|cell| cell.replace(driver as *mut JitDriver<PortalState> as usize));
    let _guard = ActiveDriverGuard(prev);
    run_cel_portal(driver, code, state, 0)
}

#[inline(always)]
fn run_owned_driver(jit: &super::code::CodeJit, code: &CelCode, state: &mut PortalState) -> i64 {
    if jit.in_use.get() {
        let mut driver = fresh_portal_driver(state, code);
        return call_portal(&mut driver, code, state);
    }
    jit.in_use.set(true);
    let in_use = &jit.in_use as *const std::cell::Cell<bool>;
    struct Guard(*const std::cell::Cell<bool>);
    impl Drop for Guard {
        fn drop(&mut self) {
            unsafe { (*self.0).set(false) };
        }
    }
    let _g = Guard(in_use);
    let driver = unsafe { driver_on_code(jit, state, code) };
    call_portal(driver, code, state)
}

fn run_table_driver(code: &CelCode, state: &mut PortalState) -> i64 {
    let id = code.identity.id;
    let live = &code.identity.live;
    PORTAL_DRIVER.with(|slot| match slot.try_borrow_mut() {
        Ok(mut table) => {
            let i = table.index_for(id, live, state, code);
            call_portal(&mut table.entries[i].driver, code, state)
        }
        Err(_) => {
            let mut driver = fresh_portal_driver(state, code);
            call_portal(&mut driver, code, state)
        }
    })
}

/// Evaluate `code` through the portal loop.
#[inline(always)]
pub(crate) fn eval_through_portal(
    vm: &mut Vm<'_>,
    code: &CelCode,
) -> Result<Value, ExecutionError> {
    let mut state = PortalState {
        frame: vm.cel_frame as usize,
        vm: vm as *mut Vm<'_> as i64,
        ret: 0,
    };
    // A host call can re-enter this function on the same thread. The nested
    // evaluation must not leave its VM in the thread-local, or the outer
    // `RETURN` parks on that VM and this one finishes with no result.
    let prev_vm = PORTAL_VM.with(|cell| cell.replace(state.vm));
    struct PortalVmGuard(i64);
    impl Drop for PortalVmGuard {
        fn drop(&mut self) {
            PORTAL_VM.with(|cell| cell.set(self.0));
        }
    }
    let _portal_vm = PortalVmGuard(prev_vm);
    // Census is not installed here — that hook is process-global and
    // would clobber the columnar machine.
    let jit = &code.identity.live.jit;
    let token = thread_owner_token();
    let owner = jit.owner.load(std::sync::atomic::Ordering::Acquire);
    let bits = if owner == token {
        run_owned_driver(jit, code, &mut state)
    } else if owner == 0 {
        match jit.owner.compare_exchange(
            0,
            token,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        ) {
            Ok(_) => run_owned_driver(jit, code, &mut state),
            Err(actual) if actual == token => run_owned_driver(jit, code, &mut state),
            Err(_) => run_table_driver(code, &mut state),
        }
    } else {
        run_table_driver(code, &mut state)
    };
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

/// [`CelClass`] with `kind` spelled as the byte it is, so the field read
/// registers that width. The name word in front is the same one [`CelClass`]
/// carries; the asserts pin the two layouts together.
#[majit_macros::jit_immutable_fields(kind)]
#[repr(C)]
struct ClassKindView {
    _name: &'static str,
    kind: u8,
}

const _: () = {
    use crate::runtime::object::CelClass;
    use core::mem::{offset_of, size_of};
    assert!(offset_of!(ClassKindView, kind) == offset_of!(CelClass, kind));
    assert!(size_of::<ClassKindView>() == size_of::<CelClass>());
};

/// Family of `w`, or `-1` when `w` is null.
///
/// The class word and the family's byte are field reads. Both fields are
/// immutable, so a constant `w` folds them away.
#[majit_macros::jit_inline(
    ref_fields = {
        crate::runtime::object::CelObject::ob_type => crate::runtime::object::CelClass,
    },
    int_fields = { ClassKindView::kind => u8 },
)]
fn cell_kind(w: *mut CelObject) -> i64 {
    if w.is_null() {
        -1
    } else {
        let obj = w as *mut CelObject;
        let cls = unsafe { (*obj).ob_type };
        let view = cls as *const ClassKindView;
        (unsafe { (*view).kind }) as i64
    }
}

/// Payload of an int leaf.
///
/// `intval` is immutable, so a constant `w` folds the read.
#[majit_macros::jit_inline(int_fields = { W_IntObject::intval => i64 })]
fn cell_int(w: *mut CelObject) -> i64 {
    let obj = w as *mut W_IntObject;
    unsafe { (*obj).intval }
}

/// `1` / `0` for a bool leaf, `-1` otherwise.
///
/// Kind and `boolval` are field reads (`rewrite_op_getfield`). A constant
/// receiver folds both.
#[majit_macros::jit_inline(
    int_fields = { W_BoolObject::boolval => i64 },
    calls = { cell_kind => inline_int },
)]
fn cell_bool(w: *mut CelObject) -> i64 {
    if cell_kind(w) != CelKind::Bool as i64 {
        -1
    } else {
        let obj = w as *mut W_BoolObject;
        let bit = unsafe { (*obj).boolval };
        if bit != 0 {
            1
        } else {
            0
        }
    }
}

/// Length of a list leaf. `W_ListObject::length` is written once on the
/// source list the loop reads (`rewrite_op_getfield`).
#[majit_macros::jit_inline(int_fields = { crate::runtime::object::W_ListObject::length => i64 })]
fn cell_list_len(w: *mut CelObject) -> i64 {
    let obj = w as *mut crate::runtime::object::W_ListObject;
    unsafe { (*obj).length }
}

/// Length of a map leaf. `ll_len` / `W_MapObject.length` (`rewrite_op_getfield`).
#[majit_macros::jit_inline(int_fields = { crate::runtime::object::W_MapObject::length => i64 })]
fn cell_map_len(w: *mut CelObject) -> i64 {
    let obj = w as *mut crate::runtime::object::W_MapObject;
    unsafe { (*obj).length }
}

/// `list.size` / `size(list)` when the receiver's family is known.
///
/// A list or map length is `getfield_gc` (`ll_length`). Anything else stays
/// on [`interned_len_cell`].
#[majit_macros::jit_inline(calls = {
    cell_kind => inline_int,
    cell_list_len => inline_int,
    cell_map_len => inline_int,
    interned_len_cell => residual_int,
})]
fn trace_len_cell(w: *mut CelObject) -> i64 {
    let kind = cell_kind(w);
    if kind == CelKind::List as i64 {
        cell_list_len(w)
    } else if kind == CelKind::Map as i64 {
        cell_map_len(w)
    } else {
        interned_len_cell(w)
    }
}

/// Concrete `struct_allocs` target for [`box_int`]. Small ints stay the
/// prebuilt singletons; the traced body allocates a fresh leaf instead.
fn alloc_traced_int(_header: CelObject, intval: i64) -> *mut W_IntObject {
    crate::runtime::heap::with_heap(|heap| new_int_in(heap, intval))
}

/// Box `n`. The traced body is `new_with_vtable` of `CEL_INT_CLASS` plus
/// `setfield_gc` of `intval` (`rewrite_op_malloc`). The concrete body is
/// [`new_int_in`] via `struct_allocs`, so small ints stay interned.
#[cfg_attr(
    feature = "jit",
    majit_macros::jit_inline(
        inlined_prefix = {
            W_IntObject::ob_header => crate::runtime::object::CelObject,
        },
        int_fields = { W_IntObject::intval => i64 },
        struct_allocs = {
            W_IntObject => alloc_traced_int,
        },
    )
)]
#[allow(unused_variables)]
fn box_int(vm: i64, n: i64) -> *mut CelObject {
    let w = W_IntObject {
        ob_header: CelObject {
            ob_type: &CEL_INT_CLASS,
        },
        intval: n,
    };
    w as *mut W_IntObject as *mut CelObject
}

/// Box a 0/1 bit as one of the two prebuilt bool leaves.
///
/// The branch is `rewrite_op_same_as` of a constant pointer
/// (`rewrite_op_cast_pointer`): the taken arm is `new_bool`'s singleton,
/// not an allocation.
#[majit_macros::jit_inline]
fn box_bool(bit: i64) -> *mut CelObject {
    if bit != 0 {
        new_bool(true) as *mut CelObject
    } else {
        new_bool(false) as *mut CelObject
    }
}

/// `1` / `0` for an int comparison, `-1` when `op` is not one.
///
/// `jtransform.py` `rewrite_op_int_lt` / `int_eq` on the unboxed words.
#[majit_macros::jit_inline]
fn trace_cmp_bit(op: i64, l: i64, r: i64) -> i64 {
    if op == OP_EQ {
        if l == r {
            1
        } else {
            0
        }
    } else if op == OP_NE {
        if l != r {
            1
        } else {
            0
        }
    } else if op == OP_LT {
        if l < r {
            1
        } else {
            0
        }
    } else if op == OP_LE {
        if l <= r {
            1
        } else {
            0
        }
    } else if op == OP_GT {
        if l > r {
            1
        } else {
            0
        }
    } else if op == OP_GE {
        if l >= r {
            1
        } else {
            0
        }
    } else {
        -1
    }
}

/// `1` when `op` is an int arithmetic the fast path can finish.
///
/// A positive divisor has no `MIN / -1` overflow, and `checked_div` /
/// `checked_rem` are the truncating llops (`opimpl.py` `op_int_floordiv`
/// / `op_int_mod`). A non-positive divisor or an add/sub/mul overflow
/// returns `0` so the caller takes [`residual_dispatch`].
#[majit_macros::jit_inline]
fn trace_arith_ok(op: i64, l: i64, r: i64) -> i64 {
    if op == OP_ADD {
        match l.checked_add(r) {
            Some(_v) => 1,
            None => 0,
        }
    } else if op == OP_SUB {
        match l.checked_sub(r) {
            Some(_v) => 1,
            None => 0,
        }
    } else if op == OP_MUL {
        match l.checked_mul(r) {
            Some(_v) => 1,
            None => 0,
        }
    } else if op == OP_DIV || op == OP_MOD {
        if r > 0 {
            1
        } else {
            0
        }
    } else {
        0
    }
}

/// Unboxed int result. Only called when [`trace_arith_ok`] returned `1`.
#[majit_macros::jit_inline]
fn trace_arith_word(op: i64, l: i64, r: i64) -> i64 {
    if op == OP_ADD {
        match l.checked_add(r) {
            Some(v) => v,
            None => 0,
        }
    } else if op == OP_SUB {
        match l.checked_sub(r) {
            Some(v) => v,
            None => 0,
        }
    } else if op == OP_MUL {
        match l.checked_mul(r) {
            Some(v) => v,
            None => 0,
        }
    } else if op == OP_DIV {
        l / r
    } else {
        l % r
    }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn append_cell(list: CelRef, item: CelRef) -> i64 {
    try_append(list as i64, item as i64)
}

/// `_ll_list_resize_ge`. Grows the int column; the caller stores the word.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn list_resize_ge_i(list: *mut CelObject, newsize: i64) -> i64 {
    if list.is_null() {
        0
    } else {
        unsafe { i64::from(list_resize_ge(list, newsize)) }
    }
}

/// Items-block capacity of an empty object list, or 1. The hint
/// `new_list_with_capacity` stored (`ll_newlist_hint`).
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn empty_list_hint(list: *mut CelObject) -> i64 {
    if list.is_null() {
        return 1;
    }
    unsafe {
        let leaf = &*list.cast::<crate::runtime::object::W_ListObject>();
        let cap = crate::runtime::object_array::items_capacity(leaf.items) as i64;
        if cap > 0 {
            cap
        } else {
            1
        }
    }
}

/// `IntegerListStrategy.get_empty_storage`: the one malloc on
/// `switch_to_correct_strategy`. The strategy tag is written by the caller.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn alloc_int_column(cap: i64) -> *mut CelObject {
    crate::runtime::object::new_int_column_capacity(cap) as *mut CelObject
}

/// First int on an empty list (`EmptyListStrategy.append`) or a grow the
/// inline path declined. Not on the traced common path.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn list_append_int_cold(list: *mut CelObject, word: i64) -> i64 {
    if list.is_null() {
        return 0;
    }
    unsafe {
        let strategy = (*list.cast::<crate::runtime::object::W_ListObject>()).strategy;
        let length = (*list.cast::<crate::runtime::object::W_ListObject>()).length;
        if strategy == ListStrategy::Ints {
            i64::from(list_store_int(list, word))
        } else if strategy == ListStrategy::Object && length == 0 {
            i64::from(list_promote_empty_to_ints(list, word))
        } else {
            0
        }
    }
}

/// `IntegerListStrategy.append`: store the unboxed word.
///
/// Common path is a length/capacity read, a guard, `setarrayitem_gc`
/// (`rewrite_op_setarrayitem`) and `setfield_gc` of `length`
/// (`rewrite_op_setfield`). The grow is [`list_resize_ge_i`].
#[majit_macros::jit_inline(
    ref_params = { list: ref(crate::runtime::object::W_ListObject) },
    ref_fields = {
        crate::runtime::object::W_ListObject::storage => crate::runtime::object::CelObject,
    },
    array_fields = { crate::runtime::object::W_IntColumn::data => i64 },
    int_fields = {
        crate::runtime::object::W_ListObject::strategy => u8,
        crate::runtime::object::W_ListObject::length => i64,
        crate::runtime::object::W_ListObject::start => i64,
        crate::runtime::object::W_IntColumn::length => i64,
    },
    calls = {
        list_resize_ge_i => residual_int,
        list_append_int_cold => residual_int,
        empty_list_hint => residual_int,
        alloc_int_column => residual_ref,
    },
)]
fn append_int_word(list: *mut CelObject, word: i64) -> i64 {
    let strategy = list.strategy as u8 as i64;
    if strategy == ListStrategy::Ints as i64 {
        let storage = list.storage;
        if (storage as *mut u8) != core::ptr::null_mut() {
            let col = storage as *mut crate::runtime::object::W_IntColumn;
            let length = list.length;
            let cap = col.length;
            if length >= 0 {
                if length < cap {
                    col.data[length] = word;
                    list.length = length + 1;
                    1
                } else if list_resize_ge_i(list, length + 1) != 0 {
                    let storage2 = list.storage;
                    if (storage2 as *mut u8) != core::ptr::null_mut() {
                        let col2 = storage2 as *mut crate::runtime::object::W_IntColumn;
                        col2.data[length] = word;
                        list.length = length + 1;
                        1
                    } else {
                        list_append_int_cold(list, word)
                    }
                } else {
                    list_append_int_cold(list, word)
                }
            } else {
                list_append_int_cold(list, word)
            }
        } else {
            list_append_int_cold(list, word)
        }
    } else if strategy == ListStrategy::Object as i64 && list.length == 0 {
        // `EmptyListStrategy.switch_to_correct_strategy`: the tag is a
        // field write; only the column malloc is a call. No object-storage
        // fill happens first (`get_strategy_from_list_objects` picks Ints
        // before `init_from_list_w`).
        let hint = empty_list_hint(list);
        let col = alloc_int_column(hint);
        if (col as *mut u8) != core::ptr::null_mut() {
            list.strategy = ListStrategy::Ints;
            list.storage = col;
            list.start = 0;
            let column = col as *mut crate::runtime::object::W_IntColumn;
            column.data[0] = word;
            list.length = 1;
            1
        } else {
            list_append_int_cold(list, word)
        }
    } else {
        list_append_int_cold(list, word)
    }
}

/// `ObjectListStrategy.append` → `rlist.ll_append`.
///
/// The item is `setarrayitem_gc` (`rewrite_op_setarrayitem`). The block's
/// capacity is the length word in front of element 0 (`CelItemsBlock`).
/// A full block, or a column that is not the object strategy, returns `0`
/// so the caller can take [`append_int_word`] or [`append_cell`]
/// (`_ll_list_resize_ge` stays the residual).
#[majit_macros::jit_inline(
    ref_params = { list: ref(crate::runtime::object::W_ListObject) },
    ref_fields = {
        crate::runtime::object::W_ListObject::items => crate::runtime::object_array::CelItemsBlock,
    },
    array_fields = {
        crate::runtime::object::W_ListObject::items => crate::runtime::object::CelRef in crate::runtime::object_array::CelItemsBlock,
    },
    int_fields = {
        crate::runtime::object::W_ListObject::strategy => u8,
        crate::runtime::object::W_ListObject::length => i64,
        crate::runtime::object_array::CelItemsBlock::capacity => usize,
    },
)]
fn append_ref(list: *mut CelObject, item: *mut CelObject) -> i64 {
    let strategy = list.strategy as u8 as i64;
    if strategy == ListStrategy::Object as i64 {
        let items = list.items as *mut crate::runtime::object_array::CelItemsBlock;
        if (items as *mut u8) != core::ptr::null_mut() {
            let length = list.length;
            let cap = items.capacity as i64;
            if length >= 0 {
                if length < cap {
                    list.items[length] = item;
                    list.length = length + 1;
                    1
                } else {
                    0
                }
            } else {
                0
            }
        } else {
            0
        }
    } else {
        0
    }
}

/// First pair of a fresh object map: two `setarrayitem_gc` and `length`.
///
/// `dictmultiobject.py` `setitem` on an empty object map has no lookup.
/// A map that already has entries stays on [`map_insert_cell`].
#[majit_macros::jit_inline(
    ref_params = { map: ref(crate::runtime::object::W_MapObject) },
    ref_fields = {
        crate::runtime::object::W_MapObject::items => crate::runtime::object_array::CelItemsBlock,
    },
    array_fields = {
        crate::runtime::object::W_MapObject::items => crate::runtime::object::CelRef in crate::runtime::object_array::CelItemsBlock,
    },
    int_fields = {
        crate::runtime::object::W_MapObject::strategy => u8,
        crate::runtime::object::W_MapObject::length => i64,
        crate::runtime::object_array::CelItemsBlock::capacity => usize,
    },
    calls = {
        cell_kind => inline_int,
        map_insert_cell => residual_int,
    },
)]
fn map_store_pair(map: *mut CelObject, key: *mut CelObject, value: *mut CelObject) -> i64 {
    let strategy = map.strategy as u8 as i64;
    if strategy == crate::runtime::object::MapStrategy::Object as i64 {
        let length = map.length;
        if length == 0 {
            let items = map.items as *mut crate::runtime::object_array::CelItemsBlock;
            if (items as *mut u8) != core::ptr::null_mut() {
                let cap = items.capacity as i64;
                let key_kind = cell_kind(key);
                if cap >= 2 {
                    if key_kind == CelKind::Str as i64
                        || key_kind == CelKind::Int as i64
                        || key_kind == CelKind::Bool as i64
                        || key_kind == CelKind::UInt as i64
                    {
                        map.items[0] = key;
                        map.items[1] = value;
                        map.length = 1;
                        1
                    } else {
                        map_insert_cell(map, key, value)
                    }
                } else {
                    map_insert_cell(map, key, value)
                }
            } else {
                map_insert_cell(map, key, value)
            }
        } else {
            map_insert_cell(map, key, value)
        }
    } else {
        map_insert_cell(map, key, value)
    }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn map_insert_cell(map: CelRef, key: CelRef, value: CelRef) -> i64 {
    try_map_insert(map as i64, key as i64, value as i64)
}

/// `e + 1` as one traced add. Null declines to [`slow_pc`].
#[majit_macros::jit_inline(calls = {
    cell_kind => inline_int,
    cell_int => inline_int,
    box_int => inline_ref,
    trace_arith_ok => inline_int,
    trace_arith_word => inline_int,
})]
fn add_local_const_cell(vm: i64, a: *mut CelObject, k: *mut CelObject) -> *mut CelObject {
    if (a as *mut u8) == core::ptr::null_mut() {
        core::ptr::null_mut()
    } else if (k as *mut u8) == core::ptr::null_mut() {
        core::ptr::null_mut()
    } else if cell_kind(a) != CelKind::Int as i64 {
        core::ptr::null_mut()
    } else if cell_kind(k) != CelKind::Int as i64 {
        core::ptr::null_mut()
    } else {
        let l = cell_int(a);
        let rv = cell_int(k);
        if trace_arith_ok(OP_ADD, l, rv) != 0 {
            box_int(vm, trace_arith_word(OP_ADD, l, rv))
        } else {
            core::ptr::null_mut()
        }
    }
}

/// Interned string for green `names[name_idx]`. Null when the index misses.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn field_name_cell(program: &CelCode, name_idx: i64) -> *mut CelObject {
    if name_idx < 0 {
        return core::ptr::null_mut();
    }
    program.name_cell(NameId(name_idx as u32))
}

/// `1` when both cells are strings with the same bytes.
///
/// Elidable: string payloads are immutable.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn str_cells_eq(left: *mut CelObject, right: *mut CelObject) -> i64 {
    match (unsafe { string_as_str(left) }, unsafe {
        string_as_str(right)
    }) {
        (Some(a), Some(b)) if a == b => 1,
        _ => 0,
    }
}

/// Object-strategy map field. Null means miss or "not this strategy".
///
/// `dictmultiobject.py` walks the interleaved entry array. A full scan
/// that misses still returns null; [`map_object_known`] tells a miss
/// from a map this loop did not scan.
#[majit_macros::jit_inline(
    ref_params = { map: ref(crate::runtime::object::W_MapObject) },
    ref_fields = {
        crate::runtime::object::W_MapObject::items => crate::runtime::object_array::CelItemsBlock,
    },
    array_fields = {
        crate::runtime::object::W_MapObject::items => crate::runtime::object::CelRef in crate::runtime::object_array::CelItemsBlock,
    },
    int_fields = {
        crate::runtime::object::W_MapObject::strategy => u8,
        crate::runtime::object::W_MapObject::length => i64,
        crate::runtime::object_array::CelItemsBlock::capacity => usize,
    },
    calls = {
        str_cells_eq => elidable_int_cannot_raise,
    },
)]
fn map_object_field(map: *mut CelObject, name: *mut CelObject) -> *mut CelObject {
    let mut found_slot = -1i64;
    let strategy = map.strategy as u8 as i64;
    if strategy == crate::runtime::object::MapStrategy::Object as i64 {
        let items = map.items as *mut crate::runtime::object_array::CelItemsBlock;
        if (items as *mut u8) != core::ptr::null_mut() {
            let length = map.length;
            let cap = items.capacity as i64;
            let mut i = 0i64;
            while i < length {
                let slot = i + i;
                let val_at = slot + 1;
                if val_at < cap {
                    let key = map.items[slot];
                    if str_cells_eq(key, name) != 0 {
                        let delta = val_at - found_slot;
                        found_slot += delta;
                        i += length;
                    } else {
                        i += 1;
                    }
                } else {
                    i += length;
                }
            }
        }
    }
    if found_slot >= 0 {
        map.items[found_slot]
    } else {
        core::ptr::null_mut()
    }
}

/// `1` when [`map_object_field`] scanned an object-strategy map.
#[majit_macros::jit_inline(
    ref_params = { map: ref(crate::runtime::object::W_MapObject) },
    ref_fields = {
        crate::runtime::object::W_MapObject::items => crate::runtime::object_array::CelItemsBlock,
    },
    int_fields = {
        crate::runtime::object::W_MapObject::strategy => u8,
        crate::runtime::object::W_MapObject::length => i64,
    },
)]
fn map_object_known(map: *mut CelObject) -> i64 {
    let strategy = map.strategy as u8 as i64;
    if strategy == crate::runtime::object::MapStrategy::Object as i64 {
        let items = map.items as *mut crate::runtime::object_array::CelItemsBlock;
        if (items as *mut u8) != core::ptr::null_mut() {
            let length = map.length;
            if length >= 0 {
                1
            } else {
                0
            }
        } else {
            0
        }
    } else {
        0
    }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn alloc_list(vm: i64, cap: i64) -> CelRef {
    new_list_with_capacity_in(vm_heap(vm), cap) as CelRef
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn alloc_map(vm: i64, cap: i64) -> CelRef {
    new_map_with_capacity_in(vm_heap(vm), cap) as CelRef
}

/// Int-column `container[key]`. Anything else is null and the caller
/// residuals, which is `interned_index`.
#[majit_macros::jit_inline(calls = {
    cell_kind => inline_int,
    cell_int => inline_int,
    item_cell => inline_ref,
})]
fn index_cell(container: *mut CelObject, key: *mut CelObject) -> *mut CelObject {
    if cell_kind(container) == CelKind::List as i64 {
        if cell_kind(key) == CelKind::Int as i64 {
            item_cell(0, container, cell_int(key))
        } else {
            core::ptr::null_mut()
        }
    } else {
        core::ptr::null_mut()
    }
}

/// Element of an int-column list, boxed (`getarrayitem_gc_i`, then
/// `new_with_vtable`). Object lists and windows return null; the caller
/// residuals through `interned_item`.
#[majit_macros::jit_inline(
    ref_params = { list: ref(crate::runtime::object::W_ListObject) },
    ref_fields = {
        crate::runtime::object::W_ListObject::storage => crate::runtime::object::CelObject,
    },
    array_fields = {
        crate::runtime::object::W_IntColumn::data => i64,
        crate::runtime::object::W_ListObject::items => crate::runtime::object::CelRef in crate::runtime::object_array::CelItemsBlock,
    },
    int_fields = {
        crate::runtime::object::W_ListObject::strategy => u8,
        crate::runtime::object::W_ListObject::length => i64,
        crate::runtime::object::W_ListObject::start => i64,
        crate::runtime::object::W_IntColumn::length => i64,
    },
    calls = { box_int => inline_ref },
)]
fn item_cell(vm: i64, list: *mut CelObject, index: i64) -> *mut CelObject {
    let strategy = list.strategy as u8 as i64;
    let length = list.length;
    let start = list.start;
    let storage = list.storage;
    if strategy == crate::runtime::object::ListStrategy::Ints as i64 {
        if index >= 0 {
            if index < length {
                if (storage as *mut u8) != core::ptr::null_mut() {
                    let col = storage as *mut crate::runtime::object::W_IntColumn;
                    let col_len = col.length;
                    let at = start + index;
                    if at >= 0 {
                        if at < col_len {
                            box_int(vm, col.data[at])
                        } else {
                            core::ptr::null_mut()
                        }
                    } else {
                        core::ptr::null_mut()
                    }
                } else {
                    core::ptr::null_mut()
                }
            } else {
                core::ptr::null_mut()
            }
        } else {
            core::ptr::null_mut()
        }
    } else if strategy == crate::runtime::object::ListStrategy::Object as i64 {
        // `listobject.py` `getitem` / `rlist.ll_getitem_fast`: bounds guard,
        // then `getarrayitem_gc` of the object column.
        if index >= 0 {
            if index < length {
                let at = start + index;
                if at >= 0 {
                    list.items[at]
                } else {
                    core::ptr::null_mut()
                }
            } else {
                core::ptr::null_mut()
            }
        } else {
            core::ptr::null_mut()
        }
    } else {
        core::ptr::null_mut()
    }
}

/// `1` when `names[idx]` is `int`.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_name_is_int(program: &CelCode, idx: i64) -> i64 {
    i64::from(program.name(NameId(idx as u32)) == Some("int"))
}

/// `W_IntObject.int`: the same leaf after the class check. Null declines.
#[majit_macros::jit_inline(calls = { cell_kind => inline_int })]
fn int_identity(w: *mut CelObject) -> *mut CelObject {
    if cell_kind(w) == CelKind::Int as i64 {
        w
    } else {
        core::ptr::null_mut()
    }
}

/// `1` when `names[idx]` is `double`.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_is_double(program: &CelCode, idx: i64) -> i64 {
    i64::from(program.name(NameId(idx as u32)) == Some("double"))
}

/// `1` when `names[idx]` is `string`.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn interned_is_string(program: &CelCode, idx: i64) -> i64 {
    i64::from(program.name(NameId(idx as u32)) == Some("string"))
}

/// Concrete `struct_allocs` target for [`box_double`].
fn alloc_traced_double(_header: CelObject, floatval: f64) -> *mut W_DoubleObject {
    crate::runtime::heap::with_heap(|heap| new_double_in(heap, floatval))
}

/// `space.newfloat`: `new_with_vtable` of `CEL_DOUBLE_CLASS` plus
/// `setfield_gc_f` of `floatval` (`jtransform.py` `rewrite_op_malloc`,
/// `rewrite_op_setfield`). The concrete body is [`new_double_in`].
#[cfg_attr(
    feature = "jit",
    majit_macros::jit_inline(
        inlined_prefix = {
            W_DoubleObject::ob_header => crate::runtime::object::CelObject,
        },
        struct_allocs = {
            W_DoubleObject => alloc_traced_double,
        },
    )
)]
#[allow(unused_variables)]
fn box_double(vm: i64, n: f64) -> *mut CelObject {
    let w = W_DoubleObject {
        ob_header: CelObject {
            ob_type: &CEL_DOUBLE_CLASS,
        },
        floatval: n,
    };
    w as *mut W_DoubleObject as *mut CelObject
}

/// `W_IntObject.descr_float`: class check, `cast_int_to_float`, `newfloat`.
/// `W_FloatObject.descr_float` is the same leaf. Null declines.
#[majit_macros::jit_inline(calls = {
    cell_kind => inline_int,
    cell_int => inline_int,
    box_double => inline_ref,
})]
fn double_from_cell(vm: i64, w: *mut CelObject) -> *mut CelObject {
    if cell_kind(w) == CelKind::Int as i64 {
        let n = cell_int(w);
        box_double(vm, n as f64)
    } else if cell_kind(w) == CelKind::Double as i64 {
        w
    } else {
        core::ptr::null_mut()
    }
}

/// `ll_int2dec` (`ll_str.py`): one residual, not traced. The string leaf
/// comes back directly (`descr_str` → `space.newtext`).
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn int_to_text(n: i64) -> *mut CelObject {
    new_string(&n.to_string()) as *mut CelObject
}

/// `W_StringObject` + `W_StringObject` via `cel_add`. Residual. Null declines.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn string_add_cell(a: *mut CelObject, b: *mut CelObject) -> *mut CelObject {
    let r = unsafe { cel_add(a, b) };
    if r.is_null() || r == ERROR_SENTINEL {
        core::ptr::null_mut()
    } else {
        r
    }
}

/// Two doubles: `cel_div` / ordered compare. Residual so the leaves are
/// forced before the read (`descr_truediv` / `descr_lt`). Null declines.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn double_binop_cell(op: i64, a: *mut CelObject, b: *mut CelObject) -> *mut CelObject {
    let r = unsafe {
        if op == OP_DIV {
            cel_div(a, b)
        } else if op == OP_ADD {
            cel_add(a, b)
        } else if op == OP_SUB {
            cel_sub(a, b)
        } else if op == OP_MUL {
            cel_mul(a, b)
        } else if op == OP_EQ {
            cel_equals(a, b)
        } else if op == OP_NE {
            cel_not_equals(a, b)
        } else if op == OP_LT {
            cel_less(a, b)
        } else if op == OP_LE {
            cel_less_equals(a, b)
        } else if op == OP_GT {
            cel_greater(a, b)
        } else if op == OP_GE {
            cel_greater_equals(a, b)
        } else {
            ERROR_SENTINEL
        }
    };
    if r.is_null() || r == ERROR_SENTINEL {
        core::ptr::null_mut()
    } else {
        r
    }
}

/// `W_IntObject.descr_str` for an int. A string is the same leaf. Null declines.
#[majit_macros::jit_inline(calls = {
    cell_kind => inline_int,
    cell_int => inline_int,
    int_to_text => residual_ref,
})]
fn string_from_cell(w: *mut CelObject) -> *mut CelObject {
    if cell_kind(w) == CelKind::Int as i64 {
        int_to_text(cell_int(w))
    } else if cell_kind(w) == CelKind::Str as i64 {
        w
    } else {
        core::ptr::null_mut()
    }
}

/// `IntegerListStrategy._safe_contains`: `2` hit, `1` miss, `0` decline.
///
/// Plain scan. A constant column is unrolled by the tracer
/// (`loop_unrolling_heuristic`), the same way `_safe_find_or_count` is.
#[majit_macros::jit_inline(
    ref_params = { list: ref(crate::runtime::object::W_ListObject) },
    ref_fields = {
        crate::runtime::object::W_ListObject::storage => crate::runtime::object::CelObject,
    },
    array_fields = { crate::runtime::object::W_IntColumn::data => i64 },
    int_fields = {
        crate::runtime::object::W_ListObject::strategy => u8,
        crate::runtime::object::W_ListObject::length => i64,
        crate::runtime::object::W_ListObject::start => i64,
        crate::runtime::object::W_IntColumn::length => i64,
    },
    calls = {
        cell_kind => inline_int,
        cell_int => inline_int,
    },
)]
fn contains_int_word(list: *mut CelObject, needle: *mut CelObject) -> i64 {
    if cell_kind(list) != CelKind::List as i64 {
        0
    } else if cell_kind(needle) != CelKind::Int as i64 {
        0
    } else {
        let strategy = list.strategy as u8 as i64;
        if strategy == ListStrategy::Ints as i64 {
            let length = list.length;
            let start = list.start;
            let storage = list.storage;
            if (storage as *mut u8) == core::ptr::null_mut() {
                0
            } else {
                let col = storage as *mut crate::runtime::object::W_IntColumn;
                let col_len = col.length;
                let want = cell_int(needle);
                let stop = start + length;
                if start < 0 {
                    0
                } else if stop > col_len {
                    0
                } else {
                    // `_safe_find_or_count`: the compare is a value, so a
                    // constant column unrolls to one `IntEq` per element
                    // instead of a side exit that drops the last hit.
                    let mut i = 0;
                    let mut hits = 0;
                    while i < length {
                        let at = start + i;
                        hits += (col.data[at] == want) as i64;
                        i += 1;
                    }
                    1 + ((hits != 0) as i64)
                }
            }
        } else {
            0
        }
    }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn map_keys_cell(vm: i64, w: CelRef) -> CelRef {
    let keys = interned_map_keys(vm, w as i64);
    if keys == 0 {
        core::ptr::null_mut()
    } else {
        keys as usize as CelRef
    }
}

/// `1` when `w` is an empty optional. A null payload is the none case.
#[majit_macros::jit_inline(
    ref_fields = {
        W_OptionalObject::w_value => CelObject,
    },
    calls = { cell_kind => inline_int },
)]
fn opt_is_none_i(w: *mut CelObject) -> i64 {
    if cell_kind(w) == CelKind::Optional as i64 {
        let obj = w as *mut W_OptionalObject;
        let inner = unsafe { (*obj).w_value };
        if inner.is_null() {
            1
        } else {
            0
        }
    } else {
        0
    }
}

/// Interned left of `&&` / `||`, as a traced kind + `boolval` read.
/// `1` short-circuits, `2` falls through, `0` declines.
#[majit_macros::jit_inline(calls = { cell_bool => inline_int })]
fn bool_short_i(w: *mut CelObject, is_or: i64) -> i64 {
    let bit = cell_bool(w);
    if bit < 0 {
        0
    } else if bit != 0 {
        if is_or != 0 {
            1
        } else {
            2
        }
    } else if is_or != 0 {
        2
    } else {
        1
    }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn keep_right_merge_i(vm: i64, slot: i64, is_or: i64) -> i64 {
    keep_right_merge(vm, slot, is_or != 0)
}

/// `1` when an `&&` / `||` merge can keep the right-hand bool.
///
/// `scratch_bits == 0` is the absent-scratch arm of [`keep_right_merge`]
/// (`InternalError` → keep). A hydrated slot takes [`keep_right_merge_i`].
#[majit_macros::jit_inline(calls = {
    cell_kind => inline_int,
    keep_right_merge_i => residual_int,
})]
fn and_merge_keep(vm: i64, w: *mut CelObject, slot: i64, is_or: i64, scratch_bits: i64) -> i64 {
    if cell_kind(w) != CelKind::Bool as i64 {
        0
    } else if scratch_bits == 0 {
        1
    } else {
        if keep_right_merge_i(vm, slot, is_or) != 0 {
            1
        } else {
            0
        }
    }
}

/// Every opcode of one portal step.
///
/// The dispatch JitCode lowers the match arm, not these bodies. The
/// arm calls this helper and writes `pc` from the value it returns.
/// A `pc` store after the match is outside that arm, so the compiled
/// loop would repeat one opcode.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn step_hot(program: &CelCode, pc: usize) -> i64 {
    let vm = PORTAL_VM.with(|cell| cell.get());
    let here = pc as i64;
    let opcode = insn_op(program, pc);
    let frame = unsafe { &mut *vm_of(vm).cel_frame };
    unsafe { force_virtualizable_if_necessary(frame) };
    frame.last_instr = here;
    match opcode {
        OP_LOAD_VAR => {
            let w = intern_var(vm, program, insn_a(program, pc));
            if w.is_null() {
                residual_dispatch(vm, here)
            } else {
                let depth = frame.valuestackdepth;
                frame.locals_stack_w[depth] = w;
                frame.valuestackdepth = depth + 1;
                here + 1
            }
        }
        OP_GET_FIELD => match operand_cell(frame, 1) {
            Some(recv) if !recv.is_null() => {
                let found = interned_field(recv as i64, program, insn_a(program, pc));
                if found == 0 {
                    residual_dispatch(vm, here)
                } else {
                    let depth = frame.valuestackdepth;
                    let r = found as usize as CelRef;
                    frame.locals_stack_w[depth - 1] = r;
                    here + 1
                }
            }
            _ => residual_dispatch(vm, here),
        },
        OP_HAS_FIELD
        | OP_HAS_FIELD_LOCAL
        | OP_HAS_FIELD_LOCAL_APPEND
        | OP_OPT_INDEX
        | OP_OPT_SELECT
        | OP_JUMP_IF_OPT_NONE
        | OP_LIST_APPEND_OPTIONAL
        | OP_NOT_STRICTLY_FALSE
        | OP_IN
        | OP_CALL_QUALIFIED
        | OP_AND_LOCAL
        | OP_OR_LOCAL
        | OP_AND_MERGE
        | OP_OR_MERGE
        | OP_NEW_MAP
        | OP_MAP_INSERT
        | OP_MAP_INSERT_OPTIONAL => portal_rare(frame, vm, program, pc, here, opcode),
        OP_GET_FIELD_LOCAL => {
            let recv = read_cell(frame, insn_a(program, pc));
            let found = interned_field(recv as i64, program, insn_b(program, pc));
            if recv.is_null() || found == 0 {
                residual_dispatch(vm, here)
            } else {
                let r = found as usize as CelRef;
                let depth = frame.valuestackdepth;
                frame.locals_stack_w[depth] = r;
                frame.valuestackdepth = depth + 1;
                here + 1
            }
        }
        OP_GET_FIELD_LOCAL_APPEND => {
            let recv = read_cell(frame, insn_a(program, pc));
            let found = interned_field(recv as i64, program, insn_b(program, pc));
            let depth = frame.valuestackdepth;
            let list = frame.locals_stack_w[depth - 1];
            if recv.is_null() || found == 0 || list.is_null() || try_append(list as i64, found) == 0
            {
                residual_dispatch(vm, here)
            } else {
                here + 1
            }
        }
        OP_INDEX => match (operand_cell(frame, 2), operand_cell(frame, 1)) {
            (Some(container), Some(key)) if !container.is_null() && !key.is_null() => {
                let item = interned_index(container as i64, key as i64);
                if item == 0 {
                    residual_dispatch(vm, here)
                } else {
                    let depth = frame.valuestackdepth;
                    let r = item as usize as CelRef;
                    frame.locals_stack_w[depth - 2] = r;
                    frame.valuestackdepth = depth - 1;
                    here + 1
                }
            }
            _ => residual_dispatch(vm, here),
        },
        OP_ACCU_LOOP_COND => {
            let w = frame.locals_stack_w[insn_a(program, pc)];
            if w.is_null() || unsafe { w_kind(w) } != CelKind::Bool {
                residual_dispatch(vm, here)
            } else if unsafe { (*w.cast::<W_BoolObject>()).boolval } == 0 {
                insn_b(program, pc)
            } else {
                here + 1
            }
        }
        OP_ACCU_LOOP_COND_NOT => {
            let w = frame.locals_stack_w[insn_a(program, pc)];
            if w.is_null() || unsafe { w_kind(w) } != CelKind::Bool {
                residual_dispatch(vm, here)
            } else if unsafe { (*w.cast::<W_BoolObject>()).boolval } != 0 {
                insn_b(program, pc)
            } else {
                here + 1
            }
        }
        OP_ADD_LOCAL_K_APPEND => {
            interned_arith_local_k_append!(frame, vm, program, pc, here, interned_add)
        }
        OP_MUL_LOCAL_K_APPEND => {
            interned_arith_local_k_append!(frame, vm, program, pc, here, interned_mul)
        }
        OP_MOD_LOCAL_K_APPEND => {
            interned_arith_local_k_append!(frame, vm, program, pc, here, interned_rem)
        }
        OP_EQ_LOCAL_K_APPEND => {
            interned_local_k_append!(frame, vm, program, pc, here, cel_equals)
        }
        OP_NE_LOCAL_K_APPEND => {
            interned_local_k_append!(frame, vm, program, pc, here, cel_not_equals)
        }
        OP_LT_LOCAL_K_APPEND => {
            interned_local_k_append!(frame, vm, program, pc, here, cel_less)
        }
        OP_GT_LOCAL_K_APPEND => {
            interned_local_k_append!(frame, vm, program, pc, here, cel_greater)
        }
        OP_GE_LOCAL_K_APPEND => {
            interned_local_k_append!(frame, vm, program, pc, here, cel_greater_equals)
        }
        OP_NOT => match operand_cell(frame, 1) {
            Some(w) if !w.is_null() && unsafe { w_kind(w) } == CelKind::Bool => {
                let r = unsafe { cel_negate(w) };
                if r == ERROR_SENTINEL {
                    residual_dispatch(vm, here)
                } else {
                    let depth = frame.valuestackdepth;
                    frame.locals_stack_w[depth - 1] = r;
                    here + 1
                }
            }
            _ => residual_dispatch(vm, here),
        },
        OP_NEGATE => match operand_cell(frame, 1) {
            Some(w) if !w.is_null() => {
                let r = unsafe { cel_negate(w) };
                if r == ERROR_SENTINEL {
                    residual_dispatch(vm, here)
                } else {
                    let depth = frame.valuestackdepth;
                    frame.locals_stack_w[depth - 1] = r;
                    here + 1
                }
            }
            _ => residual_dispatch(vm, here),
        },
        OP_LOAD_LOCAL_APPEND => {
            let w = read_cell(frame, insn_a(program, pc));
            match operand_cell(frame, 1) {
                Some(list)
                    if !w.is_null()
                        && !list.is_null()
                        && try_append(list as i64, w as i64) != 0 =>
                {
                    here + 1
                }
                _ => residual_dispatch(vm, here),
            }
        }
        OP_CALL_HOST | OP_CALL_METHOD => {
            let arity = insn_b(program, pc);
            let name = insn_a(program, pc);
            let unary_arity = if opcode == OP_CALL_HOST { 1 } else { 0 };
            if arity == unary_arity && interned_is_size(program, name) != 0 {
                match operand_cell(frame, 1) {
                    Some(w) if !w.is_null() => {
                        let n = interned_len(w as i64);
                        if n < 0 {
                            residual_dispatch(vm, here)
                        } else {
                            let depth = frame.valuestackdepth;
                            let r = new_int(n) as CelRef;
                            frame.locals_stack_w[depth - 1] = r;
                            here + 1
                        }
                    }
                    _ => residual_dispatch(vm, here),
                }
            } else if arity == unary_arity {
                match operand_cell(frame, 1) {
                    Some(w) if !w.is_null() => {
                        let mut out = interned_unary(vm, program, name, w as i64);
                        if out == 0 && opcode == OP_CALL_METHOD {
                            out = interned_temporal(vm, program, name, w as i64);
                        }
                        if out == 0 && opcode == OP_CALL_METHOD {
                            out = interned_optional_unary(program, name, w as i64);
                        }
                        if out == 0 || out == ERROR_SENTINEL as i64 {
                            residual_dispatch(vm, here)
                        } else {
                            let depth = frame.valuestackdepth;
                            let r = out as usize as CelRef;
                            frame.locals_stack_w[depth - 1] = r;
                            here + 1
                        }
                    }
                    _ => residual_dispatch(vm, here),
                }
            } else if opcode == OP_CALL_METHOD && arity == 1 {
                // After a CallQualified miss the arguments are parked and
                // only the receiver sits on the stack. `operand_cell(2)`
                // is then a local or the items-block header; decline.
                match (operand_cell(frame, 1), operand_cell(frame, 2)) {
                    (Some(recv), Some(arg)) if !recv.is_null() && !arg.is_null() => {
                        let out = interned_method1(program, name, recv as i64, arg as i64);
                        if out == 0 || out == ERROR_SENTINEL as i64 {
                            residual_dispatch(vm, here)
                        } else {
                            let depth = frame.valuestackdepth;
                            let r = out as usize as CelRef;
                            frame.locals_stack_w[depth - 2] = r;
                            frame.valuestackdepth = depth - 1;
                            here + 1
                        }
                    }
                    _ => residual_dispatch(vm, here),
                }
            } else {
                residual_dispatch(vm, here)
            }
        }
        OP_LOAD_LOCAL => {
            let slot = insn_a(program, pc);
            let w = frame.locals_stack_w[slot];
            if w.is_null() {
                residual_dispatch(vm, here)
            } else {
                let depth = frame.valuestackdepth;
                frame.locals_stack_w[depth] = w;
                frame.valuestackdepth = depth + 1;
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
                here + 1
            }
        }
        OP_ADD => interned_arith!(frame, vm, here, interned_add),
        OP_SUB => interned_arith!(frame, vm, here, interned_sub),
        OP_MUL => interned_arith!(frame, vm, here, interned_mul),
        OP_DIV => interned_arith!(frame, vm, here, interned_div),
        OP_MOD => interned_arith!(frame, vm, here, interned_rem),
        OP_EQ => interned_binop!(frame, vm, here, interned_equals),
        OP_NE => interned_binop!(frame, vm, here, interned_not_equals),
        OP_LT => interned_binop!(frame, vm, here, cel_less),
        OP_LE => interned_binop!(frame, vm, here, cel_less_equals),
        OP_GT => interned_binop!(frame, vm, here, cel_greater),
        OP_GE => interned_binop!(frame, vm, here, cel_greater_equals),
        OP_LOAD_CONST => {
            let w = intern_const(program, insn_a(program, pc));
            if w.is_null() {
                residual_dispatch(vm, here)
            } else {
                let depth = frame.valuestackdepth;
                frame.locals_stack_w[depth] = w;
                frame.valuestackdepth = depth + 1;
                here + 1
            }
        }
        OP_ADD_K => interned_arith_k!(frame, vm, program, pc, here, interned_add),
        OP_MUL_K => interned_arith_k!(frame, vm, program, pc, here, interned_mul),
        OP_MOD_K => interned_arith_k!(frame, vm, program, pc, here, interned_rem),
        OP_EQ_K => interned_binop_k!(frame, vm, program, pc, here, cel_equals),
        OP_NE_K => interned_binop_k!(frame, vm, program, pc, here, cel_not_equals),
        OP_LT_K => interned_binop_k!(frame, vm, program, pc, here, cel_less),
        OP_GT_K => interned_binop_k!(frame, vm, program, pc, here, cel_greater),
        OP_GE_K => interned_binop_k!(frame, vm, program, pc, here, cel_greater_equals),
        OP_ADD_LOCAL_K => interned_arith_local_k!(frame, vm, program, pc, here, interned_add),
        OP_MUL_LOCAL_K => interned_arith_local_k!(frame, vm, program, pc, here, interned_mul),
        OP_MOD_LOCAL_K => interned_arith_local_k!(frame, vm, program, pc, here, interned_rem),
        OP_EQ_LOCAL_K => interned_local_k!(frame, vm, program, pc, here, cel_equals),
        OP_NE_LOCAL_K => interned_local_k!(frame, vm, program, pc, here, cel_not_equals),
        OP_LT_LOCAL_K => interned_local_k!(frame, vm, program, pc, here, cel_less),
        OP_GT_LOCAL_K => interned_local_k!(frame, vm, program, pc, here, cel_greater),
        OP_GE_LOCAL_K => interned_local_k!(frame, vm, program, pc, here, cel_greater_equals),
        OP_INC_LOCAL => {
            let slot = insn_a(program, pc);
            let w = frame.locals_stack_w[slot];
            if w.is_null() || unsafe { w_kind(w) } != CelKind::Int {
                residual_dispatch(vm, here)
            } else {
                let n = unsafe { (*w.cast::<W_IntObject>()).intval };
                match n.checked_add(1) {
                    None => residual_dispatch(vm, here),
                    Some(next) => {
                        let r = new_int_in(vm_heap(vm), next) as CelRef;
                        frame.locals_stack_w[slot] = r;
                        here + 1
                    }
                }
            }
        }
        OP_JUMP => insn_a(program, pc),
        OP_JUMP_IF_FALSE | OP_JUMP_IF_TRUE => match operand_cell(frame, 1) {
            Some(w) if !w.is_null() && unsafe { w_kind(w) } == CelKind::Bool => {
                let truthy = unsafe { (*w.cast::<W_BoolObject>()).boolval } != 0;
                let depth = frame.valuestackdepth;
                frame.valuestackdepth = depth - 1;
                let want = opcode == OP_JUMP_IF_TRUE;
                if truthy == want {
                    insn_a(program, pc)
                } else {
                    here + 1
                }
            }
            _ => residual_dispatch(vm, here),
        },
        OP_NEW_LIST => {
            let w = new_list_with_capacity_in(vm_heap(vm), insn_a(program, pc)) as CelRef;
            let depth = frame.valuestackdepth;
            frame.locals_stack_w[depth] = w;
            frame.valuestackdepth = depth + 1;
            here + 1
        }
        OP_NEW_LIST_FROM_ARG => {
            let src = read_cell(frame, insn_a(program, pc));
            if src.is_null() || unsafe { w_kind(src) } != CelKind::List {
                residual_dispatch(vm, here)
            } else {
                let w = new_list_with_capacity_in(vm_heap(vm), unsafe { list_len(src) }) as CelRef;
                let depth = frame.valuestackdepth;
                frame.locals_stack_w[depth] = w;
                frame.valuestackdepth = depth + 1;
                here + 1
            }
        }
        OP_LIST_APPEND => {
            let depth = frame.valuestackdepth;
            let item = frame.locals_stack_w[depth - 1];
            let list = frame.locals_stack_w[depth - 2];
            if item.is_null() || list.is_null() || try_append(list as i64, item as i64) == 0 {
                residual_dispatch(vm, here)
            } else {
                frame.valuestackdepth = depth - 1;
                here + 1
            }
        }
        OP_ITER_ELEMS => {
            let depth = frame.valuestackdepth;
            let w = frame.locals_stack_w[depth - 1];
            if w.is_null() {
                residual_dispatch(vm, here)
            } else if unsafe { w_kind(w) } == CelKind::List {
                here + 1
            } else {
                let keys = interned_map_keys(vm, w as i64);
                if keys == 0 {
                    residual_dispatch(vm, here)
                } else {
                    let r = keys as usize as CelRef;
                    frame.locals_stack_w[depth - 1] = r;
                    here + 1
                }
            }
        }
        OP_ITER_KEYS => {
            let depth = frame.valuestackdepth;
            let w = frame.locals_stack_w[depth - 1];
            let keys = if w.is_null() {
                0
            } else if unsafe { w_kind(w) } == CelKind::List {
                interned_list_indices(vm, w as i64)
            } else {
                interned_map_keys(vm, w as i64)
            };
            if keys == 0 {
                residual_dispatch(vm, here)
            } else {
                let r = keys as usize as CelRef;
                frame.locals_stack_w[depth - 1] = r;
                here + 1
            }
        }
        OP_ITER_LEN => {
            let src = frame.locals_stack_w[insn_a(program, pc)];
            if src.is_null() || unsafe { w_kind(src) } != CelKind::List {
                residual_dispatch(vm, here)
            } else {
                let w = new_int_in(vm_heap(vm), unsafe { list_len(src) }) as CelRef;
                let depth = frame.valuestackdepth;
                frame.locals_stack_w[depth] = w;
                frame.valuestackdepth = depth + 1;
                here + 1
            }
        }
        OP_ITER_GUARD => {
            let idx_w = frame.locals_stack_w[insn_a(program, pc)];
            let src = frame.locals_stack_w[insn_b(program, pc)];
            if idx_w.is_null()
                || src.is_null()
                || unsafe { w_kind(idx_w) } != CelKind::Int
                || unsafe { w_kind(src) } != CelKind::List
            {
                residual_dispatch(vm, here)
            } else {
                let index = unsafe { (*idx_w.cast::<W_IntObject>()).intval };
                let len = unsafe { list_len(src) };
                if index >= len {
                    insn_c(program, pc)
                } else {
                    here + 1
                }
            }
        }
        OP_ITER_AT => {
            let src = frame.locals_stack_w[insn_a(program, pc)];
            let idx_w = frame.locals_stack_w[insn_b(program, pc)];
            if src.is_null()
                || idx_w.is_null()
                || unsafe { w_kind(src) } != CelKind::List
                || unsafe { w_kind(idx_w) } != CelKind::Int
            {
                residual_dispatch(vm, here)
            } else {
                let index = unsafe { (*idx_w.cast::<W_IntObject>()).intval };
                let item = interned_item(vm, src as i64, index);
                if item == 0 {
                    residual_dispatch(vm, here)
                } else {
                    let w = item as usize as CelRef;
                    let depth = frame.valuestackdepth;
                    frame.locals_stack_w[depth] = w;
                    frame.valuestackdepth = depth + 1;
                    here + 1
                }
            }
        }
        OP_ITER_BIND => {
            let src = frame.locals_stack_w[insn_a(program, pc)];
            let idx_w = frame.locals_stack_w[insn_b(program, pc)];
            if src.is_null()
                || idx_w.is_null()
                || unsafe { w_kind(src) } != CelKind::List
                || unsafe { w_kind(idx_w) } != CelKind::Int
            {
                residual_dispatch(vm, here)
            } else {
                let index = unsafe { (*idx_w.cast::<W_IntObject>()).intval };
                let item = interned_item(vm, src as i64, index);
                if item == 0 {
                    residual_dispatch(vm, here)
                } else {
                    let slot = insn_c(program, pc);
                    let w = item as usize as CelRef;
                    frame.locals_stack_w[slot] = w;
                    here + 1
                }
            }
        }
        OP_ITER_ADVANCE => {
            let slot = insn_a(program, pc);
            let w = frame.locals_stack_w[slot];
            if w.is_null() || unsafe { w_kind(w) } != CelKind::Int {
                residual_dispatch(vm, here)
            } else {
                let n = unsafe { (*w.cast::<W_IntObject>()).intval };
                match n.checked_add(1) {
                    None => residual_dispatch(vm, here),
                    Some(next) => {
                        let r = new_int_in(vm_heap(vm), next) as CelRef;
                        frame.locals_stack_w[slot] = r;
                        insn_b(program, pc)
                    }
                }
            }
        }
        OP_RETURN => {
            let depth = frame.valuestackdepth;
            let w = frame.locals_stack_w[depth - 1];
            if w.is_null() {
                residual_dispatch(vm, here)
            } else {
                vm_park_return(vm, w);
                PORTAL_DONE
            }
        }
        _ => residual_dispatch(vm, here),
    }
}

/// One residual for every traced arm that declines.
///
/// Each direct `residual_dispatch` site in the dispatch JitCode is its own
/// word-sized constant. Sharing one callee keeps that pool inside the
/// 256-wide int index space (`assembler.py` `emit_reg`).
///
/// `VirtualizableAnalyzer` marks a call that can read the virtualizable
/// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`; `handle_residual_call` records
/// that as `may_force` so the tracer syncs the vable before the call.
#[majit_macros::jit_inline(calls = { residual_dispatch => may_force_int })]
fn slow_pc(vm: i64, here: i64) -> i64 {
    residual_dispatch(vm, here)
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
    // `interp_jit.py` `_virtualizable_` on the frame, `virtualizables=['frame']`.
    // The array is one pointer to a block whose length word is at offset 0
    // and whose items begin at `CEL_ITEMS_BLOCK_ITEMS_OFFSET`
    // (`jtransform.py` `getarrayitem_vable_*`, direct `Ptr` array).
    virtualizable_fields = {
        var: frame,
        token_offset: crate::runtime::object::CELFRAME_VABLE_TOKEN_OFFSET,
        fields: {
            last_instr: int @ crate::runtime::object::CELFRAME_LAST_INSTR_OFFSET,
            valuestackdepth: int @ crate::runtime::object::CELFRAME_VALUESTACKDEPTH_OFFSET,
        },
        arrays: {
            locals_stack_w: ref @ (crate::runtime::object::CELFRAME_LOCALS_STACK_OFFSET) {
                length_offset: crate::runtime::object_array::CEL_ITEMS_BLOCK_LEN_OFFSET,
                items_offset: crate::runtime::object_array::CEL_ITEMS_BLOCK_ITEMS_OFFSET,
            },
        },
    },
    // Cells are pointer elements after the block's capacity word.
    array_fields = {
        W_CelFrame::locals_stack_w => CelRef in crate::runtime::object_array::CelItemsBlock,
        crate::runtime::object::W_IntColumn::data => i64,
    },
    int_fields = {
        crate::runtime::object::W_ListObject::strategy => u8,
        crate::runtime::object::W_ListObject::length => i64,
        crate::runtime::object::W_ListObject::start => i64,
        crate::runtime::object::W_IntColumn::length => i64,
    },
    ref_fields = {
        crate::runtime::object::W_ListObject::storage => crate::runtime::object::CelObject,
    },
    // `append_cell` / `list_resize_ge_i` are `dont_look_inside`, so their
    // calldescr would otherwise be `can_raise_effect_info` (empty write
    // set). `list_switch_to_object_append` reads the int column the traced
    // store just filled, and `_ll_list_resize_really` replaces `data`.
    residual_writes = {
        col.data[] @ crate::runtime::object::W_IntColumn => [append_cell, list_resize_ge_i],
        col.data @ crate::runtime::object::W_IntColumn => [list_resize_ge_i],
        col.length @ crate::runtime::object::W_IntColumn => [list_resize_ge_i],
        list.strategy @ crate::runtime::object::W_ListObject => [append_cell],
        list.storage @ crate::runtime::object::W_ListObject => [append_cell],
        list.length @ crate::runtime::object::W_ListObject => [append_cell],
        list.start @ crate::runtime::object::W_ListObject => [append_cell],
    },
    auto_calls = true,
    calls = {
        insn_op => elidable_int_cannot_raise,
        insn_a => elidable_int_cannot_raise,
        insn_b => elidable_int_cannot_raise,
        insn_c => elidable_int_cannot_raise,
        intern_const => elidable_ref_cannot_raise_wrapped,
        intern_var => residual_ref,
        intern_var_pure => elidable_ref_cannot_raise_wrapped,
        context_lookup_pure => elidable_int_cannot_raise,
        cell_kind => inline_int,
        cell_int => inline_int,
        cell_bool => inline_int,
        cell_list_len => inline_int,
        box_int => inline_ref,
        box_bool => inline_ref,
        trace_cmp_bit => inline_int,
        trace_arith_ok => inline_int,
        trace_arith_word => inline_int,
        slow_pc => inline_int,
        append_cell => residual_int,
        append_int_word => inline_int,
        append_ref => inline_int,
        add_local_const_cell => inline_ref,
        map_insert_cell => residual_int,
        contains_int_word => inline_int,
        int_identity => inline_ref,
        interned_name_is_int => elidable_int_cannot_raise,
        interned_is_double => elidable_int_cannot_raise,
        interned_is_string => elidable_int_cannot_raise,
        box_double => inline_ref,
        double_from_cell => inline_ref,
        int_to_text => residual_ref,
        string_from_cell => inline_ref,
        double_binop_cell => residual_ref,
        string_add_cell => residual_ref,
        map_store_pair => inline_int,
        cell_map_len => inline_int,
        trace_len_cell => inline_int,
        alloc_list => nursery_alloc_ref,
        alloc_map => nursery_alloc_ref,
        index_cell => inline_ref,
        item_cell => inline_ref,
        map_keys_cell => residual_ref,
        opt_is_none_i => inline_int,
        bool_short_i => inline_int,
        and_merge_keep => inline_int,
        interned_field => residual_int,
        field_name_cell => elidable_ref_cannot_raise_wrapped,
        str_cells_eq => elidable_int_cannot_raise,
        map_object_field => inline_ref,
        map_object_known => inline_int,
        interned_has_field => residual_int,
        interned_index => residual_int,
        interned_contains => residual_int,
        interned_len => residual_int,
        interned_len_cell => residual_int,
        interned_is_size => elidable_int_cannot_raise,
        interned_qualified_kind => elidable_int_cannot_raise,
        interned_unary => residual_int,
        interned_unary_cell => residual_ref,
        interned_temporal => residual_int,
        interned_optional_unary => residual_int,
        interned_method1 => residual_int,
        interned_optional_state => residual_int,
        interned_optional_inner => residual_int,
        interned_as_bool => residual_int,
        portal_rare => may_force_int,
        interned_map_keys => residual_int,
        interned_list_indices => residual_int,
        interned_opt_index => residual_int,
        interned_opt_select => residual_int,
        interned_item => residual_int,
        interned_equals => residual_int,
        interned_not_equals => residual_int,
        interned_add => inline_ref,
        interned_sub => inline_ref,
        interned_mul => inline_ref,
        interned_div => inline_ref,
        interned_rem => inline_ref,
        vm_heap => inline_ref,
        new_int_in => inline_ref,
        try_append => residual_int,
        new_list_with_capacity_in => inline_ref,
        residual_dispatch => may_force_int,
        residual_hydrate => may_force_int,
        vm_sync_binop => residual_int,
        vm_sync_replace => residual_int,
        vm_sync_push => residual_int,
        vm_sync_store => residual_int,
        vm_sync_write_local => residual_int,
        vm_sync_pop => residual_int,
        vm_park_return => residual_void,
        step_hot => may_force_int,
        new_int => inline_ref,
        new_bool => inline_ref,
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
        cel_negate => inline_ref,
        cel_optional_none => inline_ref,
        cel_optional_of => inline_ref,
        cel_optional_of_non_zero_value => inline_ref,
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
        // The dispatch JitCode is this match. `pc` is green, and only a
        // write inside the arm reaches the merge-point register. `return`
        // lowers only as the arm's last statement, so the exit stays after
        // the forward `continue` rather than inside the `if`.
        #[allow(clippy::collapsible_if)]
        match opcode {
            OP_LOAD_VAR => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let name_idx = insn_a(program, pc);
                // `context_lookup_pure` is elidable (`EF_ELIDABLE_CANNOT_RAISE`).
                // The pure arm is `intern_var_pure`; a resolver stays residual.
                let w = if context_lookup_pure(vm) != 0 {
                    intern_var_pure(vm, program, name_idx)
                } else {
                    intern_var(vm, program, name_idx)
                };
                let next = if w.is_null() {
                    slow_pc(vm, here)
                } else {
                    let depth = state.frame.valuestackdepth;
                    state.frame.locals_stack_w[depth] = w;
                    state.frame.valuestackdepth = depth + 1;
                    here + 1
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_LOAD_CONST => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let w = intern_const(program, insn_a(program, pc));
                let next = if w.is_null() {
                    slow_pc(vm, here)
                } else {
                    let depth = state.frame.valuestackdepth;
                    state.frame.locals_stack_w[depth] = w;
                    state.frame.valuestackdepth = depth + 1;
                    here + 1
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_LOAD_LOCAL => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let w = state.frame.locals_stack_w[insn_a(program, pc)];
                let next = if w.is_null() {
                    slow_pc(vm, here)
                } else {
                    let depth = state.frame.valuestackdepth;
                    state.frame.locals_stack_w[depth] = w;
                    state.frame.valuestackdepth = depth + 1;
                    here + 1
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_STORE_LOCAL => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let w = state.frame.locals_stack_w[depth - 1];
                let next = if w.is_null() {
                    slow_pc(vm, here)
                } else {
                    let slot = insn_a(program, pc);
                    state.frame.locals_stack_w[slot] = w;
                    state.frame.valuestackdepth = depth - 1;
                    here + 1
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_RETURN => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let w = state.frame.locals_stack_w[depth - 1];
                let next = if w.is_null() {
                    slow_pc(vm, here)
                } else {
                    vm_park_return(vm, w);
                    PORTAL_DONE
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_ADD_K => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let k = intern_const(program, insn_a(program, pc));
                let depth = state.frame.valuestackdepth;
                let i = depth - 1;
                let next = if i >= state.frame.n_slots {
                    let a = state.frame.locals_stack_w[i];
                    if !a.is_null() {
                        if !k.is_null() {
                            if cell_kind(a) == CelKind::Str as i64 {
                                if cell_kind(k) == CelKind::Str as i64 {
                                    let r = string_add_cell(a, k);
                                    if r.is_null() {
                                        slow_pc(vm, here)
                                    } else {
                                        state.frame.locals_stack_w[i] = r;
                                        here + 1
                                    }
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else if cell_kind(a) == CelKind::Int as i64 {
                                if cell_kind(k) == CelKind::Int as i64 {
                                    let l = cell_int(a);
                                    let rv = cell_int(k);
                                    match l.checked_add(rv) {
                                        Some(v) => {
                                            let r = box_int(vm, v);
                                            state.frame.locals_stack_w[i] = r;
                                            here + 1
                                        }
                                        None => slow_pc(vm, here),
                                    }
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_MUL_K => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let k = intern_const(program, insn_a(program, pc));
                let depth = state.frame.valuestackdepth;
                let i = depth - 1;
                let next = if i >= state.frame.n_slots {
                    let a = state.frame.locals_stack_w[i];
                    if !a.is_null() {
                        if !k.is_null() {
                            if cell_kind(a) == CelKind::Int as i64 {
                                if cell_kind(k) == CelKind::Int as i64 {
                                    let l = cell_int(a);
                                    let rv = cell_int(k);
                                    match l.checked_mul(rv) {
                                        Some(v) => {
                                            let r = box_int(vm, v);
                                            state.frame.locals_stack_w[i] = r;
                                            here + 1
                                        }
                                        None => slow_pc(vm, here),
                                    }
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_ADD_LOCAL_K => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let a = state.frame.locals_stack_w[insn_a(program, pc)];
                let k = intern_const(program, insn_b(program, pc));
                let r = add_local_const_cell(vm, a, k);
                let next = if r.is_null() {
                    slow_pc(vm, here)
                } else {
                    let depth = state.frame.valuestackdepth;
                    state.frame.locals_stack_w[depth] = r;
                    state.frame.valuestackdepth = depth + 1;
                    here + 1
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_MOD_K => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let k = intern_const(program, insn_a(program, pc));
                let depth = state.frame.valuestackdepth;
                let i = depth - 1;
                let next = if i >= state.frame.n_slots {
                    let a = state.frame.locals_stack_w[i];
                    if !a.is_null() {
                        if !k.is_null() {
                            if cell_kind(a) == CelKind::Int as i64 {
                                if cell_kind(k) == CelKind::Int as i64 {
                                    let l = cell_int(a);
                                    let rv = cell_int(k);
                                    // `ll_int_py_mod_zer` raises on zero.
                                    // `ll_int_py_mod_ovf` raises on
                                    // `MIN % -1` (Rust `%` panics there).
                                    // Every other pair records `int.py_mod`
                                    // plus the truncation adjustment
                                    // (`support.py` `_ll_2_int_mod`).
                                    if rv != 0 && (l != i64::MIN || rv != -1) {
                                        let v = l % rv;
                                        let r = box_int(vm, v);
                                        state.frame.locals_stack_w[i] = r;
                                        here + 1
                                    } else {
                                        slow_pc(vm, here)
                                    }
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_EQ_LOCAL_K => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let a = state.frame.locals_stack_w[insn_a(program, pc)];
                let k = intern_const(program, insn_b(program, pc));
                let next = if !a.is_null() {
                    if !k.is_null() {
                        if cell_kind(a) == CelKind::Int as i64 {
                            if cell_kind(k) == CelKind::Int as i64 {
                                let l = cell_int(a);
                                let rv = cell_int(k);
                                let bit = if l == rv { 1 } else { 0 };
                                let r = box_bool(bit);
                                let depth = state.frame.valuestackdepth;
                                state.frame.locals_stack_w[depth] = r;
                                state.frame.valuestackdepth = depth + 1;
                                here + 1
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_GT_LOCAL_K => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let a = state.frame.locals_stack_w[insn_a(program, pc)];
                let k = intern_const(program, insn_b(program, pc));
                let next = if !a.is_null() {
                    if !k.is_null() {
                        if cell_kind(a) == CelKind::Double as i64 {
                            if cell_kind(k) == CelKind::Double as i64 {
                                let r = double_binop_cell(OP_GT, a, k);
                                if r.is_null() {
                                    slow_pc(vm, here)
                                } else {
                                    let depth = state.frame.valuestackdepth;
                                    state.frame.locals_stack_w[depth] = r;
                                    state.frame.valuestackdepth = depth + 1;
                                    here + 1
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else if cell_kind(a) == CelKind::Int as i64 {
                            if cell_kind(k) == CelKind::Int as i64 {
                                let l = cell_int(a);
                                let rv = cell_int(k);
                                let bit = if l > rv { 1 } else { 0 };
                                let r = box_bool(bit);
                                let depth = state.frame.valuestackdepth;
                                state.frame.locals_stack_w[depth] = r;
                                state.frame.valuestackdepth = depth + 1;
                                here + 1
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_MUL_LOCAL_K_APPEND => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let a = state.frame.locals_stack_w[insn_a(program, pc)];
                let k = intern_const(program, insn_b(program, pc));
                let depth = state.frame.valuestackdepth;
                let top = depth - 1;
                let next = if top >= state.frame.n_slots {
                    let list = state.frame.locals_stack_w[top];
                    if !a.is_null() {
                        if !k.is_null() {
                            if !list.is_null() {
                                if cell_kind(a) == CelKind::Int as i64 {
                                    if cell_kind(k) == CelKind::Int as i64 {
                                        let l = cell_int(a);
                                        let rv = cell_int(k);
                                        match l.checked_mul(rv) {
                                            Some(v) => {
                                                if append_int_word(list, v) != 0 {
                                                    here + 1
                                                } else {
                                                    slow_pc(vm, here)
                                                }
                                            }
                                            None => slow_pc(vm, here),
                                        }
                                    } else {
                                        slow_pc(vm, here)
                                    }
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_ADD_LOCAL_K_APPEND => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let a = state.frame.locals_stack_w[insn_a(program, pc)];
                let k = intern_const(program, insn_b(program, pc));
                let depth = state.frame.valuestackdepth;
                let top = depth - 1;
                let next = if top >= state.frame.n_slots {
                    let list = state.frame.locals_stack_w[top];
                    if !a.is_null() {
                        if !k.is_null() {
                            if !list.is_null() {
                                if cell_kind(a) == CelKind::Int as i64 {
                                    if cell_kind(k) == CelKind::Int as i64 {
                                        let l = cell_int(a);
                                        let rv = cell_int(k);
                                        match l.checked_add(rv) {
                                            Some(v) => {
                                                if append_int_word(list, v) != 0 {
                                                    here + 1
                                                } else {
                                                    slow_pc(vm, here)
                                                }
                                            }
                                            None => slow_pc(vm, here),
                                        }
                                    } else {
                                        slow_pc(vm, here)
                                    }
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_MOD_LOCAL_K_APPEND => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let a = state.frame.locals_stack_w[insn_a(program, pc)];
                let k = intern_const(program, insn_b(program, pc));
                let depth = state.frame.valuestackdepth;
                let top = depth - 1;
                let next = if top >= state.frame.n_slots {
                    let list = state.frame.locals_stack_w[top];
                    if !a.is_null() {
                        if !k.is_null() {
                            if !list.is_null() {
                                if cell_kind(a) == CelKind::Int as i64 {
                                    if cell_kind(k) == CelKind::Int as i64 {
                                        let l = cell_int(a);
                                        let rv = cell_int(k);
                                        if rv != 0 && (l != i64::MIN || rv != -1) {
                                            let v = l % rv;
                                            if append_int_word(list, v) != 0 {
                                                here + 1
                                            } else {
                                                slow_pc(vm, here)
                                            }
                                        } else {
                                            slow_pc(vm, here)
                                        }
                                    } else {
                                        slow_pc(vm, here)
                                    }
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_INDEX => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let key_i = depth - 1;
                let box_i = depth - 2;
                let next = if box_i >= state.frame.n_slots {
                    let key = state.frame.locals_stack_w[key_i];
                    let container = state.frame.locals_stack_w[box_i];
                    if !container.is_null() {
                        if !key.is_null() {
                            let item = index_cell(container, key);
                            if !item.is_null() {
                                state.frame.locals_stack_w[box_i] = item;
                                state.frame.valuestackdepth = depth - 1;
                                here + 1
                            } else {
                                // A miss has to run the interpreter with the
                                // frame forced; `slow_pc` leaves the cells
                                // unflushed and the error comes back internal.
                                step_hot(program, pc)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_JUMP_IF_FALSE => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let i = depth - 1;
                let next = if i >= state.frame.n_slots {
                    let w = state.frame.locals_stack_w[i];
                    let bit = cell_bool(w);
                    if bit < 0 {
                        slow_pc(vm, here)
                    } else {
                        state.frame.valuestackdepth = depth - 1;
                        if bit == 0 {
                            insn_a(program, pc)
                        } else {
                            here + 1
                        }
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_JUMP_IF_OPT_NONE => {
                state.frame.last_instr = pc as i64;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let w = state.frame.locals_stack_w[depth - 1];
                let next = if !w.is_null() {
                    if opt_is_none_i(w) != 0 {
                        insn_a(program, pc)
                    } else {
                        here + 1
                    }
                } else {
                    here + 1
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_NEW_MAP => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let w = alloc_map(vm, insn_a(program, pc));
                let depth = state.frame.valuestackdepth;
                state.frame.locals_stack_w[depth] = w;
                state.frame.valuestackdepth = depth + 1;
                let next = here + 1;
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_MAP_INSERT => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let value = state.frame.locals_stack_w[depth - 1];
                let key = state.frame.locals_stack_w[depth - 2];
                let map = state.frame.locals_stack_w[depth - 3];
                let next = if !value.is_null() {
                    if !key.is_null() {
                        if !map.is_null() {
                            if map_store_pair(map, key, value) != 0 {
                                state.frame.valuestackdepth = depth - 2;
                                here + 1
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_ITER_ELEMS => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let w = state.frame.locals_stack_w[depth - 1];
                let next = if w.is_null() {
                    slow_pc(vm, here)
                } else if cell_kind(w) == CelKind::List as i64 {
                    here + 1
                } else {
                    let keys = map_keys_cell(vm, w);
                    if keys.is_null() {
                        slow_pc(vm, here)
                    } else {
                        state.frame.locals_stack_w[depth - 1] = keys;
                        here + 1
                    }
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_NEW_LIST => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let w = alloc_list(vm, insn_a(program, pc));
                let depth = state.frame.valuestackdepth;
                state.frame.locals_stack_w[depth] = w;
                state.frame.valuestackdepth = depth + 1;
                let next = here + 1;
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_LIST_APPEND => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let item = state.frame.locals_stack_w[depth - 1];
                let list = state.frame.locals_stack_w[depth - 2];
                let next = if !item.is_null() {
                    if !list.is_null() {
                        let stored = if cell_kind(item) == CelKind::Int as i64 {
                            append_int_word(list, cell_int(item))
                        } else {
                            let via_obj = append_ref(list, item);
                            if via_obj != 0 {
                                via_obj
                            } else {
                                append_cell(list, item)
                            }
                        };
                        if stored != 0 {
                            state.frame.valuestackdepth = depth - 1;
                            here + 1
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_NEW_LIST_FROM_ARG => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let src = state.frame.locals_stack_w[insn_a(program, pc)];
                let next = if cell_kind(src) == CelKind::List as i64 {
                    let w = alloc_list(vm, cell_list_len(src));
                    let depth = state.frame.valuestackdepth;
                    state.frame.locals_stack_w[depth] = w;
                    state.frame.valuestackdepth = depth + 1;
                    here + 1
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_ITER_GUARD => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let idx_w = state.frame.locals_stack_w[insn_a(program, pc)];
                let src = state.frame.locals_stack_w[insn_b(program, pc)];
                let next = if cell_kind(idx_w) == CelKind::Int as i64 {
                    if cell_kind(src) == CelKind::List as i64 {
                        let index = cell_int(idx_w);
                        let len = cell_list_len(src);
                        if index >= len {
                            insn_c(program, pc)
                        } else {
                            here + 1
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_ITER_BIND => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let src = state.frame.locals_stack_w[insn_a(program, pc)];
                let idx_w = state.frame.locals_stack_w[insn_b(program, pc)];
                let next = if cell_kind(src) == CelKind::List as i64 {
                    if cell_kind(idx_w) == CelKind::Int as i64 {
                        let index = cell_int(idx_w);
                        let item = item_cell(vm, src, index);
                        if !item.is_null() {
                            let slot = insn_c(program, pc);
                            state.frame.locals_stack_w[slot] = item;
                            here + 1
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_ITER_ADVANCE => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let slot = insn_a(program, pc);
                let w = state.frame.locals_stack_w[slot];
                let next = if cell_kind(w) == CelKind::Int as i64 {
                    let n = cell_int(w);
                    match n.checked_add(1) {
                        Some(v) => {
                            let r = box_int(vm, v);
                            state.frame.locals_stack_w[slot] = r;
                            insn_b(program, pc)
                        }
                        None => slow_pc(vm, here),
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_ACCU_LOOP_COND | OP_ACCU_LOOP_COND_NOT => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let w = state.frame.locals_stack_w[insn_a(program, pc)];
                let bit = cell_bool(w);
                // `all` leaves the loop while the accumulator is false.
                // `exists` leaves it while the accumulator is true.
                let leave_on = if opcode == OP_ACCU_LOOP_COND { 0 } else { 1 };
                let next = if bit < 0 {
                    slow_pc(vm, here)
                } else if bit == leave_on {
                    insn_b(program, pc)
                } else {
                    here + 1
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_AND_LOCAL | OP_OR_LOCAL => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let slot = insn_a(program, pc);
                let w = state.frame.locals_stack_w[slot];
                let is_or = if opcode == OP_OR_LOCAL { 1 } else { 0 };
                let code = bool_short_i(w, is_or);
                let next = if code == 1 {
                    let r = box_bool(is_or);
                    let depth = state.frame.valuestackdepth;
                    state.frame.locals_stack_w[depth] = r;
                    state.frame.valuestackdepth = depth + 1;
                    insn_c(program, pc)
                } else if code == 2 {
                    here + 1
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_AND_MERGE | OP_OR_MERGE => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let i = depth - 1;
                let is_or = if opcode == OP_OR_MERGE { 1 } else { 0 };
                let next = if i >= state.frame.n_slots {
                    let w = state.frame.locals_stack_w[i];
                    let bits = state.frame.scratch_bits;
                    if and_merge_keep(vm, w, insn_a(program, pc), is_or, bits) != 0 {
                        here + 1
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_LOAD_LOCAL_APPEND => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let w = state.frame.locals_stack_w[insn_a(program, pc)];
                let depth = state.frame.valuestackdepth;
                let top = depth - 1;
                let next = if top >= state.frame.n_slots {
                    let list = state.frame.locals_stack_w[top];
                    if !w.is_null() {
                        if !list.is_null() {
                            let stored = if cell_kind(w) == CelKind::Int as i64 {
                                append_int_word(list, cell_int(w))
                            } else {
                                let via_obj = append_ref(list, w);
                                if via_obj != 0 {
                                    via_obj
                                } else {
                                    append_cell(list, w)
                                }
                            };
                            if stored != 0 {
                                here + 1
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_ADD | OP_SUB | OP_MUL | OP_DIV | OP_MOD | OP_EQ | OP_NE | OP_LT | OP_LE | OP_GT
            | OP_GE => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let bi = depth - 1;
                let ai = depth - 2;
                let next = if ai >= state.frame.n_slots {
                    let a = state.frame.locals_stack_w[ai];
                    let b = state.frame.locals_stack_w[bi];
                    if !a.is_null() {
                        if !b.is_null() {
                            if cell_kind(a) == CelKind::Double as i64 {
                                if cell_kind(b) == CelKind::Double as i64 {
                                    let r = double_binop_cell(opcode, a, b);
                                    if r.is_null() {
                                        slow_pc(vm, here)
                                    } else {
                                        state.frame.locals_stack_w[ai] = r;
                                        state.frame.valuestackdepth = depth - 1;
                                        here + 1
                                    }
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else if cell_kind(a) == CelKind::Int as i64 {
                                if cell_kind(b) == CelKind::Int as i64 {
                                    let l = cell_int(a);
                                    let rv = cell_int(b);
                                    let bit = trace_cmp_bit(opcode, l, rv);
                                    if bit >= 0 {
                                        let r = box_bool(bit);
                                        state.frame.locals_stack_w[ai] = r;
                                        state.frame.valuestackdepth = depth - 1;
                                        here + 1
                                    } else if trace_arith_ok(opcode, l, rv) != 0 {
                                        let v = trace_arith_word(opcode, l, rv);
                                        let r = box_int(vm, v);
                                        state.frame.locals_stack_w[ai] = r;
                                        state.frame.valuestackdepth = depth - 1;
                                        here + 1
                                    } else {
                                        slow_pc(vm, here)
                                    }
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_EQ_K | OP_NE_K | OP_LT_K | OP_GT_K | OP_GE_K => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let k = intern_const(program, insn_a(program, pc));
                let depth = state.frame.valuestackdepth;
                let i = depth - 1;
                let cmp_op = if opcode == OP_EQ_K {
                    OP_EQ
                } else if opcode == OP_NE_K {
                    OP_NE
                } else if opcode == OP_LT_K {
                    OP_LT
                } else if opcode == OP_GT_K {
                    OP_GT
                } else {
                    OP_GE
                };
                let next = if i >= state.frame.n_slots {
                    let a = state.frame.locals_stack_w[i];
                    if !a.is_null() {
                        if !k.is_null() {
                            if cell_kind(a) == CelKind::Double as i64 {
                                if cell_kind(k) == CelKind::Double as i64 {
                                    let r = double_binop_cell(cmp_op, a, k);
                                    if r.is_null() {
                                        slow_pc(vm, here)
                                    } else {
                                        state.frame.locals_stack_w[i] = r;
                                        here + 1
                                    }
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else if cell_kind(a) == CelKind::Int as i64 {
                                if cell_kind(k) == CelKind::Int as i64 {
                                    let bit = trace_cmp_bit(cmp_op, cell_int(a), cell_int(k));
                                    if bit < 0 {
                                        slow_pc(vm, here)
                                    } else {
                                        let r = box_bool(bit);
                                        state.frame.locals_stack_w[i] = r;
                                        here + 1
                                    }
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_MOD_LOCAL_K | OP_LT_LOCAL_K | OP_NE_LOCAL_K | OP_GE_LOCAL_K => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let a = state.frame.locals_stack_w[insn_a(program, pc)];
                let k = intern_const(program, insn_b(program, pc));
                let cmp_op = if opcode == OP_MOD_LOCAL_K {
                    OP_MOD
                } else if opcode == OP_LT_LOCAL_K {
                    OP_LT
                } else if opcode == OP_NE_LOCAL_K {
                    OP_NE
                } else {
                    OP_GE
                };
                let next = if !a.is_null() {
                    if !k.is_null() {
                        if cell_kind(a) == CelKind::Int as i64 {
                            if cell_kind(k) == CelKind::Int as i64 {
                                let l = cell_int(a);
                                let rv = cell_int(k);
                                if opcode == OP_MOD_LOCAL_K {
                                    if rv != 0 && (l != i64::MIN || rv != -1) {
                                        let r = box_int(vm, l % rv);
                                        let depth = state.frame.valuestackdepth;
                                        state.frame.locals_stack_w[depth] = r;
                                        state.frame.valuestackdepth = depth + 1;
                                        here + 1
                                    } else {
                                        slow_pc(vm, here)
                                    }
                                } else {
                                    let bit = trace_cmp_bit(cmp_op, l, rv);
                                    if bit < 0 {
                                        slow_pc(vm, here)
                                    } else {
                                        let r = box_bool(bit);
                                        let depth = state.frame.valuestackdepth;
                                        state.frame.locals_stack_w[depth] = r;
                                        state.frame.valuestackdepth = depth + 1;
                                        here + 1
                                    }
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_EQ_LOCAL_K_APPEND | OP_GT_LOCAL_K_APPEND | OP_LT_LOCAL_K_APPEND
            | OP_NE_LOCAL_K_APPEND | OP_GE_LOCAL_K_APPEND => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let a = state.frame.locals_stack_w[insn_a(program, pc)];
                let k = intern_const(program, insn_b(program, pc));
                let depth = state.frame.valuestackdepth;
                let top = depth - 1;
                let cmp_op = if opcode == OP_EQ_LOCAL_K_APPEND {
                    OP_EQ
                } else if opcode == OP_GT_LOCAL_K_APPEND {
                    OP_GT
                } else if opcode == OP_LT_LOCAL_K_APPEND {
                    OP_LT
                } else if opcode == OP_NE_LOCAL_K_APPEND {
                    OP_NE
                } else {
                    OP_GE
                };
                let next = if top >= state.frame.n_slots {
                    let list = state.frame.locals_stack_w[top];
                    if !a.is_null() {
                        if !k.is_null() {
                            if !list.is_null() {
                                if cell_kind(a) == CelKind::Int as i64 {
                                    if cell_kind(k) == CelKind::Int as i64 {
                                        let bit = trace_cmp_bit(cmp_op, cell_int(a), cell_int(k));
                                        if bit < 0 {
                                            slow_pc(vm, here)
                                        } else {
                                            let r = box_bool(bit);
                                            let stored = append_ref(list, r);
                                            let stored = if stored != 0 {
                                                stored
                                            } else {
                                                append_cell(list, r)
                                            };
                                            if stored != 0 {
                                                here + 1
                                            } else {
                                                slow_pc(vm, here)
                                            }
                                        }
                                    } else {
                                        slow_pc(vm, here)
                                    }
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_NEGATE => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let i = depth - 1;
                let next = if i >= state.frame.n_slots {
                    let a = state.frame.locals_stack_w[i];
                    if cell_kind(a) == CelKind::Int as i64 {
                        let n = cell_int(a);
                        // `checked_neg` is not an ovf match. `i64::MIN` is
                        // the only overflow; `IntNeg` covers the rest.
                        if n != i64::MIN {
                            let r = box_int(vm, -n);
                            state.frame.locals_stack_w[i] = r;
                            here + 1
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_NOT => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let i = depth - 1;
                let next = if i >= state.frame.n_slots {
                    let a = state.frame.locals_stack_w[i];
                    let bit = cell_bool(a);
                    if bit < 0 {
                        slow_pc(vm, here)
                    } else {
                        let flipped = if bit == 0 { 1 } else { 0 };
                        let r = box_bool(flipped);
                        state.frame.locals_stack_w[i] = r;
                        here + 1
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_JUMP => {
                state.frame.last_instr = pc as i64;
                let next = insn_a(program, pc);
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_AND | OP_OR => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let i = depth - 1;
                let is_or = if opcode == OP_OR { 1 } else { 0 };
                let next = if i >= state.frame.n_slots {
                    let w = state.frame.locals_stack_w[i];
                    let code = bool_short_i(w, is_or);
                    if code == 1 {
                        let r = box_bool(is_or);
                        state.frame.locals_stack_w[i] = r;
                        insn_b(program, pc)
                    } else if code == 2 {
                        state.frame.valuestackdepth = depth - 1;
                        here + 1
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_CALL_HOST => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let arity = insn_b(program, pc);
                let name = insn_a(program, pc);
                let depth = state.frame.valuestackdepth;
                let i = depth - 1;
                let next = if arity == 1 {
                    if i >= state.frame.n_slots {
                        let w = state.frame.locals_stack_w[i];
                        if !w.is_null() {
                            if interned_is_size(program, name) != 0 {
                                let n = trace_len_cell(w);
                                if n < 0 {
                                    slow_pc(vm, here)
                                } else {
                                    let r = box_int(vm, n);
                                    state.frame.locals_stack_w[i] = r;
                                    here + 1
                                }
                            } else if interned_name_is_int(program, name) != 0 {
                                // `W_IntObject.int`: identity after the class check.
                                let r = int_identity(w);
                                if r.is_null() {
                                    slow_pc(vm, here)
                                } else {
                                    state.frame.locals_stack_w[i] = r;
                                    here + 1
                                }
                            } else if interned_is_double(program, name) != 0 {
                                let r = double_from_cell(vm, w);
                                if r.is_null() {
                                    slow_pc(vm, here)
                                } else {
                                    state.frame.locals_stack_w[i] = r;
                                    here + 1
                                }
                            } else if interned_is_string(program, name) != 0 {
                                let r = string_from_cell(w);
                                if r.is_null() {
                                    slow_pc(vm, here)
                                } else {
                                    state.frame.locals_stack_w[i] = r;
                                    here + 1
                                }
                            } else {
                                let r = interned_unary_cell(vm, program, name, w);
                                if !r.is_null() {
                                    state.frame.locals_stack_w[i] = r;
                                    here + 1
                                } else {
                                    slow_pc(vm, here)
                                }
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_CALL_METHOD => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let arity = insn_b(program, pc);
                let name = insn_a(program, pc);
                let depth = state.frame.valuestackdepth;
                let i = depth - 1;
                let next = if arity == 0 {
                    if i >= state.frame.n_slots {
                        let w = state.frame.locals_stack_w[i];
                        if !w.is_null() {
                            if interned_is_size(program, name) != 0 {
                                let n = trace_len_cell(w);
                                if n < 0 {
                                    slow_pc(vm, here)
                                } else {
                                    let r = box_int(vm, n);
                                    state.frame.locals_stack_w[i] = r;
                                    here + 1
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_IN => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let box_i = depth - 2;
                let next = if box_i >= state.frame.n_slots {
                    let container = state.frame.locals_stack_w[depth - 1];
                    let needle = state.frame.locals_stack_w[box_i];
                    if !container.is_null() {
                        if !needle.is_null() {
                            let found = contains_int_word(container, needle);
                            if found == 0 {
                                slow_pc(vm, here)
                            } else {
                                let bit = if found == 2 { 1 } else { 0 };
                                let r = box_bool(bit);
                                state.frame.locals_stack_w[box_i] = r;
                                state.frame.valuestackdepth = depth - 1;
                                here + 1
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_CALL_QUALIFIED => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let kind = interned_qualified_kind(program, insn_a(program, pc));
                let arity = insn_b(program, pc);
                let next = if kind == 0 {
                    if arity == 0 {
                        here + 1
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_HAS_FIELD => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let i = depth - 1;
                let name = field_name_cell(program, insn_a(program, pc));
                let next = if i >= state.frame.n_slots {
                    let recv = state.frame.locals_stack_w[i];
                    if !recv.is_null() {
                        if cell_kind(recv) == CelKind::Map as i64 {
                            if !name.is_null() {
                                let found = map_object_field(recv, name);
                                if !found.is_null() {
                                    state.frame.locals_stack_w[i] = box_bool(1);
                                    here + 1
                                } else if map_object_known(recv) != 0 {
                                    state.frame.locals_stack_w[i] = box_bool(0);
                                    here + 1
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            OP_GET_FIELD => {
                state.frame.last_instr = pc as i64;
                let vm = state.vm;
                let here = pc as i64;
                let depth = state.frame.valuestackdepth;
                let i = depth - 1;
                let name = field_name_cell(program, insn_a(program, pc));
                let next = if i >= state.frame.n_slots {
                    let recv = state.frame.locals_stack_w[i];
                    if !recv.is_null() {
                        if cell_kind(recv) == CelKind::Map as i64 {
                            if !name.is_null() {
                                let found = map_object_field(recv, name);
                                if !found.is_null() {
                                    state.frame.locals_stack_w[i] = found;
                                    here + 1
                                } else {
                                    slow_pc(vm, here)
                                }
                            } else {
                                slow_pc(vm, here)
                            }
                        } else {
                            slow_pc(vm, here)
                        }
                    } else {
                        slow_pc(vm, here)
                    }
                } else {
                    slow_pc(vm, here)
                };
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
            _ => {
                let next = step_hot(program, pc);
                if next >= 0 {
                    let tgt = next as usize;
                    if tgt < pc {
                        can_enter_jit!(driver, tgt, &mut *state, program, || {});
                    }
                    pc = tgt;
                    continue;
                }
                state.ret = next;
                return next;
            }
        }
    }
    // The merge point's compiled-run close `break`s out of this loop.
    // That path has no `return` of its own, so the value lives here.
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
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("2 > 1 ? 4 : 5").unwrap()).unwrap(),
                &ctx
            )
            .unwrap(),
            Value::Int(4)
        );
        let mut with_xs = Context::default();
        with_xs.add_variable_from_value("xs", vec![1i64, 2, 3]);
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("xs.map(x, x + 1)").unwrap()).unwrap(),
                &with_xs
            )
            .unwrap(),
            Value::list(vec![Value::Int(2), Value::Int(3), Value::Int(4)])
        );
        let record = Value::Map(crate::objects::Map::from(
            [("price", Value::Int(7)), ("qty", Value::Int(2))]
                .into_iter()
                .collect::<std::collections::HashMap<_, _>>(),
        ));
        let mut with_map = Context::default();
        with_map.add_variable_from_value("m", record.clone());
        with_map.add_variable_from_value("items", Value::list(vec![record.clone(), record]));
        with_map.add_variable_from_value("xs", vec![10i64, 20, 30]);
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("m.price").unwrap()).unwrap(),
                &with_map
            )
            .unwrap(),
            Value::Int(7)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("has(m.qty)").unwrap()).unwrap(),
                &with_map
            )
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("has(m.missing)").unwrap()).unwrap(),
                &with_map
            )
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("m['price']").unwrap()).unwrap(),
                &with_map
            )
            .unwrap(),
            Value::Int(7)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("xs[1]").unwrap()).unwrap(),
                &with_map
            )
            .unwrap(),
            Value::Int(20)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("items.map(i, i.price)").unwrap()).unwrap(),
                &with_map
            )
            .unwrap(),
            Value::list(vec![Value::Int(7), Value::Int(7)])
        );
        assert_eq!(
            cel_eval_loop(
                &compile(
                    &Parser::default()
                        .parse("items.filter(i, has(i.qty))")
                        .unwrap()
                )
                .unwrap(),
                &with_map
            )
            .unwrap(),
            Value::list(vec![
                Value::Map(crate::objects::Map::from(
                    [("price", Value::Int(7)), ("qty", Value::Int(2))]
                        .into_iter()
                        .collect::<std::collections::HashMap<_, _>>(),
                )),
                Value::Map(crate::objects::Map::from(
                    [("price", Value::Int(7)), ("qty", Value::Int(2))]
                        .into_iter()
                        .collect::<std::collections::HashMap<_, _>>(),
                )),
            ])
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("!(1 == 2)").unwrap()).unwrap(),
                &ctx
            )
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("-3 + 1").unwrap()).unwrap(),
                &ctx
            )
            .unwrap(),
            Value::Int(-2)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("20 in xs").unwrap()).unwrap(),
                &with_map
            )
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("99 in xs").unwrap()).unwrap(),
                &with_map
            )
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("xs.size()").unwrap()).unwrap(),
                &with_map
            )
            .unwrap(),
            Value::Int(3)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("size(xs)").unwrap()).unwrap(),
                &with_map
            )
            .unwrap(),
            Value::Int(3)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("xs.map(x, x)").unwrap()).unwrap(),
                &with_map
            )
            .unwrap(),
            Value::list(vec![Value::Int(10), Value::Int(20), Value::Int(30)])
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("int('3') + 1").unwrap()).unwrap(),
                &ctx
            )
            .unwrap(),
            Value::Int(4)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("optional.of(7).value()").unwrap()).unwrap(),
                &ctx
            )
            .unwrap(),
            Value::Int(7)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(
                    &Parser::default()
                        .parse("optional.none().hasValue()")
                        .unwrap()
                )
                .unwrap(),
                &ctx
            )
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("'hello'.startsWith('he')").unwrap()).unwrap(),
                &ctx
            )
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(
                    &Parser::default()
                        .enable_optional_syntax(true)
                        .parse("xs[?1].hasValue()")
                        .unwrap()
                )
                .unwrap(),
                &with_map
            )
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&Parser::default().parse("xs.all(x, x > 0)").unwrap()).unwrap(),
                &with_map
            )
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(
                    &Parser::default()
                        .parse("m.exists(k, k == 'price')")
                        .unwrap()
                )
                .unwrap(),
                &with_map
            )
            .unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn interned_opcode_numbers_match_the_enum() {
        assert_eq!(OP_LOAD_LOCAL, OpCode::LoadLocal as i64);
        assert_eq!(OP_ADD, OpCode::Add as i64);
        assert_eq!(OP_ADD_K, OpCode::AddConst as i64);
        assert_eq!(OP_JUMP, OpCode::Jump as i64);
        assert_eq!(OP_ITER_ADVANCE, OpCode::IterAdvance as i64);
        assert_eq!(OP_EQ, OpCode::Equals as i64);
        assert_eq!(OP_RETURN, OpCode::Return as i64);
        assert_eq!(OP_LOAD_VAR, OpCode::LoadVar as i64);
        assert_eq!(OP_GET_FIELD, OpCode::GetField as i64);
        assert_eq!(OP_HAS_FIELD, OpCode::HasField as i64);
        assert_eq!(OP_GET_FIELD_LOCAL, OpCode::GetFieldLocal as i64);
        assert_eq!(OP_INDEX, OpCode::Index as i64);
        assert_eq!(OP_NOT, OpCode::Not as i64);
        assert_eq!(OP_NEGATE, OpCode::Negate as i64);
        assert_eq!(OP_IN, OpCode::In as i64);
        assert_eq!(OP_LOAD_LOCAL_APPEND, OpCode::LoadLocalAppend as i64);
        assert_eq!(OP_CALL_HOST, OpCode::CallHost as i64);
        assert_eq!(OP_CALL_METHOD, OpCode::CallMethod as i64);
        assert_eq!(OP_CALL_QUALIFIED, OpCode::CallQualified as i64);
        assert_eq!(OP_OPT_INDEX, OpCode::OptIndex as i64);
        assert_eq!(OP_ITER_KEYS, OpCode::IterKeys as i64);
        assert_eq!(OP_ACCU_LOOP_COND, OpCode::AccuLoopCond as i64);
    }
}
