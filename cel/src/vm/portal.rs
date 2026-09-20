//! The JIT portal for [`super::interp::cel_eval_loop`].
//!
//! `interp_jit.py` `PyPyJitDriver`: greens are `(next_instr, code)`, the
//! red virtualizable is `frame`. `jit_merge_point` is the first statement
//! of the loop; `can_enter_jit` is only on a backward jump.
//!
//! Interned arithmetic, comparison, local load/store, context load,
//! field/index and return run on `frame.locals_stack_w[i]` — the
//! `getarrayitem_vable_*` shape — and call `cel_add` / `cel_equals`
//! with no `Result`. Everything else is residual [`Vm::dispatch_one`].

use majit_metainterp::JitDriver;

use super::code::CelCode;
use super::error::{CelErr, NameId};
use super::interp::{Step, Vm};
use super::opcode::OpCode;
use crate::runtime::binop::{
    cel_add, cel_div, cel_equals, cel_greater, cel_greater_equals, cel_less, cel_less_equals,
    cel_mul, cel_negate, cel_not_equals, cel_rem, cel_sub, map_lookup,
};
use crate::runtime::convert::{intern_leaf, interned_list_get, interned_map_lookup_string};
use crate::runtime::error::ERROR_SENTINEL;
use crate::runtime::heap::CelHeap;
use crate::runtime::object::{
    bytes_len, interned_list_eq, list_int_at, list_ints_slice, list_len, list_try_append, map_len,
    map_try_insert, new_bool, new_int, new_int_in, new_list_with_capacity, new_map_with_capacity_in,
    string_as_str, string_byte_len, w_kind, w_type, CelKind, CelRef,
    W_BoolObject, W_IntObject, W_OptionalObject, CEL_INT_CLASS,
};
#[allow(unused_imports)] // named in `virtualizable_fields`
use crate::runtime::object::{
    W_CelFrame, CELFRAME_LAST_INSTR_OFFSET, CELFRAME_LOCALS_STACK_OFFSET,
    CELFRAME_VABLE_TOKEN_OFFSET, CELFRAME_VALUESTACKDEPTH_OFFSET,
};
#[allow(unused_imports)]
use crate::runtime::object_array::{CEL_ITEMS_BLOCK_ITEMS_OFFSET, CEL_ITEMS_BLOCK_LEN_OFFSET};
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
            Some(a) if !a.is_null() && k != 0 => {
                let r = unsafe { $op(a, k as usize as CelRef) };
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
            Some(a) if !a.is_null() && k != 0 => {
                let r = unsafe { $op($vm, a, k as usize as CelRef) };
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
        if a.is_null() || k == 0 {
            residual_dispatch($vm, $here)
        } else {
            let r = unsafe { $op(a, k as usize as CelRef) };
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
        if a.is_null() || k == 0 {
            residual_dispatch($vm, $here)
        } else {
            let r = unsafe { $op($vm, a, k as usize as CelRef) };
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
            Some(list) if !a.is_null() && k != 0 && !list.is_null() => {
                let r = unsafe { $op(a, k as usize as CelRef) };
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
            Some(list) if !a.is_null() && k != 0 && !list.is_null() => {
                let r = unsafe { $op($vm, a, k as usize as CelRef) };
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
    match unsafe { w_kind(key) } {
        CelKind::Int | CelKind::UInt | CelKind::Bool | CelKind::Str => {}
        _ => return 0,
    }
    unsafe { i64::from(map_try_insert(map, key, value)) }
}

fn intern_const(program: &CelCode, idx: i64) -> i64 {
    let Some(value) = program.konst(idx as u32) else {
        return 0;
    };
    intern_leaf(value).map(|w| w as usize as i64).unwrap_or(0)
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
        CelKind::Map => match unsafe { string_as_str(k) } {
            Some(field) => unsafe { interned_map_lookup_string(w, field) },
            None => unsafe { map_lookup(w, k) },
        },
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
    if unsafe { w_kind(w) } != CelKind::Optional {
        return 0;
    }
    if unsafe { (*w.cast::<W_OptionalObject>()).w_value }.is_null() {
        2
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
        CelKind::Map => match unsafe { string_as_str(k) } {
            Some(field) => unsafe { interned_map_lookup_string(w, field) },
            None => unsafe { map_lookup(w, k) },
        },
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
            match interned_optional_state(w as i64) {
                2 => insn_a(program, pc),
                1 => here + 1,
                _ => residual_dispatch(vm, here),
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
        OP_AND_LOCAL | OP_OR_LOCAL => {
            let slot = insn_a(program, pc);
            let w = frame.locals_stack_w[slot];
            if w.is_null() || unsafe { w_kind(w) } != CelKind::Bool {
                residual_dispatch(vm, here)
            } else {
                let truthy = unsafe { (*w.cast::<W_BoolObject>()).boolval } != 0;
                let short = opcode == OP_OR_LOCAL;
                if truthy == short {
                    let r = new_bool(short) as CelRef;
                    let depth = frame.valuestackdepth;
                    frame.locals_stack_w[depth] = r;
                    frame.valuestackdepth = depth + 1;
                    insn_c(program, pc)
                } else {
                    here + 1
                }
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

/// Intern the context variable named `names[idx]`. 0 means miss or residual.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn intern_var(vm_bits: i64, program: &CelCode, idx: i64) -> i64 {
    let Some(name) = program.name(NameId(idx as u32)) else {
        return 0;
    };
    vm_of(vm_bits)
        .intern_context_var(name)
        .map(|w| w as usize as i64)
        .unwrap_or(0)
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
fn vm_park_return(vm_bits: i64, w: i64) {
    vm_of(vm_bits).park_return(w as usize as CelRef);
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
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn residual_dispatch(vm_bits: i64, pc: i64) -> i64 {
    let vm = unsafe { &mut *(vm_bits as usize as *mut Vm<'_>) };
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

#[inline(always)]
fn run_owned_driver(
    jit: &super::code::CodeJit,
    code: &CelCode,
    state: &mut PortalState,
) -> i64 {
    if jit.in_use.get() {
        let mut driver = fresh_portal_driver(state, code);
        return run_cel_portal(&mut driver, code, state, 0);
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
    run_cel_portal(driver, code, state, 0)
}

fn run_table_driver(code: &CelCode, state: &mut PortalState) -> i64 {
    let id = code.identity.id;
    let live = &code.identity.live;
    PORTAL_DRIVER.with(|slot| match slot.try_borrow_mut() {
        Ok(mut table) => {
            let i = table.index_for(id, live, state, code);
            run_cel_portal(&mut table.entries[i].driver, code, state, 0)
        }
        Err(_) => {
            let mut driver = fresh_portal_driver(state, code);
            run_cel_portal(&mut driver, code, state, 0)
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
        insn_b => residual_int,
        insn_c => residual_int,
        intern_const => residual_int,
        intern_var => residual_int,
        interned_field => residual_int,
        interned_has_field => residual_int,
        interned_index => residual_int,
        interned_contains => residual_int,
        interned_len => residual_int,
        interned_is_size => residual_int,
        interned_unary => residual_int,
        interned_temporal => residual_int,
        interned_optional_unary => residual_int,
        interned_method1 => residual_int,
        interned_qualified_kind => residual_int,
        interned_optional_state => residual_int,
        interned_optional_inner => residual_int,
        interned_as_bool => residual_int,
        portal_rare => residual_int,
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
        new_list_with_capacity => inline_ref,
        residual_dispatch => residual_int,
        vm_sync_binop => residual_int,
        vm_sync_replace => residual_int,
        vm_sync_push => residual_int,
        vm_sync_store => residual_int,
        vm_sync_write_local => residual_int,
        vm_sync_pop => residual_int,
        vm_park_return => residual_int,
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
        let frame = unsafe { &mut *(state.frame as *mut W_CelFrame) };
        frame.last_instr = pc as i64;
        let opcode = insn_op(program, pc);
        let vm = state.vm;
        let here = pc as i64;
        let next = match opcode {
            OP_LOAD_VAR => {
                let w = intern_var(vm, program, insn_a(program, pc));
                if w == 0 {
                    residual_dispatch(vm, here)
                } else {
                    let r = w as usize as CelRef;
                    let depth = frame.valuestackdepth;
                    frame.locals_stack_w[depth] = r;
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
                if recv.is_null()
                    || found == 0
                    || list.is_null()
                    || try_append(list as i64, found) == 0
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
                if w == 0 {
                    residual_dispatch(vm, here)
                } else {
                    let depth = frame.valuestackdepth;
                    frame.locals_stack_w[depth] = w as usize as CelRef;
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
                let w = new_list_with_capacity(insn_a(program, pc)) as CelRef;
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
                    let w = new_list_with_capacity(unsafe { list_len(src) }) as CelRef;
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
