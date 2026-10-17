//! Process-wide intern-table lock for tests.
//!
//! `majit_ir::const_ptr_table` (intern table + young list) is process-wide, and
//! MiniMark collection always drains `collect_young_const_ptrs`. A parallel
//! JIT test that interned a ConstPtr into its own `CelGc` heap can leave an
//! address that falls inside another test's MiniMark nursery; the drain then
//! panics (`GC BUG: invalid type_id ... extra_area=young_const_ptrs`).
//!
//! Upstream serializes intern vs collection with majit-gc's
//! [`CONST_PTR_TABLE_TEST_LOCK`] (`#[cfg(test)]` in `collector.rs`, so cel
//! cannot take it). One GC per process is the supported configuration; tests
//! here install a MiniMark and a per-thread `CelGc` in the same process.

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Cel-side equivalent of majit-gc's `CONST_PTR_TABLE_TEST_LOCK`.
///
/// MiniMark collection tests take it exclusively for their whole body. Tests
/// that can intern ConstPtrs take it shared, so they still run in parallel
/// with each other and only exclude the MiniMark collector.
static CONST_PTR_TABLE_TEST_LOCK: RwLock<()> = RwLock::new(());

pub(crate) struct SharedGuard {
    _lock: RwLockReadGuard<'static, ()>,
}

/// Holds [`CONST_PTR_TABLE_TEST_LOCK`] exclusively and drains the young intern
/// list on acquire and on drop.
///
/// Drain on acquire is required: JIT tests that already dropped their shared
/// guard leave young ConstPtrs in the process-wide table, and a MiniMark
/// minor would `drag_out_root` those addresses. Drain on drop matches
/// majit-gc's `const_ptr_table_test_guard` so this test's own young interns
/// cannot reach a later collection as young roots. The MiniMark collection
/// itself already drains with `drag_out_root`; the Drop drain is the no-op
/// visitor for anything interned after that collection.
pub(crate) struct ExclusiveGuard {
    _lock: RwLockWriteGuard<'static, ()>,
}

pub(crate) fn shared() -> SharedGuard {
    SharedGuard {
        _lock: CONST_PTR_TABLE_TEST_LOCK
            .read()
            .unwrap_or_else(|p| p.into_inner()),
    }
}

pub(crate) fn exclusive() -> ExclusiveGuard {
    let lock = CONST_PTR_TABLE_TEST_LOCK
        .write()
        .unwrap_or_else(|p| p.into_inner());
    drain_young_const_ptrs();
    ExclusiveGuard { _lock: lock }
}

impl Drop for ExclusiveGuard {
    fn drop(&mut self) {
        drain_young_const_ptrs();
    }
}

fn drain_young_const_ptrs() {
    majit_ir::const_ptr_table::collect_young_const_ptrs(&mut |_| {});
}
