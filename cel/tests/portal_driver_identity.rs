//! JIT driver table is keyed by a minted code id and sweeps dead entries.

#![cfg(feature = "jit")]

use cel::{Context, Program, Value};

#[test]
fn driver_table_is_keyed_by_code_id_and_sweeps_dead_entries() {
    let ctx = Context::default();
    let mut max_len = 0usize;
    for i in 0..1000 {
        let a = Program::compile("1").expect("A");
        assert_eq!(a.execute(&ctx).expect("eval A"), Value::Int(1));
        drop(a);
        let src = format!("{i} + {}", i + 1);
        let b = Program::compile(&src).expect("B");
        let got = b.execute(&ctx).expect("eval B");
        assert_eq!(got, Value::Int(i as i64 + (i as i64 + 1)));
        let n = cel::vm::portal_driver_len();
        max_len = max_len.max(n);
        assert!(
            n <= 8,
            "driver table length {n} after iteration {i} is not bounded"
        );
        drop(b);
    }
    assert!(max_len <= 8, "driver table grew to {max_len}");
}
