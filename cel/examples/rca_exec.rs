//! Sampling-profile target for the per-call evaluators: `Program::execute`
//! (the bytecode VM in a default build) and the tree walker
//! (`Value::resolve_value`, called directly), one fixed activation, one
//! evaluation per iteration, for long enough to be sampled.
//!
//! The per-call board prices `1 + 2 * 3 - 4 / 2` at 47.7 ns through the walker
//! and 113.8 ns through the VM with ZERO allocations on either, so the excess
//! is not the allocator and an amplification probe cannot name it. A sampling
//! profile can.
//!
//! Two modes. `sample` loops for the sampler and prints a wall-clock
//! ns/eval that is only a sanity figure. `time` prices the door the way the
//! per-call board does: batches of at least 20 ms of THIS THREAD's CPU time,
//! best batch reported, so a loaded box costs repeatability and not truth.
//!
//! RELEASE ONLY:
//!   cargo run --release -p cel --example rca_exec -- [vm|walker] [expr] [secs] [sample|time]
//!   sample <pid> 5 -file out.txt      # while it runs
//!   RCA_EXEC_DUMP=1                    # print the instruction stream first
//!   RCA_EXEC_LIST_N=10000              # size of the `list` variable (default 10)

use std::hint::black_box;
use std::time::{Duration, Instant};

fn cpu_now() -> Duration {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: the call writes through the pointer and does nothing else, and
    // the pointer is to a live local of exactly the type it expects.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime: {}", std::io::Error::last_os_error());
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

use cel::{Context, Program, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let door = args.get(1).map_or("vm", |s| s.as_str());
    let expr = args.get(2).map_or("1 + 2 * 3 - 4 / 2", |s| s.as_str());
    let secs: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let program = Program::compile(expr).expect("compiles");
    let mut ctx = Context::default();
    ctx.add_variable_from_value("x", 15i64);
    // `RCA_EXEC_LIST_N` sizes `list`; ten elements by default, so that a
    // per-eval figure on the default context is still mostly the fixed cost.
    let list_n: i64 = std::env::var("RCA_EXEC_LIST_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    ctx.add_variable_from_value("list", (1..=list_n).collect::<Vec<_>>());
    ctx.add_variable_from_value("apple", true);
    if std::env::var_os("RCA_EXEC_DUMP").is_some() {
        let code = cel::vm::compile(program.expression()).expect("lowers");
        println!(
            "n_slots={} max_stack={} n_logic={} consts={}",
            code.n_slots,
            code.max_stack,
            code.n_logic,
            code.consts.len()
        );
        for (i, insn) in code.insns.iter().enumerate() {
            println!("  [{i:3}] {:?} {:?}", insn.op, insn.ops);
        }
    }
    let answer = program.execute(&ctx).expect("evaluates");
    let walker = Value::resolve_value(program.expression(), &ctx).expect("evaluates");
    assert_eq!(answer, walker, "the two evaluators disagree");
    println!(
        "pid={} door={door} sampling {expr:?} for {secs}s (answer {answer:?})",
        std::process::id()
    );
    let one = |ctx: &Context| match door {
        "vm" => program.execute(black_box(ctx)).expect("evaluates"),
        "walker" => Value::resolve_value(program.expression(), black_box(ctx)).expect("evaluates"),
        other => panic!("door {other:?}: vm or walker"),
    };
    let mode = args.get(4).map_or("sample", |s| s.as_str());
    let start = Instant::now();
    if mode == "sample" {
        let mut n = 0u64;
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..1000 {
                black_box(one(&ctx));
            }
            n += 1000;
        }
        let ns = start.elapsed().as_nanos() as f64 / n as f64;
        println!("{door}: {expr}: evals={n} ns/eval={ns:.1} (wall, sanity only)");
        return;
    }
    // Calibrate the batch to >= 20 ms of thread CPU, then take batches until
    // the wall budget runs out. Best batch = least descheduling and cache
    // damage; the median is printed beside it as the spread.
    let mut k = 1000u64;
    loop {
        let t0 = cpu_now();
        for _ in 0..k {
            black_box(one(&ctx));
        }
        if cpu_now() - t0 >= Duration::from_millis(20) {
            break;
        }
        k *= 2;
    }
    let mut batches: Vec<f64> = Vec::new();
    while start.elapsed().as_secs_f64() < secs {
        let t0 = cpu_now();
        for _ in 0..k {
            black_box(one(&ctx));
        }
        batches.push((cpu_now() - t0).as_nanos() as f64 / k as f64);
    }
    batches.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{door}: {expr}: batches={} k={k} best={:.1} median={:.1} ns/eval (thread CPU)",
        batches.len(),
        batches[0],
        batches[batches.len() / 2]
    );
}
