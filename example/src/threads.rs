use cel::{Context, Program};
use std::thread::scope;

/// One compiled program shared across threads.
///
/// `Program` contains interned `*mut` leaves, so the type system does not
/// mark it `Sync`. The portal thread test uses the same opt-in: the const
/// pool is shared, and each thread builds its own `Context`.
struct SharedProgram(Program);

unsafe impl Send for SharedProgram {}
unsafe impl Sync for SharedProgram {}

impl SharedProgram {
    fn execute(&self, context: &Context) -> cel::Value {
        self.0.execute(context).unwrap()
    }
}

fn main() {
    let program = SharedProgram(Program::compile("a + b").unwrap());

    scope(|scope| {
        scope.spawn(|| {
            let mut context = Context::default();
            context.add_variable("a", 1).unwrap();
            context.add_variable("b", 2).unwrap();
            let value = program.execute(&context);
            assert_eq!(value, 3.into());
        });
        scope.spawn(|| {
            let mut context = Context::default();
            context.add_variable("a", 2).unwrap();
            context.add_variable("b", 4).unwrap();
            let value = program.execute(&context);
            assert_eq!(value, 6.into());
        });
    });
}
