//! Carrier for the two front-end B probes over cel's Charon LLBC in `tests/`.
//!
//! Cargo needs a library or binary target to build a package's tests, and both
//! probes are entirely test-side: they open `../build/llbc/cel.ullbc` and drive
//! `majit-translate`'s public API, with nothing for a library to hold.
