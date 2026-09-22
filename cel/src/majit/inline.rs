//! Small buffers the columnar encode path holds in the caller's frame.
//!
//! Task #58 priced `BatchProgram::bind` at ~100 ns FIXED per bind, invariant in
//! the row count, and named the mechanism: exactly six heap allocations, with no
//! single named stage of the encoding over 12% of the whole. The lever is
//! therefore the allocation COUNT and not any one stage.
//!
//! Five of those allocations are buffers that do not survive the bind, and every
//! one of them is sized by a quantity the LOWERING already fixed — the slot
//! count, the column count, the broadcast-scalar count, the output field count.
//! Those counts are small: every expression in this repo's own columnar examples
//! reads four paths or fewer. A `Vec` still pays a malloc and a free for each of
//! them on every bind, because a `Vec` cannot know that.
//!
//! Both types below hold `N` elements in the frame and fall back to a `Vec` past
//! that. Neither is a general-purpose container: they support the one shape the
//! encode path needs, which is fill once and then read.
//!
//! ⛔ A shape WIDER than `N` must cost what it costs today and no more. That is
//! what the spill arm is: it moves what is already held into one `Vec` sized for
//! it, and every later push goes there. No panic, no truncation.
//!
//! Two types rather than one because the element kinds genuinely differ.
//! [`Inline`] hands out a `&[T]`, which an array of `T` can only do if every
//! element is initialized — so it needs a fill value and `T: Copy`.
//! [`InlineOwned`] carries `SlotSource`, which owns `String`s and cannot be
//! either; it stores `Option<T>` and hands out an iterator, which is all its one
//! consumer wants.

/// How many elements the encode path's buffers hold before spilling.
///
/// Four, because that covers every expression the columnar examples carry:
/// `balance >= amount && !frozen` reads two paths, `role + "@" + region` four,
/// and a list comprehension three. A wider expression spills to exactly the
/// `Vec` that was there before this type existed.
pub(super) const INLINE_SLOTS: usize = 4;

/// A `Copy` buffer of up to `N` elements, readable as a slice.
pub(super) struct Inline<T: Copy, const N: usize> {
    /// Written left to right up to `len`. The slots at and past `len` still
    /// hold the fill value and are never read — an array has to be whole before
    /// any prefix of it can be borrowed, which is what a fill value buys over
    /// `MaybeUninit` and its unsafety.
    stack: [T; N],
    len: usize,
    /// Empty until the `N + 1`th push moves everything here. `Vec::new` does not
    /// allocate, so a buffer that never spills costs nothing at all, which is
    /// the whole point.
    heap: Vec<T>,
    /// The final length, if the caller knows it. See [`Inline::with_capacity`].
    hint: usize,
}

impl<T: Copy, const N: usize> Inline<T, N> {
    /// A buffer whose final length is `hint`.
    ///
    /// ⛔ The hint is what keeps a WIDE shape from paying more than it used to.
    /// Every one of these buffers is filled to a length the lowering already
    /// fixed, and the `Vec`s this type replaces were built with exactly that
    /// capacity — `Vec::with_capacity(slots.len())`, or a `collect` over an
    /// exact-size iterator, which is the same thing. A spill that reserved only
    /// for the arrival would re-grow on the next push and cost TWO allocations
    /// where one was paid before, turning a shape wider than `N` from unchanged
    /// into a regression. Measured, not supposed: at six slots that is what it
    /// did.
    pub(super) fn with_capacity(fill: T, hint: usize) -> Self {
        Inline {
            stack: [fill; N],
            len: 0,
            heap: Vec::new(),
            hint,
        }
    }

    pub(super) fn push(&mut self, value: T) {
        // `heap` non-empty is the spilled state: it only ever becomes non-empty
        // through the arm below, which puts at least one element in it, and
        // nothing here ever removes one.
        if !self.heap.is_empty() {
            self.heap.push(value);
        } else if self.len < N {
            self.stack[self.len] = value;
            self.len += 1;
        } else {
            // Exact, so the spill is the one allocation the `Vec` this replaces
            // made and not the start of a growth sequence. `max` covers a hint
            // that undercounts: the arrival still has to fit.
            self.heap.reserve_exact(self.hint.max(self.len + 1));
            self.heap.extend_from_slice(&self.stack[..self.len]);
            self.heap.push(value);
            self.len = 0;
        }
    }
}

impl<T: Copy, const N: usize> core::ops::Deref for Inline<T, N> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        if self.heap.is_empty() {
            &self.stack[..self.len]
        } else {
            &self.heap
        }
    }
}

/// [`Inline`] for an element type that owns heap data, and so is neither `Copy`
/// nor cheap to fill `N` copies of.
///
/// Yields an iterator rather than a slice: `Option<T>` is what makes the unused
/// slots expressible without `MaybeUninit`, and `[Option<T>]` is not `[T]`. The
/// one consumer, `BatchProgram::encode_reduce`, walks the buffer once and never
/// indexes it.
pub(super) struct InlineOwned<T, const N: usize> {
    stack: [Option<T>; N],
    len: usize,
    /// Spilled state, as in [`Inline`]: non-empty exactly when the stack half
    /// has been emptied into it.
    heap: Vec<T>,
    /// The final length, if the caller knows it — see [`Inline::with_capacity`]
    /// for why a spill must reserve for it and not for the arrival.
    hint: usize,
}

impl<T, const N: usize> InlineOwned<T, N> {
    pub(super) fn with_capacity(hint: usize) -> Self {
        InlineOwned {
            stack: [const { None }; N],
            len: 0,
            heap: Vec::new(),
            hint,
        }
    }

    pub(super) fn push(&mut self, value: T) {
        if self.heap.is_empty() {
            if self.len < N {
                self.stack[self.len] = Some(value);
                self.len += 1;
                return;
            }
            self.heap.reserve_exact(self.hint.max(self.len + 1));
            for slot in &mut self.stack[..self.len] {
                if let Some(held) = slot.take() {
                    self.heap.push(held);
                }
            }
            self.len = 0;
        }
        self.heap.push(value);
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &T> {
        // After a spill the stack prefix is empty and `len` is zero, so exactly
        // one of the two halves yields anything.
        self.stack[..self.len]
            .iter()
            .flatten()
            .chain(self.heap.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spill boundary, element for element. A buffer that truncated or
    /// reordered past `N` would be a wrong ANSWER rather than a slow one, and a
    /// count-only check would not see it.
    #[test]
    fn inline_spills_without_losing_or_reordering() {
        for count in 0..=(2 * INLINE_SLOTS + 1) {
            let mut buf: Inline<i64, INLINE_SLOTS> = Inline::with_capacity(-1, count);
            for k in 0..count {
                buf.push(k as i64);
            }
            let want: Vec<i64> = (0..count as i64).collect();
            assert_eq!(&*buf, &want[..], "count {count}");
        }
    }

    #[test]
    fn inline_owned_spills_without_losing_or_reordering() {
        for count in 0..=(2 * INLINE_SLOTS + 1) {
            let mut buf: InlineOwned<String, INLINE_SLOTS> = InlineOwned::with_capacity(count);
            for k in 0..count {
                buf.push(k.to_string());
            }
            let got: Vec<&str> = buf.iter().map(String::as_str).collect();
            let want: Vec<String> = (0..count).map(|k| k.to_string()).collect();
            assert_eq!(got, want.iter().map(String::as_str).collect::<Vec<_>>());
        }
    }

    /// A buffer that stays within `N` must not allocate — the property this
    /// whole file exists for, asserted rather than assumed.
    #[test]
    fn an_unspilled_buffer_holds_no_heap_capacity() {
        let mut buf: Inline<i64, INLINE_SLOTS> = Inline::with_capacity(0, INLINE_SLOTS);
        for k in 0..INLINE_SLOTS {
            buf.push(k as i64);
        }
        assert_eq!(buf.heap.capacity(), 0);

        let mut owned: InlineOwned<String, INLINE_SLOTS> = InlineOwned::with_capacity(INLINE_SLOTS);
        for k in 0..INLINE_SLOTS {
            owned.push(k.to_string());
        }
        assert_eq!(owned.heap.capacity(), 0);
    }

    /// A shape wider than `N` must cost what a plain `Vec` of the same known
    /// length costs — ONE allocation, exactly sized — and not the two a
    /// reserve-for-the-arrival spill produced. The capacity is the observable
    /// that separates them.
    #[test]
    fn a_spill_reserves_the_hint_and_does_not_regrow() {
        for count in (INLINE_SLOTS + 1)..=(3 * INLINE_SLOTS) {
            let mut buf: Inline<i64, INLINE_SLOTS> = Inline::with_capacity(0, count);
            for k in 0..count {
                buf.push(k as i64);
            }
            assert_eq!(buf.heap.capacity(), count, "count {count}");

            let mut owned: InlineOwned<String, INLINE_SLOTS> = InlineOwned::with_capacity(count);
            for k in 0..count {
                owned.push(k.to_string());
            }
            assert_eq!(owned.heap.capacity(), count, "count {count}");
        }
    }
}
