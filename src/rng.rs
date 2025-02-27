use rand::prelude::*;
use rand::Rng;
use std::rc::Rc;
use std::cell::RefCell;

/// The trait to generate random values.
/// ランダムな値を生成するためのトレイトです。
pub trait NextRandValue
where
    Self: Sized,
{
    /// `next_i64` generates an `i64` and an updated instance of Self.
    /// `next_i64`は`i64`と更新されたSelfを生成します。
    fn next_i64(&self) -> (i64, Self);

    /// `next_u64` generates a `u64` and an updated instance of Self.
    fn next_u64(&self) -> (u64, Self) {
        let (i, r) = self.next_i64();
        (if i < 0 { -(i + 1) as u64 } else { i as u64 }, r)
    }

    /// `next_i32` generates an `i32` and an updated instance of Self.
    fn next_i32(&self) -> (i32, Self);

    /// `next_u32` generates a `u32` and an updated instance of Self.
    fn next_u32(&self) -> (u32, Self) {
        let (i, r) = self.next_i32();
        (if i < 0 { -(i + 1) as u32 } else { i as u32 }, r)
    }

    /// `next_i16` generates an `i16` and an updated instance of Self.
    fn next_i16(&self) -> (i16, Self);

    /// `next_u16` generates a `u16` and an updated instance of Self.
    fn next_u16(&self) -> (u16, Self) {
        let (i, r) = self.next_i16();
        (if i < 0 { -(i + 1) as u16 } else { i as u16 }, r)
    }

    /// `next_i8` generates an `i8` and an updated instance of Self.
    fn next_i8(&self) -> (i8, Self);

    /// `next_u8` generates a `u8` and an updated instance of Self.
    fn next_u8(&self) -> (u8, Self) {
        let (i, r) = self.next_i8();
        (if i < 0 { -(i + 1) as u8 } else { i as u8 }, r)
    }

    /// `next_f64` generates an `f64` and an updated instance of Self.
    fn next_f64(&self) -> (f64, Self) {
        let (i, r) = self.next_i64();
        (i as f64 / (i64::MAX as f64 + 1.0), r)
    }

    /// `next_f32` generates an `f32` and an updated instance of Self.
    fn next_f32(&self) -> (f32, Self) {
        let (i, r) = self.next_i32();
        (i as f32 / (i32::MAX as f32 + 1.0), r)
    }

    /// `next_bool` generates a `bool` and an updated instance of Self.
    fn next_bool(&self) -> (bool, Self) {
        let (i, r) = self.next_i32();
        ((i % 2) != 0, r)
    }
}

/// `RandGen` is a trait to generate random values.
pub trait RandGen<T: NextRandValue>
where
    Self: Sized,
{
    /// `rnd_gen` generates a tuple of `Self` and `T`.
    fn rnd_gen(rng: T) -> (Self, T);
}

impl<T: NextRandValue> RandGen<T> for i64 {
    fn rnd_gen(rng: T) -> (Self, T) {
        rng.next_i64()
    }
}

impl<T: NextRandValue> RandGen<T> for u32 {
    fn rnd_gen(rng: T) -> (Self, T) {
        rng.next_u32()
    }
}

impl<T: NextRandValue> RandGen<T> for i32 {
    fn rnd_gen(rng: T) -> (Self, T) {
        rng.next_i32()
    }
}

impl<T: NextRandValue> RandGen<T> for i16 {
    fn rnd_gen(rng: T) -> (Self, T) {
        rng.next_i16()
    }
}

impl<T: NextRandValue> RandGen<T> for f32 {
    fn rnd_gen(rng: T) -> (Self, T) {
        rng.next_f32()
    }
}

impl<T: NextRandValue> RandGen<T> for bool {
    fn rnd_gen(rng: T) -> (Self, T) {
        rng.next_bool()
    }
}

/// `RNG` is a random number generator.
/// `RNG`は乱数生成器です。
#[derive(Clone, Debug, PartialEq)]
pub struct RNG {
    rng: Rc<RefCell<StdRng>>,
}

impl Default for RNG {
    fn default() -> Self {
        Self::new()
    }
}

impl NextRandValue for RNG {
    fn next_i64(&self) -> (i64, Self) {
        let n = { self.rng.borrow_mut().gen::<i64>() };
        (n, Self { rng: Rc::clone(&self.rng) })
    }

    fn next_i32(&self) -> (i32, Self) {
        let n = { self.rng.borrow_mut().gen::<i32>() };
        (n, Self { rng: Rc::clone(&self.rng) })
    }

    fn next_i16(&self) -> (i16, Self) {
        let n = { self.rng.borrow_mut().gen::<i16>() };
        (n, Self { rng: Rc::clone(&self.rng) })
    }

    fn next_i8(&self) -> (i8, Self) {
        let n = { self.rng.borrow_mut().gen::<i8>() };
        (n, Self { rng: Rc::clone(&self.rng) })
    }
}

impl RNG {
    /// `new` is a constructor.
    /// `new`はファクトリです。
    pub fn new() -> Self {
        Self {
            rng: Rc::new(RefCell::new(StdRng::seed_from_u64(0))),
        }
    }

    /// `new_with_seed` is a constructor with seed.
    /// `new_with_seed`はシード値を指定するファクトリです。
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rc::new(RefCell::new(StdRng::seed_from_u64(seed)));
        self
    }

    /// `i32_f32` generates a tuple of `i32` and `f32`.
    /// `i32_f32`は`i32`と`f32`のタプルを生成します。
    pub fn i32_f32(&self) -> ((i32, f32), Self) {
        let (i, r1) = self.next_i32();
        let (d, r2) = r1.next_f32();
        ((i, d), r2)
    }

    /// `f32_i32` generates a tuple of `f32` and `i32`.
    /// `f32_i32`は`f32`と`i32`のタプルを生成します。
    pub fn f32_i32(&self) -> ((f32, i32), Self) {
        let ((i, d), r) = self.i32_f32();
        ((d, i), r)
    }

    /// `f32_3` generates a tuple of `f32`, `f32` and `f32`.
    /// `f32_3`は`f32`と`f32`と`f32`のタプルを生成します。
    pub fn f32_3(&self) -> ((f32, f32, f32), Self) {
        let (d1, r1) = self.next_f32();
        let (d2, r2) = r1.next_f32();
        let (d3, r3) = r2.next_f32();
        ((d1, d2, d3), r3)
    }

    /// `i32s` generates a vector of `i32`.
    /// `i32s`は`i32`のベクタを生成します。
    pub fn i32s(self, count: u32) -> (Vec<i32>, Self) {
        let mut index = count;
        let mut acc = vec![];
        let mut current_rng = self;
        while index > 0 {
            let (x, new_rng) = current_rng.next_i32();
            acc.push(x);
            index -= 1;
            current_rng = new_rng;
        }
        (acc, current_rng)
    }

    /// `unit` generates a function that returns a tuple of `A` and `RNG`.
    /// `unit`は`A`と`RNG`のタプルを返す関数を生成します。
    pub fn unit<A>(a: A) -> Box<dyn FnMut(RNG) -> (A, RNG)>
    where
        A: Clone + 'static,
    {
        Box::new(move |rng: RNG| (a.clone(), rng))
    }

    /// `sequence` generates a function that returns a tuple of `Vec<A>` and `RNG`.
    /// `sequence`は`Vec<A>`と`RNG`のタプルを返す関数を生成します。
    pub fn sequence<A, F>(fs: Vec<F>) -> Box<dyn FnMut(RNG) -> (Vec<A>, RNG)>
    where
        A: Clone + 'static,
        F: FnMut(RNG) -> (A, RNG) + 'static,
    {
        let unit = Self::unit(Vec::<A>::new());
        let result = fs.into_iter().fold(unit, |acc, e| {
            Self::map2(acc, e, |mut a, b| {
                a.push(b);
                a
            })
        });
        result
    }

    /// `int_value` generates a function that returns a tuple of `i32` and `RNG`.
    /// `int_value`は`i32`と`RNG`のタプルを返す関数を生成します。
    pub fn int_value() -> Box<dyn FnMut(RNG) -> (i32, RNG)> {
        Box::new(move |rng| rng.next_i32())
    }

    /// `double_value` generates a function that returns a tuple of `f32` and `RNG`.
    /// `double_value`は`f32`と`RNG`のタプルを返す関数を生成します。
    pub fn double_value() -> Box<dyn FnMut(RNG) -> (f32, RNG)> {
        Box::new(move |rng| rng.next_f32())
    }

    /// `map` generates a function that returns a tuple of `B` and `RNG`.
    /// `map`は`B`と`RNG`のタプルを返す関数を生成します。
    pub fn map<A, B, F1, F2>(mut s: F1, mut f: F2) -> Box<dyn FnMut(RNG) -> (B, RNG)>
    where
        F1: FnMut(RNG) -> (A, RNG) + 'static,
        F2: FnMut(A) -> B + 'static,
    {
        Box::new(move |rng| {
            let (a, rng2) = s(rng);
            (f(a), rng2)
        })
    }

    /// `map2` generates a function that returns a tuple of `C` and `RNG`.
    /// `map2`は`C`と`RNG`のタプルを返す関数を生成します。
    pub fn map2<F1, F2, F3, A, B, C>(
        mut ra: F1,
        mut rb: F2,
        mut f: F3,
    ) -> Box<dyn FnMut(RNG) -> (C, RNG)>
    where
        F1: FnMut(RNG) -> (A, RNG) + 'static,
        F2: FnMut(RNG) -> (B, RNG) + 'static,
        F3: FnMut(A, B) -> C + 'static,
    {
        Box::new(move |rng| {
            let (a, r1) = ra(rng);
            let (b, r2) = rb(r1);
            (f(a, b), r2)
        })
    }

    /// `both` generates a function that returns a tuple of `(A, B)` and `RNG`.
    /// `both`は`(A, B)`と`RNG`のタプルを返す関数を生成します。
    pub fn both<F1, F2, A, B>(
        ra: F1,
        rb: F2,
    ) -> Box<dyn FnMut(RNG) -> ((A, B), RNG)>
    where
        F1: FnMut(RNG) -> (A, RNG) + 'static,
        F2: FnMut(RNG) -> (B, RNG) + 'static,
    {
        Self::map2(ra, rb, |a, b| (a, b))
    }

    /// `rand_int_double` generates a function that returns a tuple of `(i32, f32)` and `RNG`.
    /// `rand_int_double`は`(i32, f32)`と`RNG`のタプルを返す関数を生成します。
    pub fn rand_int_double() -> Box<dyn FnMut(RNG) -> ((i32, f32), RNG)> {
        Self::both(Self::int_value(), Self::double_value())
    }

    /// `rand_double_int` generates a function that returns a tuple of `(f32, i32)` and `RNG`.
    /// `rand_double_int`は`(f32, i32)`と`RNG`のタプルを返す関数を生成します。
    pub fn rand_double_int() -> Box<dyn FnMut(RNG) -> ((f32, i32), RNG)> {
        Self::both(Self::double_value(), Self::int_value())
    }

    /// `flat_map` generates a function that returns a tuple of `B` and `RNG`.
    /// `flat_map`は`B`と`RNG`のタプルを返す関数を生成します。
    pub fn flat_map<A, B, F, GF, BF>(mut f: F, mut g: GF) -> Box<dyn FnMut(RNG) -> (B, RNG)>
    where
        F: FnMut(RNG) -> (A, RNG) + 'static,
        BF: FnMut(RNG) -> (B, RNG) + 'static,
        GF: FnMut(A) -> BF + 'static,
    {
        Box::new(move |rng| {
            let (a, r1) = f(rng);
            (g(a))(r1)
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::rng::{RandGen, RNG};
    use std::env;

    fn init() {
        env::set_var("RUST_LOG", "info");
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn next_i32() {
        init();
        let rng = RNG::new();
        let (v1, r1) = i32::rnd_gen(rng);
        log::info!("{:?}", v1);
        let (v2, _) = u32::rnd_gen(r1);
        log::info!("{:?}", v2);
    }
}
