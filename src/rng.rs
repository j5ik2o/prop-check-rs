use rand::prelude::*;

/// The trait to generate random values.<br/>
/// ランダムな値を生成するためのトレイトです。
pub trait NextRandValue
where
  Self: Sized, {
  /// `next_i64` generates a tuple of `i64` and `Self`.<br/>
  /// `next_i64`は`i64`と`Self`のタプルを生成します。
  fn next_i64(&self) -> (i64, Self);

  /// `next_u64` generates a tuple of `u64` and `Self`.<br/>
  /// `next_u64`は`u64`と`Self`のタプルを生成します。
  fn next_u64(&self) -> (u64, Self) {
    let (i, r) = self.next_i64();
    (if i < 0 { -(i + 1) as u64 } else { i as u64 }, r)
  }

  /// `next_i32` generates a tuple of `i32` and `Self`.<br/>
  /// `next_i32`は`i32`と`Self`のタプルを生成します。
  fn next_i32(&self) -> (i32, Self);

  /// `next_u32` generates a tuple of `u32` and `Self`.<br/>
  /// `next_u32`は`u32`と`Self`のタプルを生成します。
  fn next_u32(&self) -> (u32, Self) {
    let (i, r) = self.next_i32();
    (if i < 0 { -(i + 1) as u32 } else { i as u32 }, r)
  }

  /// `next_i16` generates a tuple of `i16` and `Self`.<br/>
  /// `next_i16`は`i16`と`Self`のタプルを生成します。
  fn next_i16(&self) -> (i16, Self);

  /// `next_u16` generates a tuple of `u16` and `Self`.<br/>
  /// `next_u16`は`u16`と`Self`のタプルを生成します。
  fn next_u16(&self) -> (u16, Self) {
    let (i, r) = self.next_i16();
    (if i < 0 { -(i + 1) as u16 } else { i as u16 }, r)
  }

  /// `next_i8` generates a tuple of `i8` and `Self`.<br/>
  /// `next_i8`は`i8`と`Self`のタプルを生成します。
  fn next_i8(&self) -> (i8, Self);

  /// `next_u8` generates a tuple of `u8` and `Self`.<br/>
  /// `next_u8`は`u8`と`Self`のタプルを生成します。
  fn next_u8(&self) -> (u8, Self) {
    let (i, r) = self.next_i8();
    (if i < 0 { -(i + 1) as u8 } else { i as u8 }, r)
  }

  /// `next_f64` generates a tuple of `f64` and `Self`.<br/>
  /// `next_f64`は`f64`と`Self`のタプルを生成します。
  fn next_f64(&self) -> (f64, Self) {
    let (i, r) = self.next_i64();
    (i as f64 / (i64::MAX as f64 + 1.0f64), r)
  }

  /// `next_f32` generates a tuple of `f32` and `Self`.<br/>
  /// `next_f32`は`f32`と`Self`のタプルを生成します。
  fn next_f32(&self) -> (f32, Self) {
    let (i, r) = self.next_i32();
    (i as f32 / (i32::MAX as f32 + 1.0f32), r)
  }

  /// `next_bool` generates a tuple of `bool` and `Self`.<br/>
  /// `next_bool`は`bool`と`Self`のタプルを生成します。
  fn next_bool(&self) -> (bool, Self) {
    let (i, r) = self.next_i32();
    ((i % 2) != 0, r)
  }
}

/// `RandGen` is a trait to generate random values.<br/>
/// `RandGen`はランダムな値を生成するためのトレイトです。
pub trait RandGen<T: NextRandValue>
where
  Self: Sized, {
  /// `rnd_gen` generates a tuple of `Self` and `T`.<br/>
  /// `rnd_gen`は`Self`と`T`のタプルを生成します。
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
  rng: StdRng,
}

type DynRand<A> = dyn FnMut(RNG) -> (A, RNG);
type BoxRand<A> = Box<DynRand<A>>;

impl Default for RNG {
  fn default() -> Self {
    Self::new()
  }
}

impl NextRandValue for RNG {
  fn next_i64(&self) -> (i64, Self) {
    let mut mr = self.rng.clone();
    let n = mr.gen();
    (n, Self { rng: mr })
  }

  fn next_u64(&self) -> (u64, Self) {
    let mut mr = self.rng.clone();
    let n = mr.gen();
    (n, Self { rng: mr })
  }

  fn next_i32(&self) -> (i32, Self) {
    let mut mr = self.rng.clone();
    let n = mr.gen();
    (n, Self { rng: mr })
  }

  fn next_u32(&self) -> (u32, Self) {
    let mut mr = self.rng.clone();
    let n = mr.gen();
    (n, Self { rng: mr })
  }

  fn next_i16(&self) -> (i16, Self) {
    let mut mr = self.rng.clone();
    let n = mr.gen();
    (n, Self { rng: mr })
  }

  fn next_u16(&self) -> (u16, Self) {
    let mut mr = self.rng.clone();
    let n = mr.gen();
    (n, Self { rng: mr })
  }

  fn next_i8(&self) -> (i8, Self) {
    let mut mr = self.rng.clone();
    let n = mr.gen();
    (n, Self { rng: mr })
  }

  fn next_u8(&self) -> (u8, Self) {
    let mut mr = self.rng.clone();
    let n = mr.gen();
    (n, Self { rng: mr })
  }

  fn next_f64(&self) -> (f64, Self) {
    let mut mr = self.rng.clone();
    let n = mr.gen();
    (n, Self { rng: mr })
  }

  fn next_f32(&self) -> (f32, Self) {
    let mut mr = self.rng.clone();
    let n = mr.gen();
    (n, Self { rng: mr })
  }

  fn next_bool(&self) -> (bool, Self) {
    let mut mr = self.rng.clone();
    let n = mr.gen();
    (n, Self { rng: mr })
  }
}

impl RNG {
  /// `new` is a constructor.<br/>
  /// `new`はファクトリです。
  pub fn new() -> Self {
    Self {
      rng: StdRng::from_rng(thread_rng()).unwrap(),
    }
  }

  /// `new_with_seed` is a constructor with seed.<br/>
  /// `new_with_seed`はシード値を指定するファクトリです。
  pub fn with_seed(mut self, seed: u64) -> Self {
    self.rng = StdRng::seed_from_u64(seed);
    self
  }

  /// `i32_f32` generates a tuple of `i32` and `f32`.<br/>
  /// `i32_f32`は`i32`と`f32`のタプルを生成します。
  pub fn i32_f32(&self) -> ((i32, f32), Self) {
    let (i, r1) = self.next_i32();
    let (d, r2) = r1.next_f32();
    ((i, d), r2)
  }

  /// `f32_i32` generates a tuple of `f32` and `i32`.<br/>
  /// `f32_i32`は`f32`と`i32`のタプルを生成します。
  pub fn f32_i32(&self) -> ((f32, i32), Self) {
    let ((i, d), r) = self.i32_f32();
    ((d, i), r)
  }

  /// `f32_3` generates a tuple of `f32`, `f32` and `f32`.<br/>
  /// `f32_3`は`f32`と`f32`と`f32`のタプルを生成します。
  pub fn f32_3(&self) -> ((f32, f32, f32), Self) {
    let (d1, r1) = self.next_f32();
    let (d2, r2) = r1.next_f32();
    let (d3, r3) = r2.next_f32();
    ((d1, d2, d3), r3)
  }

  /// `f32s` generates a vector of `f32`.<br/>
  /// `f32s`は`f32`のベクタを生成します。
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

  /// `unit` generates a function that returns a tuple of `A` and `RNG`.<br/>
  /// `unit`は`A`と`RNG`のタプルを返す関数を生成します。
  pub fn unit<A>(a: A) -> BoxRand<A>
  where
    A: Clone + 'static, {
    Box::new(move |rng: RNG| (a.clone(), rng))
  }

  /// `sequence` generates a function that returns a tuple of `Vec<A>` and `RNG`.<br/>
  /// `sequence`は`Vec<A>`と`RNG`のタプルを返す関数を生成します。
  pub fn sequence<A, F>(fs: Vec<F>) -> BoxRand<Vec<A>>
  where
    A: Clone + 'static,
    F: FnMut(RNG) -> (A, RNG) + 'static, {
    let unit = Self::unit(Vec::<A>::new());
    let result = fs.into_iter().fold(unit, |acc, e| {
      Self::map2(acc, e, |mut a, b| {
        a.push(b);
        a
      })
    });
    result
  }

  /// `int_value` generates a function that returns a tuple of `i32` and `RNG`.<br/>
  /// `int_value`は`i32`と`RNG`のタプルを返す関数を生成します。
  pub fn int_value() -> BoxRand<i32> {
    Box::new(move |rng| rng.next_i32())
  }

  /// `double_value` generates a function that returns a tuple of `f32` and `RNG`.<br/>
  /// `double_value`は`f32`と`RNG`のタプルを返す関数を生成します。
  pub fn double_value() -> BoxRand<f32> {
    Box::new(move |rng| rng.next_f32())
  }

  /// `map` generates a function that returns a tuple of `B` and `RNG`.<br/>
  /// `map`は`B`と`RNG`のタプルを返す関数を生成します。
  pub fn map<A, B, F1, F2>(mut s: F1, mut f: F2) -> BoxRand<B>
  where
    F1: FnMut(RNG) -> (A, RNG) + 'static,
    F2: FnMut(A) -> B + 'static, {
    Box::new(move |rng| {
      let (a, rng2) = s(rng);
      (f(a), rng2)
    })
  }

  /// `map2` generates a function that returns a tuple of `C` and `RNG`.<br/>
  /// `map2`は`C`と`RNG`のタプルを返す関数を生成します。
  pub fn map2<F1, F2, F3, A, B, C>(mut ra: F1, mut rb: F2, mut f: F3) -> BoxRand<C>
  where
    F1: FnMut(RNG) -> (A, RNG) + 'static,
    F2: FnMut(RNG) -> (B, RNG) + 'static,
    F3: FnMut(A, B) -> C + 'static, {
    Box::new(move |rng| {
      let (a, r1) = ra(rng);
      let (b, r2) = rb(r1);
      (f(a, b), r2)
    })
  }

  /// `both` generates a function that returns a tuple of `(A, B)` and `RNG`.<br/>
  /// `both`は`(A, B)`と`RNG`のタプルを返す関数を生成します。
  pub fn both<F1, F2, A, B>(ra: F1, rb: F2) -> BoxRand<(A, B)>
  where
    F1: FnMut(RNG) -> (A, RNG) + 'static,
    F2: FnMut(RNG) -> (B, RNG) + 'static, {
    Self::map2(ra, rb, |a, b| (a, b))
  }

  /// `rand_int_double` generates a function that returns a tuple of `(i32, f32)` and `RNG`.<br/>
  /// `rand_int_double`は`(i32, f32)`と`RNG`のタプルを返す関数を生成します。
  pub fn rand_int_double() -> BoxRand<(i32, f32)> {
    Self::both(Self::int_value(), Self::double_value())
  }

  /// `rand_double_int` generates a function that returns a tuple of `(f32, i32)` and `RNG`.<br/>
  /// `rand_double_int`は`(f32, i32)`と`RNG`のタプルを返す関数を生成します。
  pub fn rand_double_int<'a>() -> BoxRand<(f32, i32)> {
    Self::both(Self::double_value(), Self::int_value())
  }

  /// `flat_map` generates a function that returns a tuple of `B` and `RNG`.<br/>
  /// `flat_map`は`B`と`RNG`のタプルを返す関数を生成します。
  pub fn flat_map<A, B, F, GF, BF>(mut f: F, mut g: GF) -> BoxRand<B>
  where
    F: FnMut(RNG) -> (A, RNG) + 'static,
    BF: FnMut(RNG) -> (B, RNG) + 'static,
    GF: FnMut(A) -> BF + 'static, {
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
