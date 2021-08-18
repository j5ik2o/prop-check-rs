pub mod choose;
pub mod one;

use crate::rng::{NextRandValue, RNG};
use crate::state::State;
use bigdecimal::Num;
use crate::gen::choose::Choose;
use crate::gen::one::One;

pub struct Gens;

impl Gens {
  pub fn list_of_n<B, GF>(n: usize, g: GF) -> Gen<Vec<B>>
  where
    GF: Fn() -> Gen<B>,
    B: Clone + 'static, {
    let mut v: Vec<State<RNG, B>> = Vec::with_capacity(n);
    v.resize_with(n, move || g().sample);
    Gen{ sample: State::sequence(v) }
  }

  pub fn one<T: One>() -> Gen<T> {
    One::one()
  }

  pub fn one_i64() -> Gen<i64> {
    Gen{ sample: State::<RNG, i64>::new(move |rng: RNG| rng.next_i64()) }
  }

  pub fn one_u64() -> Gen<u64> {
    Gen{ sample: State::<RNG, u64>::new(move |rng: RNG| rng.next_u64()) }
  }

  pub fn one_i32() -> Gen<i32> {
    Gen{ sample: State::<RNG, i32>::new(move |rng: RNG| rng.next_i32()) }
  }

  pub fn one_u32() -> Gen<u32> {
    Gen{ sample: State::<RNG, u16>::new(move |rng: RNG| rng.next_u32()) }
  }

  pub fn one_i16() -> Gen<i16> {
    Gen{ sample: State::<RNG, i16>::new(move |rng: RNG| rng.next_i16()) }
  }

  pub fn one_u16() -> Gen<u16> {
    Gen{ sample: State::<RNG, u32>::new(move |rng: RNG| rng.next_u16()) }
  }

  pub fn one_i8() -> Gen<i8> {
    Gen{ sample: State::<RNG, i8>::new(move |rng: RNG| rng.next_i8()) }
  }

  pub fn one_u8() -> Gen<u8> {
    Gen{ sample: State::<RNG, u8>::new(move |rng: RNG| rng.next_u8()) }
  }

  pub fn one_char() -> Gen<char> {
    Self::one_u8().fmap(|v| v as char)
  }
  pub fn one_bool() -> Gen<bool> {
    Gen{ sample: State::<RNG, bool>::new(|rng: RNG| rng.next_bool()) }
  }

  pub fn one_f64() -> Gen<f64> {
    Gen{ sample: State::<RNG, f64>::new(move |rng: RNG| rng.next_f64()) }
  }

  pub fn one_f32() -> Gen<f32> {
    Gen{ sample: State::<RNG, f32>::new(move |rng: RNG| rng.next_f32()) }
  }

  pub fn choose<T: Choose>(min: T, max: T) -> Gen<T> {
    Choose::choose(min, max)
  }

  pub fn choose_char(start: char, stop_exclusive: char) -> Gen<char> {
    Self::choose_u8(start as u8, stop_exclusive as u8).fmap(|b| b as char)
  }

  pub fn choose_i64(start: i64, stop_exclusive: i64) -> Gen<i64> {
    Gen{ sample: State::<RNG, i64>::new(move |rng: RNG| rng.next_i64()) }
        .fmap(move |n| start + n % (stop_exclusive - start))
  }

  pub fn choose_u64(start: u64, stop_exclusive: u64) -> Gen<u64> {
    Gen{ sample: State::<RNG, u64>::new(move |rng: RNG| rng.next_u64()) }
        .fmap(move |n| start + n % (stop_exclusive - start))
  }

  pub fn choose_i32(start: i32, stop_exclusive: i32) -> Gen<i32> {
    Gen{ sample: State::<RNG, i32>::new(move |rng: RNG| rng.next_i32()) }
        .fmap(move |n| start + n % (stop_exclusive - start))
  }

  pub fn choose_u32(start: u32, stop_exclusive: u32) -> Gen<u32> {
    Gen{ sample: State::<RNG, u32>::new(move |rng: RNG| rng.next_u32()) }
      .fmap(move |n| start + n % (stop_exclusive - start))
  }

  pub fn choose_i16(start: i16, stop_exclusive: i16) -> Gen<i16> {
    Gen{ sample: State::<RNG, i16>::new(move |rng: RNG| rng.next_i16()) }
        .fmap(move |n| start + n % (stop_exclusive - start))
  }

  pub fn choose_u16(start: u16, stop_exclusive: u16) -> Gen<u16> {
    Gen{ sample: State::<RNG, u16>::new(move |rng: RNG| rng.next_u16()) }
        .fmap(move |n| start + n % (stop_exclusive - start))
  }

  pub fn choose_i8(start: i8, stop_exclusive: i8) -> Gen<i8> {
    Gen{ sample: State::<RNG, i8>::new(move |rng: RNG| rng.next_i8()) }
        .fmap(move |n| start + n % (stop_exclusive - start))
  }

  pub fn choose_u8(start: u8, stop_exclusive: u8) -> Gen<u8> {
    Gen{ sample: State::<RNG, u8>::new(move |rng: RNG| rng.next_u8()) }
        .fmap(move |n| start + n % (stop_exclusive - start))
  }

  pub fn choose_f64(i: f64, j: f64) -> Gen<f64> {
    Gen{ sample: State::<RNG, f64>::new(move |rng: RNG| rng.next_f64()) }
        .fmap(move |d| i + d * (j - i))
  }

  pub fn choose_f32(i: f32, j: f32) -> Gen<f32> {
    Gen{ sample: State::<RNG, f32>::new(move |rng: RNG| rng.next_f32()) }
        .fmap(move |d| i + d * (j - i))
  }

  pub fn even<T: Choose + Num + Copy + 'static>(start: T, stop_exclusive: T) -> Gen<T> {
    let two = T::one().add(T::one());
    Self::choose(
      start,
      if stop_exclusive % two == T::zero() {
        stop_exclusive - T::one()
      } else {
        stop_exclusive
      },
    )
    .fmap(move |n| if n % two == T::zero() { n + T::one() } else { n })
  }

  pub fn odd<T: Choose + Num + Copy + 'static>(start: T, stop_exclusive: T) -> Gen<T> {
    let two = T::one().add(T::one());
    Self::choose(
      start,
      if stop_exclusive % two != T::zero() {
        stop_exclusive - T::one()
      } else {
        stop_exclusive
      },
    )
    .fmap(move |n| if n % two != T::zero() { n + T::one() } else { n })
  }
}

pub struct Gen<A> {
  pub sample: State<RNG, A>,
}

impl<A: 'static> Gen<A> {
  pub fn unit<B, F>(f: F) -> Gen<B>
  where
    F: FnOnce() -> B,
    B: 'static, {
    Gen::<B>::new(State::unit(f()))
  }

  pub fn new<B>(b: State<RNG, B>) -> Gen<B> {
    Gen { sample: b }
  }

  pub fn fmap<B, F>(self, f: F) -> Gen<B>
  where
    F: FnOnce(A) -> B + 'static,
    B: Clone + 'static, {
    Self::new(self.sample.fmap(f))
  }

  pub fn fmap2<B, C, F>(self, g: Gen<B>, f: F) -> Gen<C>
  where
    F: FnOnce(A, B) -> C + 'static,
    A: Clone,
    B: Clone + 'static,
    C: Clone + 'static, {
    Self::new(self.sample.fmap2(g.sample, f))
  }

  pub fn bind<B, F>(self, f: F) -> Gen<B>
  where
    F: FnOnce(A) -> Gen<B> + 'static,
    B: Clone + 'static, {
    Self::new(self.sample.bind(|a| f(a).sample))
  }
}

