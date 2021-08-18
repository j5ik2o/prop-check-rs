pub mod choose;
pub mod one;

use crate::gen::choose::Choose;
use crate::gen::one::One;
use crate::rng::{NextRandValue, RNG};
use crate::state::State;
use bigdecimal::Num;
use std::ops::{Range, RangeInclusive};
use itertools::Itertools;
use std::hash::Hash;

pub struct Gens;

impl Gens {
  pub fn list_of_n<B, GF>(n: usize, g: GF) -> Gen<Vec<B>>
  where
    GF: Fn() -> Gen<B>,
    B: Clone + 'static, {
    let mut v: Vec<State<RNG, B>> = Vec::with_capacity(n);
    v.resize_with(n, move || g().sample);
    Gen {
      sample: State::sequence(v),
    }
  }

  pub fn one<T: One>() -> Gen<T> {
    One::one()
  }

  pub fn one_i64() -> Gen<i64> {
    Gen {
      sample: State::<RNG, i64>::new(move |rng: RNG| rng.next_i64()),
    }
  }

  pub fn one_u64() -> Gen<u64> {
    Gen {
      sample: State::<RNG, u64>::new(move |rng: RNG| rng.next_u64()),
    }
  }

  pub fn one_i32() -> Gen<i32> {
    Gen {
      sample: State::<RNG, i32>::new(move |rng: RNG| rng.next_i32()),
    }
  }

  pub fn one_u32() -> Gen<u32> {
    Gen {
      sample: State::<RNG, u16>::new(move |rng: RNG| rng.next_u32()),
    }
  }

  pub fn one_i16() -> Gen<i16> {
    Gen {
      sample: State::<RNG, i16>::new(move |rng: RNG| rng.next_i16()),
    }
  }

  pub fn one_u16() -> Gen<u16> {
    Gen {
      sample: State::<RNG, u32>::new(move |rng: RNG| rng.next_u16()),
    }
  }

  pub fn one_i8() -> Gen<i8> {
    Gen {
      sample: State::<RNG, i8>::new(move |rng: RNG| rng.next_i8()),
    }
  }

  pub fn one_u8() -> Gen<u8> {
    Gen {
      sample: State::<RNG, u8>::new(move |rng: RNG| rng.next_u8()),
    }
  }

  pub fn one_char() -> Gen<char> {
    Self::one_u8().fmap(|v| v as char)
  }

  pub fn one_bool() -> Gen<bool> {
    Gen {
      sample: State::<RNG, bool>::new(|rng: RNG| rng.next_bool()),
    }
  }

  pub fn one_f64() -> Gen<f64> {
    Gen {
      sample: State::<RNG, f64>::new(move |rng: RNG| rng.next_f64()),
    }
  }

  pub fn one_f32() -> Gen<f32> {
    Gen {
      sample: State::<RNG, f32>::new(move |rng: RNG| rng.next_f32()),
    }
  }

  pub fn one_of_vec<T: Choose + Clone + 'static>(values: Vec<T>) -> Gen<T> {
    Self::choose(0usize, values.len() - 1)
        .fmap(move |idx| values[idx as usize].clone())
  }

  pub fn choose<T: Choose>(min: T, max: T) -> Gen<T> {
    Choose::choose(min, max)
  }

  pub fn choose_char(start: char, stop_exclusive: char) -> Gen<char> {
    Self::choose_u8(start as u8, stop_exclusive as u8).fmap(|b| b as char)
  }

  pub fn choose_i64(start: i64, stop_exclusive: i64) -> Gen<i64> {
    Gen {
      sample: State::<RNG, i64>::new(move |rng: RNG| rng.next_i64()),
    }
    .fmap(move |n| start + n % (stop_exclusive - start))
  }

  pub fn choose_u64(min: u64, max: u64) -> Gen<u64> {
    Gen {
      sample: State::<RNG, u64>::new(move |rng: RNG| rng.next_u64()),
    }
    .fmap(move |n| min + n % (max - min + 1))
  }

  pub fn choose_i32(min: i32, max: i32) -> Gen<i32> {
    Gen {
      sample: State::<RNG, i32>::new(move |rng: RNG| rng.next_i32()),
    }
    .fmap(move |n| min + n % (max - min + 1))
  }

  pub fn choose_u32(min: u32, max: u32) -> Gen<u32> {
    Gen {
      sample: State::<RNG, u32>::new(move |rng: RNG| rng.next_u32()),
    }
    .fmap(move |n| min + n % (max - min + 1))
  }

  pub fn choose_i16(min: i16, max: i16) -> Gen<i16> {
    Gen {
      sample: State::<RNG, i16>::new(move |rng: RNG| rng.next_i16()),
    }
    .fmap(move |n| min + n % (max - min + 1))
  }

  pub fn choose_u16(min: u16, max: u16) -> Gen<u16> {
    Gen {
      sample: State::<RNG, u16>::new(move |rng: RNG| rng.next_u16()),
    }
    .fmap(move |n| min + n % (max - min + 1))
  }

  pub fn choose_i8(min: i8, max: i8) -> Gen<i8> {
    Gen {
      sample: State::<RNG, i8>::new(move |rng: RNG| rng.next_i8()),
    }
    .fmap(move |n| min + n % (max - min + 1))
  }

  pub fn choose_u8(min: u8, max: u8) -> Gen<u8> {
    Gen {
      sample: State::<RNG, u8>::new(move |rng: RNG| rng.next_u8()),
    }
    .fmap(move |n| min + n % (max - min + 1))
  }

  pub fn choose_f64(min: f64, max: f64) -> Gen<f64> {
    Gen {
      sample: State::<RNG, f64>::new(move |rng: RNG| rng.next_f64()),
    }
    .fmap(move |d| min + d * (max - min))
  }

  pub fn choose_f32(min: f32, max: f32) -> Gen<f32> {
    Gen {
      sample: State::<RNG, f32>::new(move |rng: RNG| rng.next_f32()),
    }
    .fmap(move |d| min + d * (max - min))
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
