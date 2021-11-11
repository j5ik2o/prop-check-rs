pub mod choose;
pub mod one;

use crate::gen::choose::Choose;
use crate::gen::one::One;
use crate::rng::{NextRandValue, RNG};
use crate::state::State;
use bigdecimal::Num;

pub struct Gens;

impl Gens {
  pub fn list_of_n<B>(n: usize, mut g: Gen<B>) -> Gen<Vec<B>>
  where
    B: Clone + 'static, {
    let mut v: Vec<State<RNG, B>> = Vec::with_capacity(n);
    v.resize_with(n, move || g.clone().sample);
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
    Self::one_u8().map(|v| v as char)
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
    Self::choose(0usize, values.len() - 1).map(move |idx| values[idx as usize].clone())
  }

  pub fn choose<T: Choose>(min: T, max: T) -> Gen<T> {
    Choose::choose(min, max)
  }

  pub fn choose_char(min: char, max: char) -> Gen<char> {
    let chars = (min..=max).into_iter().collect::<Vec<char>>();
    Self::one_of_vec(chars)
  }

  pub fn choose_i64(min: i64, max: i64) -> Gen<i64> {
    Gen {
      sample: State::<RNG, i64>::new(move |rng: RNG| rng.next_i64()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  pub fn choose_u64(min: u64, max: u64) -> Gen<u64> {
    Gen {
      sample: State::<RNG, u64>::new(move |rng: RNG| rng.next_u64()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  pub fn choose_i32(min: i32, max: i32) -> Gen<i32> {
    Gen {
      sample: State::<RNG, i32>::new(move |rng: RNG| rng.next_i32()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  pub fn choose_u32(min: u32, max: u32) -> Gen<u32> {
    Gen {
      sample: State::<RNG, u32>::new(move |rng: RNG| rng.next_u32()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  pub fn choose_i16(min: i16, max: i16) -> Gen<i16> {
    Gen {
      sample: State::<RNG, i16>::new(move |rng: RNG| rng.next_i16()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  pub fn choose_u16(min: u16, max: u16) -> Gen<u16> {
    Gen {
      sample: State::<RNG, u16>::new(move |rng: RNG| rng.next_u16()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  pub fn choose_i8(min: i8, max: i8) -> Gen<i8> {
    Gen {
      sample: State::<RNG, i8>::new(move |rng: RNG| rng.next_i8()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  pub fn choose_u8(min: u8, max: u8) -> Gen<u8> {
    Gen {
      sample: State::<RNG, u8>::new(move |rng: RNG| rng.next_u8()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  pub fn choose_f64(min: f64, max: f64) -> Gen<f64> {
    Gen {
      sample: State::<RNG, f64>::new(move |rng: RNG| rng.next_f64()),
    }
    .map(move |d| min + d * (max - min))
  }

  pub fn choose_f32(min: f32, max: f32) -> Gen<f32> {
    Gen {
      sample: State::<RNG, f32>::new(move |rng: RNG| rng.next_f32()),
    }
    .map(move |d| min + d * (max - min))
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
    .map(move |n| if n % two == T::zero() { n + T::one() } else { n })
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
    .map(move |n| if n % two != T::zero() { n + T::one() } else { n })
  }
}

pub struct Gen<A> {
  pub sample: State<RNG, A>,
}

impl<A: Clone + 'static> Clone for Gen<A> {
  fn clone(&self) -> Self {
    Self {
      sample: self.sample.clone(),
    }
  }
}

impl<A: Clone + 'static> Gen<A> {
  pub fn unit<B, F>(mut f: F) -> Gen<B>
  where
    F: FnMut() -> B,
    B: Clone + 'static, {
    Gen::<B>::new(State::unit(f()))
  }

  pub fn new<B>(b: State<RNG, B>) -> Gen<B> {
    Gen { sample: b }
  }

  pub fn map<B, F>(self, f: F) -> Gen<B>
  where
    F: Fn(A) -> B + 'static,
    B: Clone + 'static, {
    Self::new(self.sample.map(f))
  }

  pub fn and_then<B, C, F>(self, g: Gen<B>, f: F) -> Gen<C>
  where
    F: Fn(A, B) -> C + 'static,
    A: Clone,
    B: Clone + 'static,
    C: Clone + 'static, {
    Self::new(self.sample.and_then(g.sample).map(move |(a, b)| f(a, b)))
  }

  pub fn flat_map<B, F>(self, mut f: F) -> Gen<B>
  where
    F: Fn(A) -> Gen<B> + 'static,
    B: Clone + 'static, {
    Self::new(self.sample.flat_map(move |a| f(a).sample))
  }
}
