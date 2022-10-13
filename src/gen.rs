pub mod choose;
pub mod one;

use crate::gen::choose::Choose;
use crate::gen::one::One;
use crate::rng::{NextRandValue, RNG};
use crate::state::State;
use bigdecimal::Num;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::rc::Rc;

pub struct Gens;

impl Gens {
  pub fn unit() -> Gen<()> {
    Self::pure(())
  }

  pub fn pure<B>(value: B) -> Gen<B>
  where
    B: Clone + 'static, {
    Gen::<B>::new(State::unit(value))
  }

  pub fn pure_lazy<B, F>(f: F) -> Gen<B>
  where
    F: Fn() -> B + 'static,
    B: Clone + 'static, {
    Self::pure(()).map(move |_| f())
  }

  pub fn some<B>(g: Gen<B>) -> Gen<Option<B>>
  where
    B: Clone + 'static, {
    g.map(Some)
  }

  pub fn option<B>(g: Gen<B>) -> Gen<Option<B>>
  where
    B: Debug + Clone + 'static, {
    Self::frequency([(1, Self::pure(None)), (9, Self::some(g))])
  }

  pub fn either<T, E>(gt: Gen<T>, ge: Gen<E>) -> Gen<Result<T, E>>
  where
    T: Choose + Clone + 'static,
    E: Clone + 'static, {
    Self::one_of([gt.map(Ok), ge.map(Err)])
  }

  pub fn frequency_values<B>(values: impl IntoIterator<Item = (u32, B)>) -> Gen<B>
  where
    B: Debug + Clone + 'static, {
    Self::frequency(values.into_iter().map(|(n, v)| (n, Gens::pure(v))))
  }

  pub fn frequency<B>(values: impl IntoIterator<Item = (u32, Gen<B>)>) -> Gen<B>
  where
    B: Debug + Clone + 'static, {
    let filtered = values.into_iter().filter(|kv| kv.0 > 0).collect::<Vec<_>>();
    let (tree, total) = filtered
      .into_iter()
      .fold((BTreeMap::new(), 0), |(mut tree, total), (weight, value)| {
        let t = total + weight;
        tree.insert(t, value.clone());
        (tree, t)
      });
    Self::choose_u32(1, total).flat_map(move |n| tree.range(n..).into_iter().next().unwrap().1.clone())
  }

  pub fn list_of_n<B>(n: usize, g: Gen<B>) -> Gen<Vec<B>>
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

  pub fn one_of<T: Choose + Clone + 'static>(values: impl IntoIterator<Item = Gen<T>>) -> Gen<T> {
    let mut vec = vec![];
    vec.extend(values.into_iter());
    Self::choose(0usize, vec.len() - 1).flat_map(move |idx| vec[idx as usize].clone())
  }

  pub fn one_of_values<T: Choose + Clone + 'static>(values: impl IntoIterator<Item = T>) -> Gen<T> {
    Self::one_of(values.into_iter().map(Gens::pure))
  }

  pub fn choose<T: Choose>(min: T, max: T) -> Gen<T> {
    Choose::choose(min, max)
  }

  pub fn choose_char(min: char, max: char) -> Gen<char> {
    let chars = (min..=max).into_iter().map(|e| Self::pure(e)).collect::<Vec<_>>();
    Self::one_of(chars)
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

#[derive(Debug)]
pub struct Gen<A> {
  sample: State<RNG, A>,
}

impl<A: Clone + 'static> Clone for Gen<A> {
  fn clone(&self) -> Self {
    Self {
      sample: self.sample.clone(),
    }
  }
}

impl<A: Clone + 'static> Gen<A> {
  pub fn run(self, rng: RNG) -> (A, RNG) {
    self.sample.run(rng)
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

  pub fn flat_map<B, F>(self, f: F) -> Gen<B>
  where
    F: Fn(A) -> Gen<B> + 'static,
    B: Clone + 'static, {
    Self::new(self.sample.flat_map(move |a| f(a).sample))
  }
}

pub enum SGen<A> {
  Sized(Rc<RefCell<dyn Fn(u32) -> Gen<A>>>),
  Unsized(Gen<A>),
}

impl<A: Clone + 'static> Clone for SGen<A> {
  fn clone(&self) -> Self {
    match self {
      SGen::Sized(f) => SGen::Sized(f.clone()),
      SGen::Unsized(g) => SGen::Unsized(g.clone()),
    }
  }
}

impl<A: Clone + 'static> SGen<A> {
  pub fn of_sized<F>(f: F) -> SGen<A>
  where
    F: Fn(u32) -> Gen<A> + 'static, {
    SGen::Sized(Rc::new(RefCell::new(f)))
  }

  pub fn of_unsized(gen: Gen<A>) -> SGen<A> {
    SGen::Unsized(gen)
  }

  pub fn run(&self, i: Option<u32>) -> Gen<A> {
    match self {
      SGen::Sized(f) => {
        let mf = f.borrow_mut();
        mf(i.unwrap())
      }
      SGen::Unsized(g) => g.clone(),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::prop;
  use anyhow::Result;

  use std::cell::RefCell;
  use std::collections::HashMap;
  use std::env;
  use std::rc::Rc;

  #[ctor::ctor]
  fn init() {
    env::set_var("RUST_LOG", "info");
    let _ = env_logger::builder().is_test(true).try_init();
  }

  fn new_rng() -> RNG {
    RNG::new()
  }

  pub mod laws {
    use super::*;

    #[test]
    fn test_left_identity_law() -> Result<()> {
      let gen = Gens::choose_i32(1, i32::MAX / 2).map(|e| (RNG::new_with_seed(e as u64), e));
      let f = |x| Gens::pure(x);
      let laws_prop = prop::for_all_gen(gen, move |(s, n)| {
        Gens::pure(n).flat_map(f).run(s.clone()) == f(n).run(s)
      });
      prop::test_with_prop(laws_prop, 1, 100, new_rng())
    }

    #[test]
    fn test_right_identity_law() -> Result<()> {
      let gen = Gens::choose_i32(1, i32::MAX / 2).map(|e| (RNG::new_with_seed(e as u64), e));

      let laws_prop = prop::for_all_gen(gen, move |(s, x)| {
        Gens::pure(x).flat_map(|y| Gens::pure(y)).run(s.clone()) == Gens::pure(x).run(s)
      });

      prop::test_with_prop(laws_prop, 1, 100, new_rng())
    }

    #[test]
    fn test_associativity_law() -> Result<()> {
      let gen = Gens::choose_i32(1, i32::MAX / 2).map(|e| (RNG::new_with_seed(e as u64), e));
      let f = |x| Gens::pure(x * 2);
      let g = |x| Gens::pure(x + 1);
      let laws_prop = prop::for_all_gen(gen, move |(s, x)| {
        Gens::pure(x).flat_map(f).flat_map(g).run(s.clone()) == f(x).flat_map(g).run(s)
      });
      prop::test_with_prop(laws_prop, 1, 100, new_rng())
    }
  }

  #[test]
  fn test_frequency() -> Result<()> {
    let gens = [
      (1, Gens::choose_u32(1, 10)),
      (1, Gens::choose_u32(50, 100)),
      (1, Gens::choose_u32(200, 300)),
    ];
    let gen = Gens::frequency(gens);
    let prop = prop::for_all_gen(gen, move |a| {
      log::info!("a: {}", a);
      if a >= 1 && a <= 10 {
        true
      } else if a >= 50 && a <= 100 {
        true
      } else if a >= 200 && a <= 300 {
        true
      } else {
        false
      }
    });
    prop::test_with_prop(prop, 1, 100, new_rng())
  }

  #[test]
  fn test_frequency_values() -> Result<()> {
    let result = Rc::new(RefCell::new(HashMap::new()));
    let cloned_map = result.clone();
    let gens = [(1, "a"), (1, "b"), (8, "c")];
    let gen = Gens::frequency_values(gens);
    let prop = prop::for_all_gen(gen, move |a| {
      let mut map = result.borrow_mut();
      let r = map.entry(a).or_insert_with(|| 0);
      *r += 1;
      true
    });
    let r = prop::test_with_prop(prop, 1, 100, new_rng());
    let map = cloned_map.borrow();
    let a_count = map.get(&"a").unwrap();
    let b_count = map.get(&"b").unwrap();
    let c_count = map.get(&"c").unwrap();
    assert_eq!(*a_count + *b_count + *c_count, 100);
    println!("{cloned_map:?}");
    r
  }
}
