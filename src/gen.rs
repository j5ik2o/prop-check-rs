use crate::rng::{NextRandValue, RNG};
use crate::state::State;

pub struct Gen<A> {
  pub sample: State<RNG, A>,
}

pub mod gen {
  use super::*;

  pub fn bool() -> Gen<bool> {
    Gen::<bool>::new(State::<RNG, bool>::new(|rng: RNG| rng.next_bool()))
  }
}

impl<A: 'static> Gen<A> {
  fn cloned(self) -> Self {
    Self {
      sample: self.sample.cloned(),
    }
  }

  pub fn unit<F>(f: F) -> Gen<A>
  where
    F: FnOnce() -> A, {
    Self::new(State::unit(f()))
  }

  // pub fn list_of_n_from_gen(n: u32, g: Gen<A>) -> Gen<Vec<A>>
  // where
  //   A: Clone, {
  //   let mut v: Vec<State<RNG, A>> = Vec::with_capacity(n as usize);
  //   v.resize_with(n as usize, move || g.sample.cloned());
  //   Self::new(State::sequence(v))
  // }
  //
  // pub fn list_of_n(self, size: u32) -> Gen<Vec<A>>
  // where
  //   A: Clone, {
  //   Self::list_of_n_from_gen(size, self)
  // }

  pub fn uniform() -> Gen<f32> {
    Self::new(State::<RNG, f32>::new(move |rng: RNG| rng.next_f32()))
  }

  fn choose_u32(start: u32, stop_exclusive: u32) -> Gen<u32> {
    Self::new(State::<RNG, u32>::new(move |rng: RNG| rng.next_u32()))
      .fmap(move |n| start.clone() + n % (stop_exclusive - start))
  }

  fn choose_f32(i: f32, j: f32) -> Gen<f32> {
    Self::new(State::<RNG, f32>::new(move |rng: RNG| rng.next_f32())).fmap(move |d| i + d * (j - i))
  }

  pub fn even(start: u32, stop_exclusive: u32) -> Gen<u32> {
    Self::choose_u32(
      start,
      if stop_exclusive % 2 == 0 {
        stop_exclusive - 1
      } else {
        stop_exclusive
      },
    )
    .fmap(move |n| if n % 2 == 0 { n + 1 } else { n })
  }

  pub fn odd(start: u32, stop_exclusive: u32) -> Gen<u32> {
    Self::choose_u32(
      start,
      if stop_exclusive % 2 != 0 {
        stop_exclusive - 1
      } else {
        stop_exclusive
      },
    )
    .fmap(move |n| if n % 2 != 0 { n + 1 } else { n })
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
