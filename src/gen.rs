use crate::rng::{NextRandValue, RNG};
use crate::state::State;
use bigdecimal::Num;

pub struct Gens;

impl Gens {
  pub fn list_of_n<B, GF>(n: u32, g: GF) -> Gen<Vec<B>>
  where
    GF: Fn() -> Gen<B>,
    B: Clone + 'static, {
    let mut v: Vec<State<RNG, B>> = Vec::with_capacity(n as usize);
    v.resize_with(n as usize, move || g().sample);
    Gen::<B>::new(State::sequence(v))
  }

  pub fn bool() -> Gen<bool> {
    Gen::<bool>::new(State::<RNG, bool>::new(|rng: RNG| rng.next_bool()))
  }

  pub fn uniform() -> Gen<f32> {
    Gen::<f32>::new(State::<RNG, f32>::new(move |rng: RNG| rng.next_f32()))
  }

  pub fn choose_u32(start: u32, stop_exclusive: u32) -> Gen<u32> {
    Gen::<u32>::new(State::<RNG, u32>::new(move |rng: RNG| rng.next_u32()))
      .fmap(move |n| start.clone() + n % (stop_exclusive - start))
  }

  pub fn choose_f32(i: f32, j: f32) -> Gen<f32> {
    Gen::<f32>::new(State::<RNG, f32>::new(move |rng: RNG| rng.next_f32()))
        .fmap(move |d| i + d * (j - i))
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
    .fmap(|n| if n % 2 == 0 { n + 1 } else { n })
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
