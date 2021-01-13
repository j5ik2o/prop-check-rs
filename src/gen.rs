use crate::rng::{NextRandValue, RNG};
use crate::state::State;

#[derive(Clone)]
pub struct Gen<'a, A> {
  pub(crate) sample: State<'a, RNG, A>,
}

impl<'a, A> Gen<'a, A> {
  pub fn unit<F>(f: F) -> Gen<'a, A>
  where
    F: FnOnce() -> A,
    A: 'a, {
    Self::new(State::unit(f()))
  }

  pub fn bool() -> Gen<'a, bool>
  where
    A: 'a, {
    Self::new(State::<'a, RNG, bool>::new(|rng: RNG| rng.next_bool()))
  }

  pub fn list_of_n_from_gen(n: u32, g: Gen<'a, A>) -> Gen<'a, Vec<A>>
  where
    A: Clone + 'a, {
    let mut v: Vec<State<RNG, A>> = Vec::with_capacity(n as usize);
    v.resize_with(n as usize, move || g.sample.clone());
    Self::new(State::sequence(v))
  }

  pub fn list_of_n(self, size: u32) -> Gen<'a, Vec<A>>
  where
    A: Clone + 'a, {
    Self::list_of_n_from_gen(size, self)
  }

  pub fn uniform() -> Gen<'a, f32> {
    Self::new(State::<'a, RNG, f32>::new(move |rng: RNG| rng.next_f32()))
  }

  fn choose_u32(start: u32, stop_exclusive: u32) -> Gen<'a, u32> {
    Self::new(State::<'a, RNG, u32>::new(move |rng: RNG| rng.next_u32()))
      .fmap(move |n| start.clone() + n % (stop_exclusive - start))
  }

  fn choose_f32(i: f32, j: f32) -> Gen<'a, f32> {
    Self::new(State::<'a, RNG, f32>::new(move |rng: RNG| rng.next_f32())).fmap(move |d| i + d * (j - i))
  }

  pub fn even(start: u32, stop_exclusive: u32) -> Gen<'a, u32> {
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

  pub fn odd(start: u32, stop_exclusive: u32) -> Gen<'a, u32> {
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

  pub fn fmap<B, F>(self, f: F) -> Gen<'a, B>
  where
    F: FnOnce(A) -> B + 'a,
    A: 'a,
    B: Clone + 'a, {
    Self::new(self.sample.fmap(f))
  }

  pub fn fmap2<'b, 'c, B, C, F>(self, g: Gen<'b, B>, f: F) -> Gen<'c, C>
  where
    F: FnOnce(A, B) -> C + 'c,
    A: Clone + 'a,
    B: Clone + 'b,
    C: Clone + 'c,
    'a: 'b,
    'b: 'c, {
    Self::new(self.sample.fmap2(g.sample, f))
  }

  pub fn bind<B, F>(self, f: F) -> Gen<'a, B>
  where
    F: FnOnce(A) -> Gen<'a, B> + 'a,
    A: 'a,
    B: Clone + 'a, {
    Self::new(self.sample.bind(|a| f(a).sample))
  }
}
