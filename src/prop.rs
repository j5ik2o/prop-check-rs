use std::rc::Rc;

use crate::rng::{NextRandValue, RNG};
use crate::state::State;

pub type MaxSize = i32;
pub type TestCases = i32;
pub type FailedCase = String;
pub type SuccessCount = i32;

pub trait IsFalsified {
  fn is_falsified(&self) -> bool;
}

pub enum Result {
  Passed,
  Falsified {
    failure: FailedCase,
    successes: SuccessCount,
  },
  Proved,
}

impl IsFalsified for Result {
  fn is_falsified(&self) -> bool {
    match self {
      Result::Passed => false,
      Result::Falsified { .. } => true,
      Result::Proved => false,
    }
  }
}

pub struct Gen<'a, A> {
  sample: State<'a, RNG, A>,
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

  pub fn choose(start: u32, stop_exclusive: u32) -> Gen<'a, u32> {
    Self::new(State::<'a, RNG, u32>::new(move |rng: RNG| rng.next_u32()))
      .fmap(move |n| start.clone() + n % (stop_exclusive - start))
  }

  pub fn list_of_n_(n: u32, g: Gen<'a, A>) -> Gen<'a, Vec<A>>
  where
    A: Clone + 'a, {
    let mut v: Vec<State<RNG, A>> = Vec::with_capacity(n as usize);
    v.resize_with(n as usize, move || g.sample.clone());
    Self::new(State::sequence(v))
  }

  pub fn list_of_n(self, size: u32) -> Gen<'a, Vec<A>>
  where
    A: Clone + 'a, {
    Self::list_of_n_(size, self)
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

pub struct Prop<'a> {
  run_f: Box<dyn Fn(MaxSize, TestCases, RNG) -> Result + 'a>,
}

impl<'a> Prop<'a> {
  pub fn run(&self, max_size: MaxSize, test_cases: TestCases, rng: RNG) -> Result {
    (self.run_f)(max_size, test_cases, rng)
  }

  pub fn tag(&self, msg: String) -> Prop {
    Prop {
      run_f: Box::new(move |max, n, rng| match self.run(max, n, rng) {
        Result::Falsified {
          failure: e,
          successes: c,
        } => Result::Falsified {
          failure: format!("{}\n{}", msg, e),
          successes: c,
        },
        x => x,
      }),
    }
  }

  pub fn and(&self, p: Self) -> Prop {
    Prop {
      run_f: Box::new(
        move |max: MaxSize, n: TestCases, rng: RNG| match self.run(max, n, rng.clone()) {
          Result::Passed | Result::Proved => p.run(max, n, rng),
          x => x,
        },
      ),
    }
  }

  pub fn or(&self, p: Self) -> Prop {
    Prop {
      run_f: Box::new(move |max, n, rng| match self.run(max, n, rng.clone()) {
        Result::Falsified { failure: msg, .. } => p.tag(msg).run(max, n, rng),
        x => x,
      }),
    }
  }

  pub fn run_(p: Prop, max_size: i32, test_cases: i32, rng: RNG) {
    match p.run(max_size, test_cases, rng) {
      Result::Falsified {
        failure: msg,
        successes: n,
      } => println!("! Falsified after {} passed tests:\n {}", n, msg),
      Result::Passed => println!("+ OK, passed {} tests.", test_cases),
      Result::Proved => println!("+ OK, proved property."),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  // #[test]
  // fn choose() {
  //   let rng = RNG::new();
  //   Gen::choose(1, 10)
  //   let r = Gen::bool().sample.run(rng);
  //
  // }


}