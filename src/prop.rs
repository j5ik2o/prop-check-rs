use std::fmt::Display;
use std::rc::Rc;

use crate::gen::Gen;
use crate::rng::{NextRandValue, RNG};
use crate::state::State;
use itertools::{Itertools, Unfold};

pub type MaxSize = u32;
pub type TestCases = u32;
pub type FailedCase = String;
pub type SuccessCount = u32;

pub trait IsFalsified {
  fn is_falsified(&self) -> bool;
}

#[derive(Clone)]
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

pub struct Prop<'a> {
  run_f: Box<dyn FnOnce(MaxSize, TestCases, RNG) -> Result + 'a>,
}

impl<'a> Prop<'a> {
  pub fn run(self, max_size: MaxSize, test_cases: TestCases, rng: RNG) -> Result {
    (self.run_f)(max_size, test_cases, rng)
  }

  pub fn tag(self, msg: String) -> Prop<'a> {
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

  pub fn and(self, p: Self) -> Prop<'a> {
    Prop {
      run_f: Box::new(
        move |max: MaxSize, n: TestCases, rng: RNG| match self.run(max, n, rng.clone()) {
          Result::Passed | Result::Proved => p.run(max, n, rng),
          x => x,
        },
      ),
    }
  }

  pub fn or(self, p: Self) -> Prop<'a> {
    Prop {
      run_f: Box::new(move |max, n, rng| match self.run(max, n, rng.clone()) {
        Result::Falsified { failure: msg, .. } => p.tag(msg).run(max, n, rng),
        x => x,
      }),
    }
  }

  pub fn random_stream<A, GF>(g: GF, rng: RNG) -> Unfold<RNG, Box<dyn Fn(&mut RNG) -> Option<A>>>
  where
    GF: Fn() -> Gen<A> + 'static,
    A: Clone + 'static, {
    itertools::unfold(
      rng,
      Box::new(move |rng| {
        let (a, s) = g().sample.run(rng.clone());
        *rng = s;
        Some(a)
      }),
    )
  }

  pub fn for_all<A, GF, F>(g: GF, f: F) -> Prop<'a>
  where
    GF: Fn() -> Gen<A> + 'static,
    F: Fn(A) -> bool + 'static,
    A: Clone + Display + 'static, {
    Prop {
      run_f: Box::new(move |_, n, rng| {
        Prop::random_stream(g, rng)
          .zip(itertools::unfold(0u32, move |n| Some(*n + 1)).into_iter())
          .take(n as usize)
          .map(move |(a, i): (A, u32)| {
            if f(a.clone()) {
              Result::Passed
            } else {
              Result::Falsified {
                failure: a.to_string(),
                successes: i,
              }
            }
          })
          .find(move |e| e.is_falsified())
          .unwrap_or(Result::Passed)
      }),
    }
  }

  pub fn run_with_prop(p: Prop, max_size: MaxSize, test_cases: TestCases, rng: RNG) {
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
  use crate::gen::gen;

  #[test]
  fn choose() {
    let gen = || gen::bool();
    let prop = Prop::for_all(gen, |a| {
      println!("a = {}", a);
      a == a
    });
    Prop::run_with_prop(prop, 1, 100, RNG::new());
  }
}
