use std::fmt::Display;

use crate::gen::Gen;
use crate::rng::RNG;

use itertools::Unfold;

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

pub fn for_all<A, GF, F>(g: GF, f: F) -> Prop
where
  GF: Fn() -> Gen<A> + 'static,
  F: Fn(A) -> bool + 'static,
  A: Clone + Display + 'static, {
  Prop {
    run_f: Box::new(move |_, n, rng| {
      let nl = itertools::iterate(1, |&i| i + 1).into_iter();
      random_stream(g, rng)
        .zip(nl)
        .take(n as usize)
        .map(|(a, i): (A, u32)| {
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

pub struct Prop {
  run_f: Box<dyn FnOnce(MaxSize, TestCases, RNG) -> Result>,
}

impl Prop {
  pub fn run(self, max_size: MaxSize, test_cases: TestCases, rng: RNG) -> Result {
    (self.run_f)(max_size, test_cases, rng)
  }

  pub fn tag(self, msg: String) -> Prop {
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

  pub fn and(self, p: Self) -> Prop {
    Prop {
      run_f: Box::new(
        move |max: MaxSize, n: TestCases, rng: RNG| match self.run(max, n, rng.clone()) {
          Result::Passed | Result::Proved => p.run(max, n, rng),
          x => x,
        },
      ),
    }
  }

  pub fn or(self, p: Self) -> Prop {
    Prop {
      run_f: Box::new(move |max, n, rng| match self.run(max, n, rng.clone()) {
        Result::Falsified { failure: msg, .. } => p.tag(msg).run(max, n, rng),
        x => x,
      }),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::gen::Gens;
  use crate::{gen, prop};
  use std::slice::Iter;
  use itertools::Itertools;
  use std::ops::RangeInclusive;

  #[test]
  fn choose() {
    let gf = || {
      let range: RangeInclusive<i32> = (1..=10);
      let vec = range.collect_vec();
      Gens::one_of_vec(vec)
    };
    let prop = prop::for_all(gf, |a| {
      println!("prop1:a = {}", a);
      a == a
    });
    prop::run_with_prop(prop, 1, 100, RNG::new());
  }
}
