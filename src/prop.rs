use std::fmt::Display;

use anyhow::*;
use itertools::Unfold;

use crate::gen::Gen;
use crate::rng::RNG;

pub type MaxSize = u32;
pub type TestCases = u32;
pub type FailedCase = String;
pub type SuccessCount = u32;

pub trait IsFalsified {
  fn is_falsified(&self) -> bool;
}

#[derive(Clone)]
pub enum PropResult {
  Passed,
  Falsified {
    failure: FailedCase,
    successes: SuccessCount,
  },
  Proved,
}

impl IsFalsified for PropResult {
  fn is_falsified(&self) -> bool {
    match self {
      PropResult::Passed => false,
      PropResult::Falsified { .. } => true,
      PropResult::Proved => false,
    }
  }
}

pub fn random_stream<A, GF>(mut g: GF, rng: RNG) -> Unfold<RNG, Box<dyn FnMut(&mut RNG) -> Option<A>>>
where
  GF: FnMut() -> Gen<A> + 'static,
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

pub fn for_all<A, GF, F>(g: GF, mut f: F) -> Prop
where
  GF: FnMut() -> Gen<A> + 'static,
  F: FnMut(A) -> bool + 'static,
  A: Clone + Display + 'static, {
  Prop {
    run_f: Box::new(move |_, n, rng| {
      let nl = itertools::iterate(1, |&i| i + 1).into_iter();
      random_stream(g, rng)
        .zip(nl)
        .take(n as usize)
        .map(|(a, i): (A, u32)| {
          if f(a.clone()) {
            PropResult::Passed
          } else {
            PropResult::Falsified {
              failure: a.to_string(),
              successes: i,
            }
          }
        })
        .find(move |e| e.is_falsified())
        .unwrap_or(PropResult::Passed)
    }),
  }
}

pub fn run_with_prop(p: Prop, max_size: MaxSize, test_cases: TestCases, rng: RNG) -> Result<String> {
  match p.run(max_size, test_cases, rng) {
    PropResult::Passed => Ok(format!("+ OK, passed {} tests.", test_cases)),
    PropResult::Proved => Ok("+ OK, proved property.".to_string()),
    PropResult::Falsified {
      failure: msg,
      successes: n,
    } => Err(anyhow!("! Falsified after {} passed tests:\n {}", msg, n)),
  }
}

pub fn test_with_prop(p: Prop, max_size: MaxSize, test_cases: TestCases, rng: RNG) -> Result<()> {
  match p.run(max_size, test_cases, rng) {
    PropResult::Passed => {
      log::info!("+ OK, passed {} tests.", test_cases);
      Ok(())
    }
    PropResult::Proved => {
      log::info!("{}", "+ OK, proved property.".to_string());
      Ok(())
    }
    PropResult::Falsified {
      failure: msg,
      successes: n,
    } => {
      let error_message = format!("! Falsified after {} passed tests:\n {}", msg, n);
      log::error!("{}", error_message);
      Err(anyhow!(error_message))
    }
  }
}

pub struct Prop {
  run_f: Box<dyn FnOnce(MaxSize, TestCases, RNG) -> PropResult>,
}

impl Prop {
  pub fn run(self, max_size: MaxSize, test_cases: TestCases, rng: RNG) -> PropResult {
    (self.run_f)(max_size, test_cases, rng)
  }

  pub fn tag(self, msg: String) -> Prop {
    Prop {
      run_f: Box::new(move |max, n, rng| match self.run(max, n, rng) {
        PropResult::Falsified {
          failure: e,
          successes: c,
        } => PropResult::Falsified {
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
          PropResult::Passed | PropResult::Proved => p.run(max, n, rng),
          x => x,
        },
      ),
    }
  }

  pub fn or(self, p: Self) -> Prop {
    Prop {
      run_f: Box::new(move |max, n, rng| match self.run(max, n, rng.clone()) {
        PropResult::Falsified { failure: msg, .. } => p.tag(msg).run(max, n, rng),
        x => x,
      }),
    }
  }
}

#[cfg(test)]
mod tests {
  use log::{debug, error, info, log_enabled, Level};
  

  

  use crate::gen::Gens;
  use crate::prop;

  use super::*;
  use anyhow::Result;
  use std::env;

  fn init() {
    env::set_var("RUST_LOG", "info");
    let _ = env_logger::builder().is_test(true).try_init();
  }

  #[test]
  fn choose() -> Result<(), Error> {
    init();
    let mut counter = 0;
    let gf = || Gens::one_of_vec(vec!['a', 'b', 'c', 'x', 'y', 'z']);
    let prop = prop::for_all(gf, move |a| {
      counter += 1;
      info!("prop1:a = {}", a);
      a == a
    });
    prop::test_with_prop(prop, 1, 100, RNG::new())
  }
}
