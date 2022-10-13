use crate::gen::{Gen, SGen};
use crate::rng::RNG;

use anyhow::*;
use itertools::Unfold;
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

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

fn random_stream<A>(g: Gen<A>, rng: RNG) -> Unfold<RNG, Box<dyn FnMut(&mut RNG) -> Option<A>>>
where
  A: Clone + 'static, {
  itertools::unfold(
    rng,
    Box::new(move |rng| {
      let (a, s) = g.clone().run(rng.clone());
      *rng = s;
      Some(a)
    }),
  )
}

pub fn for_all_sgen<A, F, FF>(sgen: SGen<A>, mut f: FF) -> Prop
where
  F: FnMut(A) -> bool + 'static,
  FF: FnMut() -> F + 'static,
  A: Clone + Debug + 'static, {
  match sgen {
    SGen::Unsized(g) => for_all_gen(g, f()),
    s @ SGen::Sized(..) => for_all_gen_for_size(move |i| s.run(Some(i)), f),
  }
}

pub fn for_all_gen_for_size<A, GF, F, FF>(gf: GF, mut f: FF) -> Prop
where
  GF: Fn(u32) -> Gen<A> + 'static,
  F: FnMut(A) -> bool + 'static,
  FF: FnMut() -> F + 'static,
  A: Clone + Debug + 'static, {
  Prop {
    run_f: Rc::new(RefCell::new(move |max, n, rng| {
      let cases_per_size = n / max;
      let props = itertools::iterate(0, |i| *i + 1)
        .map(|i| for_all_gen(gf(i), f()))
        .take(max as usize)
        .collect::<Vec<_>>();
      let p = props
        .into_iter()
        .map(|p| Prop::new(move |max, _, rng| p.run(max, cases_per_size, rng)))
        .reduce(|l, r| l.and(r))
        .unwrap();
      p.run(max, n, rng)
    })),
  }
}

pub fn for_all_gen<A, F>(g: Gen<A>, mut f: F) -> Prop
where
  F: FnMut(A) -> bool + 'static,
  A: Clone + Debug + 'static, {
  Prop {
    run_f: Rc::new(RefCell::new(move |_, n, rng| {
      let nl = itertools::iterate(1, |&i| i + 1).into_iter();
      random_stream(g.clone(), rng)
        .zip(nl)
        .take(n as usize)
        .map(|(a, i): (A, u32)| {
          if f(a.clone()) {
            PropResult::Passed
          } else {
            PropResult::Falsified {
              failure: format!("{:?}", a),
              successes: i,
            }
          }
        })
        .find(move |e| e.is_falsified())
        .unwrap_or(PropResult::Passed)
    })),
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
  run_f: Rc<RefCell<dyn FnMut(MaxSize, TestCases, RNG) -> PropResult>>,
}

impl Clone for Prop {
  fn clone(&self) -> Self {
    Self {
      run_f: self.run_f.clone(),
    }
  }
}

impl Prop {
  pub fn new<F>(f: F) -> Prop
  where
    F: Fn(MaxSize, TestCases, RNG) -> PropResult + 'static, {
    Prop {
      run_f: Rc::new(RefCell::new(f)),
    }
  }

  pub fn run(&self, max_size: MaxSize, test_cases: TestCases, rng: RNG) -> PropResult {
    let mut f = self.run_f.borrow_mut();
    f(max_size, test_cases, rng)
  }

  pub fn tag(self, msg: String) -> Prop {
    Prop::new(move |max, n, rng| match self.run(max, n, rng) {
      PropResult::Falsified {
        failure: e,
        successes: c,
      } => PropResult::Falsified {
        failure: format!("{}\n{}", msg, e),
        successes: c,
      },
      x => x,
    })
  }

  pub fn and(self, p: Self) -> Prop {
    Self::new(
      move |max: MaxSize, n: TestCases, rng: RNG| match self.run(max, n, rng.clone()) {
        PropResult::Passed | PropResult::Proved => p.run(max, n, rng),
        x => x,
      },
    )
  }

  pub fn or(self, p: Self) -> Prop {
    Self::new(move |max, n, rng: RNG| match self.run(max, n, rng.clone()) {
      PropResult::Falsified { failure: msg, .. } => p.clone().tag(msg).run(max, n, rng),
      x => x,
    })
  }
}

#[cfg(test)]
mod tests {

  use crate::gen::Gens;

  use super::*;
  use anyhow::Result;
  use std::env;

  #[ctor::ctor]
  fn init() {
    env::set_var("RUST_LOG", "info");
    let _ = env_logger::builder().is_test(true).try_init();
  }

  fn new_rng() -> RNG {
    RNG::new()
  }

  #[test]
  fn test_one_of() -> Result<()> {
    let gen = Gens::one_of_values(['a', 'b', 'c', 'x', 'y', 'z']);
    let prop = for_all_gen(gen, move |a| {
      log::info!("value = {}", a);
      true
    });
    test_with_prop(prop, 1, 100, new_rng())
  }

  #[test]
  fn test_one_of_2() -> Result<()> {
    let mut counter = 0;
    let gen = Gens::one_of_values(['a', 'b', 'c', 'x', 'y', 'z']);
    let prop = for_all_gen_for_size(
      move |size| Gens::list_of_n(size as usize, gen.clone()),
      move || {
        move |a| {
          counter += 1;
          log::info!("value = {},{:?}", counter, a);
          true
        }
      },
    );
    test_with_prop(prop, 10, 100, new_rng())
  }
}
