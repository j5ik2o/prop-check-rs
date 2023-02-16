use crate::gen::{Gen, SGen};
use crate::rng::PropRng;

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
  fn non_falsified(&self) -> bool {
    !self.is_falsified()
  }
}

#[derive(Clone)]
pub enum PropResult {
  Passed {
    test_cases: TestCases,
  },
  Falsified {
    failure: FailedCase,
    successes: SuccessCount,
  },
  Proved,
}

impl PropResult {
  pub fn map<F>(self, f: F) -> PropResult
  where
    F: FnOnce(u32) -> u32, {
    match self {
      PropResult::Passed { test_cases } => PropResult::Passed {
        test_cases: f(test_cases),
      },
      PropResult::Proved => PropResult::Proved,
      other => other,
    }
  }

  pub fn flat_map<F>(self, f: F) -> PropResult
  where
    F: FnOnce(Option<u32>) -> PropResult, {
    match self {
      PropResult::Passed { test_cases } => f(Some(test_cases)),
      PropResult::Proved => f(None),
      other => other,
    }
  }

  pub fn into_result(self) -> Result<String> {
    match self {
      p @ PropResult::Passed { .. } => Ok(p.message()),
      p @ PropResult::Proved => Ok(p.message()),
      f @ PropResult::Falsified { .. } => Err(anyhow!(f.message())),
    }
  }

  pub fn into_result_unit(self) -> Result<()> {
    self
      .into_result()
      .map(|msg| {
        log::info!("{}", msg);
      })
      .map_err(|err| {
        log::error!("{}", err);
        err
      })
  }

  pub fn message(&self) -> String {
    match self {
      PropResult::Passed { test_cases } => format!("OK, passed {} tests", test_cases),
      PropResult::Proved => "OK, proved property".to_string(),
      PropResult::Falsified { failure, successes } => {
        format!("Falsified after {} passed tests: {}", failure, successes)
      }
    }
  }
}

impl IsFalsified for PropResult {
  fn is_falsified(&self) -> bool {
    match self {
      PropResult::Passed { .. } => false,
      PropResult::Falsified { .. } => true,
      PropResult::Proved => false,
    }
  }
}

type DynOptFn<T, A> = dyn FnMut(&mut T) -> Option<A>;
fn random_stream<T: PropRng, A>(g: Gen<T, A>, rng: T) -> Unfold<T, Box<DynOptFn<T, A>>>
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

/// Represents the function to evaluate the properties by using SGens.
pub fn for_all_sgen<T: PropRng, A, F, FF>(sgen: SGen<T, A>, mut f: FF) -> Prop<T>
where
  F: FnMut(A) -> bool + 'static,
  FF: FnMut() -> F + 'static,
  A: Clone + Debug + 'static, {
  match sgen {
    SGen::Unsized(g) => for_all_gen(g, f()),
    s @ SGen::Sized(..) => for_all_gen_for_size(move |i| s.run(Some(i)), f),
  }
}

/// Represents the function to evaluate the properties by using Gens.
pub fn for_all_gen_for_size<T: PropRng + 'static, A, GF, F, FF>(gf: GF, mut f: FF) -> Prop<T>
where
  GF: Fn(u32) -> Gen<T, A> + 'static,
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
      p.run(max, n, rng).flat_map(|_| PropResult::Proved)
    })),
  }
}

/// Represents the function to evaluate the properties by using Gens.
pub fn for_all_gen<T: PropRng, A, F>(g: Gen<T, A>, mut test: F) -> Prop<T>
where
  F: FnMut(A) -> bool + 'static,
  A: Clone + Debug + 'static, {
  Prop {
    run_f: Rc::new(RefCell::new(move |_, n, rng| {
      let success_counter = itertools::iterate(1, |&i| i + 1);
      random_stream(g.clone(), rng)
        .zip(success_counter)
        .take(n as usize)
        .map(|(test_value, success_count)| {
          if test(test_value.clone()) {
            PropResult::Passed { test_cases: n }
          } else {
            PropResult::Falsified {
              failure: format!("{:?}", test_value),
              successes: success_count,
            }
          }
        })
        .find(move |e| e.is_falsified())
        .unwrap_or(PropResult::Passed { test_cases: n })
    })),
  }
}

/// Execute the Prop.
///
/// # Arguments
///
/// * `max_size` - The maximum size of the generated value.
/// * `test_cases` - The number of test cases.
/// * `rng` - The random number generator.
pub fn run_with_prop<T: PropRng>(p: Prop<T>, max_size: MaxSize, test_cases: TestCases, rng: T) -> Result<String> {
  p.run(max_size, test_cases, rng).into_result()
}

/// Test the Prop.
///
/// # Arguments
///
/// * `max_size` - The maximum size of the generated value.
/// * `test_cases` - The number of test cases.
/// * `rng` - The random number generator.
pub fn test_with_prop<T: PropRng>(p: Prop<T>, max_size: MaxSize, test_cases: TestCases, rng: T) -> Result<()> {
  p.run(max_size, test_cases, rng).into_result_unit()
}

type DynProp<T> = dyn FnMut(MaxSize, TestCases, T) -> PropResult;
pub struct Prop<T: PropRng> {
  run_f: Rc<RefCell<DynProp<T>>>,
}

impl<T: PropRng> Clone for Prop<T> {
  fn clone(&self) -> Self {
    Self {
      run_f: self.run_f.clone(),
    }
  }
}

impl<T: PropRng> Prop<T> {
  pub fn new<F>(f: F) -> Prop<T>
  where
    F: Fn(MaxSize, TestCases, T) -> PropResult + 'static, {
    Prop {
      run_f: Rc::new(RefCell::new(f)),
    }
  }

  pub fn run(&self, max_size: MaxSize, test_cases: TestCases, rng: T) -> PropResult {
    let mut f = self.run_f.borrow_mut();
    f(max_size, test_cases, rng)
  }

  pub fn tag(self, msg: String) -> Prop<T> {
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

  pub fn and(self, other: Self) -> Prop<T> {
    Self::new(
      move |max: MaxSize, n: TestCases, rng: T| match self.run(max, n, rng.clone()) {
        PropResult::Passed { .. } | PropResult::Proved => other.run(max, n, rng),
        x => x,
      },
    )
  }

  pub fn or(self, p: Self) -> Prop<T> {
    Self::new(move |max, n, rng: T| match self.run(max, n, rng.clone()) {
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
  use crate::rand::thread_rng;
  use std::env;

  #[ctor::ctor]
  fn init() {
    env::set_var("RUST_LOG", "info");
    let _ = env_logger::builder().is_test(true).try_init();
  }

  #[test]
  fn test_one_of() -> Result<()> {
    let gen = Gens::one_of_values(['a', 'b', 'c', 'x', 'y', 'z']);
    let prop = for_all_gen(gen, move |a| {
      log::info!("value = {}", a);
      true
    });
    test_with_prop(prop, 1, 100, thread_rng())
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
    test_with_prop(prop, 10, 100, thread_rng())
  }
}
