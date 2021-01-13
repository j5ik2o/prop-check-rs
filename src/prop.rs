use crate::prop::Result::Falsified;
use crate::rng::RNG;
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
  pub fn new<B>(b: State<RNG, B>) -> Gen<B> {
    Gen { sample: b }
  }
  // pub fn map<'b, B: 'b, F>(self, f: F) -> Gen<'b, B> where F: Fn(A) -> B + 'b, B: Clone {
  //   Self::new(self.sample.fmap(f))
  // }
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
