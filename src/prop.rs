use crate::gen::{Gen, SGen};
use crate::rng::RNG;

use anyhow::*;
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

pub type MaxSize = u32;
pub type TestCases = u32;
pub type FailedCase = String;
pub type SuccessCount = u32;

/// The trait to return the failure of the property.
pub trait IsFalsified {
  fn is_falsified(&self) -> bool;
  fn non_falsified(&self) -> bool {
    !self.is_falsified()
  }
}

/// Represents the result of the property.
#[derive(Clone)]
pub enum PropResult {
  /// The property is passed.
  Passed {
    /// The number of test cases.
    test_cases: TestCases,
  },
  /// The property is falsified.
  Falsified {
    /// The failure of the property.
    failure: FailedCase,
    /// The number of successes.
    successes: SuccessCount,
  },
  Proved,
}

impl PropResult {
  /// The `map` method can change the number of test cases.
  ///
  /// # Arguments
  /// - `f` - The function to change the number of test cases.
  ///
  /// # Returns
  /// - `PropResult` - The new PropResult.
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

  /// The `flat_map` method can change the number of test cases.
  ///
  /// # Arguments
  /// - `f` - The function to change the number of test cases.
  ///
  /// # Returns
  /// - `PropResult` - The new PropResult.
  pub fn flat_map<F>(self, f: F) -> PropResult
  where
    F: FnOnce(Option<u32>) -> PropResult, {
    match self {
      PropResult::Passed { test_cases } => f(Some(test_cases)),
      PropResult::Proved => f(None),
      other => other,
    }
  }

  /// The `to_result` method can convert the PropResult to Result.
  ///
  /// # Returns
  /// - `Result<String>` - The result of the PropResult.
  pub fn to_result(self) -> Result<String> {
    match self {
      p @ PropResult::Passed { .. } => Ok(p.message()),
      p @ PropResult::Proved => Ok(p.message()),
      f @ PropResult::Falsified { .. } => Err(anyhow!(f.message())),
    }
  }

  /// The `to_result_unit` method can convert the PropResult to Result with the message.
  ///
  /// # Returns
  /// - `Result<()>` - The result without the message of the PropResult
  pub fn to_result_unit(self) -> Result<()> {
    self
      .to_result()
      .map(|msg| {
        log::info!("{}", msg);
        ()
      })
      .map_err(|err| {
        log::error!("{}", err);
        err
      })
  }

  /// The `message` method can return the message of the PropResult.
  ///
  /// # Returns
  /// - `String` - The message of the PropResult.
  pub fn message(&self) -> String {
    match self {
      PropResult::Passed { test_cases } => format!("OK, passed {} tests", test_cases),
      PropResult::Proved => "OK, proved property".to_string(),
      PropResult::Falsified { failure, successes } => {
        format!("Falsified after {} passed tests: {}", successes, failure)
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

fn random_stream<A>(g: Gen<A>, mut rng: RNG) -> impl Iterator<Item = A>
where
  A: Clone + 'static, {
  std::iter::from_fn(move || {
    let (a, new_rng) = g.clone().run(rng.clone());
    rng = new_rng;
    Some(a)
  })
}

/// Returns a Prop that executes a function to evaluate properties using SGen.
///
/// # Arguments
/// - `sgen` - The SGen.
/// - `test` - The function to evaluate the properties.
///
/// # Returns
/// - `Prop` - The new Prop.
pub fn for_all_sgen<A, F, FF>(sgen: SGen<A>, mut test: FF) -> Prop
where
  F: FnMut(A) -> bool + 'static,
  FF: FnMut() -> F + 'static,
  A: Clone + Debug + 'static, {
  match sgen {
    SGen::Unsized(g) => for_all_gen(g, test()),
    s @ SGen::Sized(..) => for_all_gen_for_size(move |i| s.run(Some(i)), test),
  }
}

/// Returns a Prop that executes a function to evaluate properties using Gen with size.
///
/// # Arguments
/// - `gf` - The function to create a Gen with size.
/// - `f` - The function to evaluate the properties.
///
/// # Returns
/// - `Prop` - The new Prop.
pub fn for_all_gen_for_size<A, GF, F, FF>(gf: GF, mut test: FF) -> Prop
where
  GF: Fn(u32) -> Gen<A> + 'static,
  F: FnMut(A) -> bool + 'static,
  FF: FnMut() -> F + 'static,
  A: Clone + Debug + 'static, {
  Prop {
    run_f: Rc::new(RefCell::new(move |max, n, rng| {
      let cases_per_size = n / max;

      // Create a Vec with pre-allocated capacity
      let mut props = Vec::with_capacity(max as usize);

      // Replace iterator chains with a simple loop
      for i in 0..max {
        props.push(for_all_gen(gf(i), test()));
      }

      // Early return if empty
      if props.is_empty() {
        return PropResult::Passed { test_cases: 0 };
      }

      // Clone and get the first Prop
      let first_prop = props[0].clone();
      let mut result_prop = Prop::new(move |max, _, rng| first_prop.run(max, cases_per_size, rng));

      // Process the remaining Props
      for i in 1..props.len() {
        let p = props[i].clone();
        let prop = Prop::new(move |max, _, rng| p.run(max, cases_per_size, rng));
        result_prop = result_prop.and(prop);
      }

      // Execute the final result
      match result_prop.run(max, n, rng) {
        _ => PropResult::Proved,
      }
    })),
  }
}

/// Returns a Prop that executes a function to evaluate properties using Gen.
///
/// # Arguments
/// - `g` - The Gen.
/// - `test` - The function to evaluate the properties.
///
/// # Returns
/// - `Prop` - The new Prop.
pub fn for_all_gen<A, F>(g: Gen<A>, mut test: F) -> Prop
where
  F: FnMut(A) -> bool + 'static,
  A: Clone + Debug + 'static, {
  Prop {
    run_f: Rc::new(RefCell::new(move |_, n, mut rng| {
      // Replace iterator chains with a simple loop
      let mut success_count = 1;

      for _ in 0..n {
        // Generate test value
        let (test_value, new_rng) = g.clone().run(rng);
        rng = new_rng;

        // Execute test
        if !test(test_value.clone()) {
          return PropResult::Falsified {
            failure: format!("{:?}", test_value),
            successes: success_count,
          };
        }

        success_count += 1;
      }

      // If all tests pass
      PropResult::Passed { test_cases: n }
    })),
  }
}

/// Execute the Prop.
///
/// # Arguments
/// - `max_size` - The maximum size of the generated value.
/// - `test_cases` - The number of test cases.
/// - `rng` - The random number generator.
///
/// # Returns
/// - `Result<String>` - The result of the Prop.
pub fn run_with_prop(p: Prop, max_size: MaxSize, test_cases: TestCases, rng: RNG) -> Result<String> {
  p.run(max_size, test_cases, rng).to_result()
}

/// Execute the Prop.
///
/// # Arguments
/// - `max_size` - The maximum size of the generated value.
/// - `test_cases` - The number of test cases.
/// - `rng` - The random number generator.
///
/// # Returns
/// - `Result<()>` - The result of the Prop.
pub fn test_with_prop(p: Prop, max_size: MaxSize, test_cases: TestCases, rng: RNG) -> Result<()> {
  p.run(max_size, test_cases, rng).to_result_unit()
}

/// Represents the property.
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
  /// Create a new Prop.
  ///
  /// # Arguments
  /// - `f` - The function to evaluate the properties.
  ///
  /// # Returns
  /// - `Prop` - The new Prop.
  pub fn new<F>(f: F) -> Prop
  where
    F: Fn(MaxSize, TestCases, RNG) -> PropResult + 'static, {
    Prop {
      run_f: Rc::new(RefCell::new(f)),
    }
  }

  /// Execute the Prop.
  ///
  /// # Arguments
  /// - `max_size` - The maximum size of the generated value.
  /// - `test_cases` - The number of test cases.
  /// - `rng` - The random number generator.
  ///
  /// # Returns
  /// - `PropResult` - The result of the Prop.
  pub fn run(&self, max_size: MaxSize, test_cases: TestCases, rng: RNG) -> PropResult {
    let mut f = self.run_f.borrow_mut();
    f(max_size, test_cases, rng)
  }

  /// The `tag` method can add a message to the PropResult.
  ///
  /// # Arguments
  /// - `msg` - The message.
  ///
  /// # Returns
  /// - `Prop` - The tagged Prop.
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

  /// The `and` method can combine a other Prop.
  /// If the first Prop is passed, the second Prop is executed.
  ///
  /// # Arguments
  /// - `other` - The other Prop.
  ///
  /// # Returns
  /// - `Prop` - The combined Prop.
  pub fn and(self, other: Self) -> Prop {
    Self::new(
      move |max: MaxSize, n: TestCases, rng: RNG| match self.run(max, n, rng.clone()) {
        PropResult::Passed { .. } | PropResult::Proved => other.run(max, n, rng),
        x => x,
      },
    )
  }

  /// The 'or' method can combine a other Prop.
  /// If the first Prop is falsified, the second Prop is executed.
  ///
  /// # Arguments
  /// - `other` - The other Prop.
  ///
  /// # Returns
  /// - `Prop` - The combined Prop.
  pub fn or(self, other: Self) -> Prop {
    Self::new(move |max, n, rng: RNG| match self.run(max, n, rng.clone()) {
      PropResult::Falsified { failure: msg, .. } => other.clone().tag(msg).run(max, n, rng),
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

  fn init() {
    env::set_var("RUST_LOG", "info");
    let _ = env_logger::builder().is_test(true).try_init();
  }

  fn new_rng() -> RNG {
    RNG::new()
  }

  #[test]
  fn test_one_of() -> Result<()> {
    init();
    let gen = Gens::one_of_values(['a', 'b', 'c', 'x', 'y', 'z']);
    let prop = for_all_gen(gen, move |a| {
      log::info!("value = {}", a);
      true
    });
    test_with_prop(prop, 1, 100, new_rng())
  }

  #[test]
  fn test_one_of_2() -> Result<()> {
    init();
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

  #[test]
  fn test_prop_result_passed() {
    init();
    let result = PropResult::Passed { test_cases: 100 };
    assert!(!result.is_falsified());
    assert!(result.non_falsified());
    assert_eq!(result.message(), "OK, passed 100 tests");
  }

  #[test]
  fn test_prop_result_proved() {
    init();
    let result = PropResult::Proved;
    assert!(!result.is_falsified());
    assert!(result.non_falsified());
    assert_eq!(result.message(), "OK, proved property");
  }

  #[test]
  fn test_prop_result_falsified() {
    init();
    let result = PropResult::Falsified {
      failure: "test failed".to_string(),
      successes: 42,
    };
    assert!(result.is_falsified());
    assert!(!result.non_falsified());
    assert_eq!(result.message(), "Falsified after 42 passed tests: test failed");
  }

  #[test]
  fn test_prop_result_map() {
    init();
    let result = PropResult::Passed { test_cases: 100 };
    let mapped = result.map(|n| n * 2);
    match mapped {
      PropResult::Passed { test_cases } => assert_eq!(test_cases, 200),
      _ => panic!("Expected PropResult::Passed"),
    }
  }

  #[test]
  fn test_prop_result_flat_map() {
    init();
    let result = PropResult::Passed { test_cases: 100 };
    let flat_mapped = result.flat_map(|n| PropResult::Passed {
      test_cases: n.unwrap() * 2,
    });
    match flat_mapped {
      PropResult::Passed { test_cases } => assert_eq!(test_cases, 200),
      _ => panic!("Expected PropResult::Passed"),
    }
  }

  #[test]
  fn test_prop_result_to_result() {
    init();
    let passed = PropResult::Passed { test_cases: 100 };
    let passed_result = passed.to_result();
    assert!(passed_result.is_ok());
    assert_eq!(passed_result.unwrap(), "OK, passed 100 tests");

    let falsified = PropResult::Falsified {
      failure: "test failed".to_string(),
      successes: 42,
    };
    let falsified_result = falsified.to_result();
    assert!(falsified_result.is_err());
  }

  #[test]
  fn test_for_all_gen() {
    init();
    // Property that always succeeds
    let gen = Gens::choose_i32(1, 100);
    let prop = for_all_gen(gen.clone(), |_| true);
    let result = prop.run(1, 10, new_rng());
    match result {
      PropResult::Passed { test_cases } => assert_eq!(test_cases, 10),
      _ => panic!("Expected PropResult::Passed"),
    }

    // Property that always fails
    let prop = for_all_gen(gen, |_| false);
    let result = prop.run(1, 10, new_rng());
    assert!(result.is_falsified());
  }

  #[test]
  fn test_for_all_sgen() {
    init();
    // Property that always succeeds
    let gen = Gens::choose_i32(1, 100);
    let sgen = crate::gen::SGen::Unsized(gen.clone());
    let prop = for_all_sgen(sgen, || |_| true);
    let result = prop.run(1, 10, new_rng());
    match result {
      PropResult::Passed { test_cases } => assert_eq!(test_cases, 10),
      _ => panic!("Expected PropResult::Passed"),
    }
  }

  #[test]
  fn test_prop_and() {
    init();
    // Properties that both succeed
    let gen1 = Gens::choose_i32(1, 100);
    let gen2 = Gens::choose_i32(1, 100);
    let prop1 = for_all_gen(gen1, |_| true);
    let prop2 = for_all_gen(gen2, |_| true);
    let combined = prop1.and(prop2);
    let result = combined.run(1, 10, new_rng());
    match result {
      PropResult::Passed { test_cases } => assert_eq!(test_cases, 10),
      _ => panic!("Expected PropResult::Passed"),
    }

    // Property where the first one fails
    let gen1 = Gens::choose_i32(1, 100);
    let gen2 = Gens::choose_i32(1, 100);
    let prop1 = for_all_gen(gen1, |_| false);
    let prop2 = for_all_gen(gen2, |_| true);
    let combined = prop1.and(prop2);
    let result = combined.run(1, 10, new_rng());
    assert!(result.is_falsified());
  }

  #[test]
  fn test_prop_or() {
    init();
    // Properties that both succeed
    let gen1 = Gens::choose_i32(1, 100);
    let gen2 = Gens::choose_i32(1, 100);
    let prop1 = for_all_gen(gen1, |_| true);
    let prop2 = for_all_gen(gen2, |_| true);
    let combined = prop1.or(prop2);
    let result = combined.run(1, 10, new_rng());
    match result {
      PropResult::Passed { test_cases } => assert_eq!(test_cases, 10),
      _ => panic!("Expected PropResult::Passed"),
    }

    // Property where the first one fails
    let gen1 = Gens::choose_i32(1, 100);
    let gen2 = Gens::choose_i32(1, 100);
    let prop1 = for_all_gen(gen1, |_| false);
    let prop2 = for_all_gen(gen2, |_| true);
    let combined = prop1.or(prop2);
    let result = combined.run(1, 10, new_rng());
    match result {
      PropResult::Passed { test_cases } => assert_eq!(test_cases, 10),
      _ => panic!("Expected PropResult::Passed"),
    }

    // Properties that both fail
    let gen1 = Gens::choose_i32(1, 100);
    let gen2 = Gens::choose_i32(1, 100);
    let prop1 = for_all_gen(gen1, |_| false);
    let prop2 = for_all_gen(gen2, |_| false);
    let combined = prop1.or(prop2);
    let result = combined.run(1, 10, new_rng());
    assert!(result.is_falsified());
  }

  #[test]
  fn test_prop_tag() {
    init();
    let gen = Gens::choose_i32(1, 100);
    let prop = for_all_gen(gen, |_| false);
    let tagged = prop.tag("Custom error message".to_string());
    let result = tagged.run(1, 10, new_rng());
    match result {
      PropResult::Falsified { failure, .. } => {
        assert!(failure.contains("Custom error message"));
      }
      _ => panic!("Expected PropResult::Falsified"),
    }
  }

  #[test]
  fn test_run_with_prop() -> Result<()> {
    init();
    let gen = Gens::choose_i32(1, 100);
    let prop = for_all_gen(gen, |_| true);
    let result = run_with_prop(prop, 1, 10, new_rng());
    assert!(result.is_ok());
    Ok(())
  }
}
