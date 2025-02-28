use std::fmt::{Debug, Formatter};
use std::rc::Rc;

/// The `State` monad represents a stateful computation.
pub struct State<S, A> {
  pub(crate) run_f: Rc<dyn Fn(S) -> (A, S)>,
}

impl<S, A> Debug for State<S, A> {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    write!(f, "Fn")
  }
}

impl<S, A> Clone for State<S, A>
where
  S: 'static,
  A: Clone + 'static,
{
  fn clone(&self) -> Self {
    Self {
      run_f: self.run_f.clone(),
    }
  }
}

impl<S, A> State<S, A>
where
  S: 'static,
  A: Clone + 'static,
{
  /// Creates a new State with a constant value.
  ///
  /// This method creates a State that, when run, will return the provided value
  /// and the unchanged state.
  ///
  /// # Arguments
  /// - `a` - The value to be returned by the State
  ///
  /// # Returns
  /// - `State<S, A>` - A new State that returns the provided value
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::state::State;
  /// let state = State::<i32, String>::value("hello".to_string());
  /// let (value, new_state) = state.run(42);
  /// assert_eq!(value, "hello");
  /// assert_eq!(new_state, 42); // State is unchanged
  /// ```
  pub fn value(a: A) -> State<S, A> {
    Self::new(move |s| (a.clone(), s))
  }

  /// Creates a new State with a custom state transformation function.
  ///
  /// This is the core constructor for State, allowing you to define exactly how
  /// the state should be transformed and what value should be produced.
  ///
  /// # Arguments
  /// - `f` - A function that takes a state and returns a tuple of (value, new_state)
  ///
  /// # Returns
  /// - `State<T, B>` - A new State that applies the provided function
  ///
  /// # Type Parameters
  /// - `T` - The type of the state
  /// - `B` - The type of the value produced
  /// - `F` - The type of the function
  pub fn new<T, B, F>(f: F) -> State<T, B>
  where
    F: Fn(T) -> (B, T) + 'static, {
    State { run_f: Rc::new(f) }
  }

  /// Creates a new State with a constant value (alias for value).
  ///
  /// This method is functionally identical to `value` but is named to align with
  /// the functional programming concept of "pure" or "return".
  ///
  /// # Arguments
  /// - `b` - The value to be returned by the State
  ///
  /// # Returns
  /// - `State<S, B>` - A new State that returns the provided value
  ///
  /// # Type Parameters
  /// - `B` - The type of the value to be returned
  pub fn pure<B>(b: B) -> State<S, B>
  where
    B: Clone + 'static, {
    Self::new(move |s| (b.clone(), s))
  }

  /// Executes the State with the provided initial state.
  ///
  /// This method runs the State computation with the given state and returns
  /// both the resulting value and the final state.
  ///
  /// # Arguments
  /// - `s` - The initial state
  ///
  /// # Returns
  /// - `(A, S)` - A tuple containing the resulting value and the final state
  pub fn run(self, s: S) -> (A, S) {
    (self.run_f)(s)
  }

  /// Transforms the value produced by this State using a function.
  ///
  /// This method allows you to transform the value produced by a State without
  /// affecting how the state itself is transformed.
  ///
  /// # Arguments
  /// - `f` - A function that transforms the value
  ///
  /// # Returns
  /// - `State<S, B>` - A new State that produces the transformed value
  ///
  /// # Type Parameters
  /// - `B` - The type of the transformed value
  /// - `F` - The type of the transformation function
  pub fn map<B, F>(self, f: F) -> State<S, B>
  where
    F: Fn(A) -> B + 'static,
    B: Clone + 'static, {
    self.flat_map(move |a| Self::pure(f(a)))
  }

  /// Chains this State with a function that returns another State.
  ///
  /// This method allows for sequential composition of stateful computations.
  /// The function `f` is applied to the value produced by this State, and the
  /// resulting State is then run with the updated state.
  ///
  /// # Arguments
  /// - `f` - A function that takes the value from this State and returns a new State
  ///
  /// # Returns
  /// - `State<S, B>` - A new State representing the sequential composition
  ///
  /// # Type Parameters
  /// - `B` - The type of the value produced by the resulting State
  /// - `F` - The type of the function
  pub fn flat_map<B, F>(self, f: F) -> State<S, B>
  where
    F: Fn(A) -> State<S, B> + 'static,
    B: Clone + 'static, {
    State::<S, B>::new(move |s| {
      let (a, s1) = self.clone().run(s);
      f(a).run(s1)
    })
  }

  /// Combines this State with another State, producing both values.
  ///
  /// This method runs this State, then runs the provided State with the
  /// updated state, and returns both values as a tuple.
  ///
  /// # Arguments
  /// - `sb` - Another State to run after this one
  ///
  /// # Returns
  /// - `State<S, (A, B)>` - A new State that produces both values as a tuple
  ///
  /// # Type Parameters
  /// - `B` - The type of the value produced by the second State
  pub fn and_then<B>(self, sb: State<S, B>) -> State<S, (A, B)>
  where
    A: Clone,
    B: Clone + 'static, {
    self.flat_map(move |a| sb.clone().flat_map(move |b| Self::pure((a.clone(), b))))
  }

  /// Creates a State that returns the current state as its value.
  ///
  /// This method is useful for accessing the current state without modifying it.
  ///
  /// # Returns
  /// - `State<T, T>` - A State that returns the current state as its value
  ///
  /// # Type Parameters
  /// - `T` - The type of the state
  pub fn get<T>() -> State<T, T>
  where
    T: Clone, {
    Self::new(move |t: T| (t.clone(), t))
  }

  /// Creates a State that replaces the current state with a new value.
  ///
  /// This method is useful for setting the state to a specific value,
  /// regardless of its current value.
  ///
  /// # Arguments
  /// - `t` - The new state value
  ///
  /// # Returns
  /// - `State<T, ()>` - A State that sets the state to the provided value
  ///
  /// # Type Parameters
  /// - `T` - The type of the state
  pub fn set<T>(t: T) -> State<T, ()>
  where
    T: Clone + 'static, {
    Self::new(move |_| ((), t.clone()))
  }

  /// Creates a State that modifies the current state using a function.
  ///
  /// This method allows you to transform the state based on its current value.
  ///
  /// # Arguments
  /// - `f` - A function that transforms the state
  ///
  /// # Returns
  /// - `State<T, ()>` - A State that modifies the state using the provided function
  ///
  /// # Type Parameters
  /// - `T` - The type of the state
  /// - `F` - The type of the transformation function
  pub fn modify<T, F>(f: F) -> State<T, ()>
  where
    F: Fn(T) -> T + 'static,
    T: Clone + 'static, {
    let s = Self::get();
    s.flat_map(move |t: T| Self::set(f(t)))
  }

  /// Executes a sequence of States and collects their results into a vector.
  ///
  /// This method runs each State in the provided vector in sequence, threading
  /// the state through each computation, and collects all the resulting values.
  ///
  /// # Arguments
  /// - `sas` - A vector of States to execute in sequence
  ///
  /// # Returns
  /// - `State<S, Vec<A>>` - A State that produces a vector of all the values
  pub fn sequence(sas: Vec<State<S, A>>) -> State<S, Vec<A>> {
    Self::new(move |s| {
      let mut s_ = s;
      // Pre-allocate capacity
      let mut acc = Vec::with_capacity(sas.len());

      // Iterate without moving ownership
      for x in sas.iter() {
        let (a, s2) = x.clone().run(s_);
        s_ = s2;
        acc.push(a);
      }
      (acc, s_)
    })
  }
}

impl<S, A> Default for State<S, A>
where
  S: Default + 'static,
  A: Default + Clone + 'static,
{
  fn default() -> Self {
    Self::new(|_| (A::default(), S::default()))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::env;

  fn init() {
    env::set_var("RUST_LOG", "info");
    let _ = env_logger::builder().is_test(true).try_init();
  }

  pub mod laws {
    use super::*;

    #[test]
    fn test_left_identity_law() {
      init();
      let n = 10;
      let s = 11;
      let f = |x| State::<i32, i32>::pure(x);
      let result = State::<i32, i32>::pure(n).flat_map(f).run(s.clone()) == f(n).run(s);
      assert!(result);
    }

    #[test]
    fn test_right_identity_law() {
      init();
      let x = 10;
      let s = 11;
      let result = State::<i32, i32>::pure(x)
        .flat_map(|y| State::<i32, i32>::pure(y))
        .run(s.clone())
        == State::<i32, i32>::pure(x).run(s);
      assert!(result);
    }

    #[test]
    fn test_associativity_law() {
      init();
      let x = 10;
      let s = 11;
      let f = |x| State::<i32, i32>::pure(x * 2);
      let g = |x| State::<i32, i32>::pure(x + 1);
      let result = State::<i32, i32>::pure(x).flat_map(f).flat_map(g).run(s.clone()) == f(x).flat_map(g).run(s);
      assert!(result);
    }
  }

  #[test]
  fn pure() {
    init();
    let s = State::<u32, u32>::pure(10);
    let r = s.run(10);
    assert_eq!(r, (10, 10));
  }

  #[test]
  // #[should_panic]
  fn should_panic_when_running_with_null_state() {
    let state: State<Option<i32>, i32> = State::<Option<i32>, i32>::new(|s| (10, s));
    let result = state.run(None);
    assert_eq!(result, (10, None));
  }
}
