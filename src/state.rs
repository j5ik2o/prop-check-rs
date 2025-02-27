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
  /// Create a new State with a value.
  pub fn value(a: A) -> State<S, A> {
    Self::new(move |s| (a.clone(), s))
  }

  /// Create a new State.
  pub fn new<T, B, F>(f: F) -> State<T, B>
  where
    F: Fn(T) -> (B, T) + 'static, {
    State { run_f: Rc::new(f) }
  }

  /// Alias for Self::value.
  pub fn pure<B>(b: B) -> State<S, B>
  where
    B: Clone + 'static, {
    Self::new(move |s| (b.clone(), s))
  }

  /// Run the State.
  pub fn run(self, s: S) -> (A, S) {
    (self.run_f)(s)
  }

  /// Map the State.
  pub fn map<B, F>(self, f: F) -> State<S, B>
  where
    F: Fn(A) -> B + 'static,
    B: Clone + 'static, {
    self.flat_map(move |a| Self::pure(f(a)))
  }

  /// FlatMap the State.
  pub fn flat_map<B, F>(self, f: F) -> State<S, B>
  where
    F: Fn(A) -> State<S, B> + 'static,
    B: Clone + 'static, {
    State::<S, B>::new(move |s| {
      let (a, s1) = self.clone().run(s);
      f(a).run(s1)
    })
  }

  /// Compose two States.
  pub fn and_then<B>(self, sb: State<S, B>) -> State<S, (A, B)>
  where
    A: Clone,
    B: Clone + 'static, {
    self.flat_map(move |a| sb.clone().flat_map(move |b| Self::pure((a.clone(), b))))
  }

  /// Get the state.
  pub fn get<T>() -> State<T, T>
  where
    T: Clone, {
    Self::new(move |t: T| (t.clone(), t))
  }

  /// Set the state.
  pub fn set<T>(t: T) -> State<T, ()>
  where
    T: Clone + 'static, {
    Self::new(move |_| ((), t.clone()))
  }

  /// Modify the state by applying a function.
  pub fn modify<T, F>(f: F) -> State<T, ()>
  where
    F: Fn(T) -> T + 'static,
    T: Clone + 'static, {
    let s = Self::get();
    s.flat_map(move |t: T| Self::set(f(t)))
  }

  /// Execute the State and return a State with the result in the collection.
  pub fn sequence(sas: Vec<State<S, A>>) -> State<S, Vec<A>> {
    Self::new(move |s| {
      let mut s_ = s;
      // 事前に容量を確保
      let mut acc = Vec::with_capacity(sas.len());

      // 所有権を移動せずにイテレート
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
