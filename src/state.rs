use std::rc::Rc;

pub struct State<S, A> {
  pub(crate) run_f: Box<dyn FnOnce(S) -> (A, S)>,
}

impl<S: 'static, A: 'static> State<S, A> {
  pub fn cloned(self) -> Self {
    Self::new(move |s| self.run(s))
  }

  pub fn unit(a: A) -> State<S, A> {
    Self::new(|s| (a, s))
  }

  pub fn new<T, B, F>(f: F) -> State<T, B>
  where
    F: FnOnce(T) -> (B, T) + 'static, {
    State { run_f: Box::new(f) }
  }

  pub fn run(self, s: S) -> (A, S) {
    (self.run_f)(s)
  }

  pub fn pure<T, B>(b: B) -> State<T, B>
  where
    B: Clone + 'static, {
    Self::new(move |s| (b.clone(), s))
  }

  pub fn fmap<B, F>(self, f: F) -> State<S, B>
  where
    F: FnOnce(A) -> B + 'static,
    B: Clone + 'static, {
    self.bind(move |a| Self::pure(f(a)))
  }

  pub fn fmap2<B, C, F>(self, sb: State<S, B>, f: F) -> State<S, C>
  where
    F: FnOnce(A, B) -> C + 'static,
    A: Clone,
    B: Clone + 'static,
    C: Clone + 'static, {
    self.bind(move |a| sb.fmap(move |b| f(a.clone(), b.clone())))
  }

  pub fn bind<B, F>(self, f: F) -> State<S, B>
  where
    F: FnOnce(A) -> State<S, B> + 'static,
    B: Clone + 'static, {
    Self::new(move |s| {
      let (a, s1) = self.run(s);
      f(a).run(s1)
    })
  }

  pub fn modify<T, F>(f: F) -> State<T, ()>
  where
    F: FnOnce(T) -> T + 'static,
    T: Clone + 'static, {
    let s = Self::get();
    s.bind(move |t: T| Self::set(f(t)))
  }

  pub fn get<T>() -> State<T, T>
  where
    T: Clone, {
    Self::new(move |t: T| (t.clone(), t))
  }

  pub fn set<T>(t: T) -> State<T, ()>
  where
    T: Clone + 'static, {
    Self::new(move |_| ((), t.clone()))
  }

  pub fn sequence(sas: Vec<State<S, A>>) -> State<S, Vec<A>> {
    Self::new(move |s| {
      let mut s_ = s;
      let actions = sas;
      let mut acc: Vec<A> = vec![];
      for x in actions.into_iter() {
        let (a, s2) = x.run(s_);
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
  A: Default + 'static,
{
  fn default() -> Self {
    Self::new(|_| (A::default(), S::default()))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn state() {
    let s = State::<i32, i32>::pure(10);
    let r = s.cloned().run(10);
    println!("{:?}", r);
  }
}
