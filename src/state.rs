use std::rc::Rc;

pub struct State<S, A> {
  pub(crate) run_f: Rc<dyn Fn(S) -> (A, S)>,
}

impl<S: 'static, A: Clone + 'static> Clone for State<S, A> {
  fn clone(&self) -> Self {
    Self {
      run_f: self.run_f.clone(),
    }
  }
}



impl<S: 'static, A: Clone + 'static> State<S, A> {
  pub fn unit(a: A) -> State<S, A> {
    Self::new(move |s| (a.clone(), s))
  }

  pub fn new<T, B, F>(f: F) -> State<T, B>
  where
    F: Fn(T) -> (B, T) + 'static, {
    State { run_f: Rc::new(f) }
  }

  pub fn pure<B>(b: B) -> State<S, B>
    where
        B: Clone + 'static, {
    Self::new(move |s| (b.clone(), s))
  }

  pub fn run(self, s: S) -> (A, S) {
    (self.run_f)(s)
  }

  pub fn map<B, F>(self, f: F) -> State<S, B>
  where
    F: Fn(A) -> B + 'static,
    B: Clone + 'static, {
    self.flat_map(move |a| Self::pure(f(a)))
  }

  pub fn and_then<B>(self, sb: State<S, B>) -> State<S, (A, B)>
  where
    A: Clone,
    B: Clone + 'static, {
    self.flat_map(move |a| sb.clone().flat_map(move |b| Self::pure((a.clone(), b))))
  }

  pub fn flat_map<B, F>(self, f: F) -> State<S, B>
  where
    F: Fn(A) -> State<S, B> + 'static,
    B: Clone + 'static, {
    State::<S, B>::new(move |s| {
      let (a, s1) = self.clone().run(s);
      f(a).run(s1)
    })
  }

  pub fn modify<T, F>(f: F) -> State<T, ()>
  where
    F: Fn(T) -> T + 'static,
    T: Clone + 'static, {
    let s = Self::get();
    s.flat_map(move |t: T| Self::set(f(t)))
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
      let actions = sas.clone();
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
  A: Default + Clone + 'static,
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
    let s = State::<u32, u32>::pure(10);
    let r = s.run(10);
    println!("{:?}", r);
  }
}
