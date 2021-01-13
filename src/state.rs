use std::rc::Rc;

#[derive(Clone)]
pub struct State<'a, S, A> {
  run_f: Rc<Box<dyn FnOnce(S) -> (A, S) + 'a>>,
}

impl<'a, S, A> State<'a, S, A> {
  pub fn unit(a: A) -> State<'a, S, A>
  where
    A: 'a, {
    Self::new(|s| (a, s))
  }

  pub fn new<'b, T, B, F>(f: F) -> State<'b, T, B>
  where
    F: FnOnce(T) -> (B, T) + 'b, {
    State {
      run_f: Rc::new(Box::new(f)),
    }
  }

  // pub fn new<'b, T, B>(run_f: Rc<Box<dyn FnOnce(T) -> (B, T) + 'b>>) -> State<'b, T, B> {
  //   State { run_f }
  // }

  pub fn run(self, s: S) -> (A, S) {
    let f = Rc::try_unwrap(self.run_f).unwrap_or_else(|err| panic!());
    f(s)
  }

  pub fn pure<'b, T, B>(b: B) -> State<'b, T, B>
  where
    B: Clone + 'b, {
    Self::new(move |s| (b.clone(), s))
  }

  pub fn fmap<'b, B, F>(self, f: F) -> State<'b, S, B>
  where
    F: FnOnce(A) -> B + 'b,
    B: Clone + 'b,
    A: 'a,
    S: 'a,
    'a: 'b, {
    self.bind(move |a| Self::pure(f(a)))
  }

  pub fn fmap2<'b, 'c, B, C, F>(self, sb: State<'b, S, B>, f: F) -> State<'c, S, C>
  where
    F: FnOnce(A, B) -> C + 'c,
    A: Clone + 'a,
    B: Clone + 'b,
    C: Clone + 'c,
    S: 'a,
    'a: 'b,
    'b: 'c, {
    self.bind(move |a| sb.fmap(move |b| f(a.clone(), b.clone())))
  }

  pub fn bind<'b, B, F>(self, f: F) -> State<'b, S, B>
  where
    F: FnOnce(A) -> State<'b, S, B> + 'b,
    B: Clone + 'b,
    A: 'a,
    S: 'a,
    'a: 'b, {
    Self::new(move |s| {
      let (a, s1) = self.run(s);
      f(a).run(s1)
    })
  }

  pub fn modify<'b, T, F>(f: F) -> State<'b, T, ()>
  where
    F: FnOnce(T) -> T + 'b,
    T: Clone + 'b, {
    let s = Self::get();
    s.bind(move |t: T| Self::set(f(t)))
  }

  pub fn get<'b, T>() -> State<'b, T, T>
  where
    T: Clone + 'b, {
    Self::new(move |t: T| (t.clone(), t))
  }

  pub fn set<'b, T>(t: T) -> State<'b, T, ()>
  where
    T: Clone + 'b, {
    Self::new(move |_| ((), t.clone()))
  }

  pub fn sequence(sas: Vec<State<'a, S, A>>) -> State<'a, S, Vec<A>>
  where
    S: 'a,
    A: 'a, {
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

impl<'a, S, A> Default for State<'a, S, A>
where
  S: Default,
  A: Default,
{
  fn default() -> Self {
    Self::new(|_| (A::default(), S::default()))
  }
}

#[cfg(test)]
mod tests {
  use crate::state::*;

  #[test]
  fn state() {
    let s = State::<i32, i32>::pure(10);
    let r = s.run(10);
    println!("{:?}", r);
  }
}
