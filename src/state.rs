pub struct State<'a, S, A> {
  run_f: Box<dyn Fn(S) -> (A, S) + 'a>,
}

impl<'a, S, A> Default for State<'a, S, A>
where
  S: Default,
  A: Default,
{
  fn default() -> Self {
    Self::new(Box::new(|_| (A::default(), S::default())))
  }
}

impl<'a, S, A> State<'a, S, A> {
  pub fn new<'b, T, B>(run_f: Box<dyn Fn(T) -> (B, T) + 'b>) -> State<'b, T, B> {
    State { run_f }
  }

  pub fn run(&self, s: S) -> (A, S) {
    (self.run_f)(s)
  }

  pub fn pure<'b, T, B>(b: B) -> State<'b, T, B>
  where
    B: Clone + 'b, {
    Self::new(Box::new(move |s| (b.clone(), s)))
  }

  pub fn fmap<'b, B, F>(self, f: F) -> State<'b, S, B>
  where
    F: Fn(A) -> B + 'b,
    B: Clone + 'b,
    A: 'a,
    S: 'a,
    'a: 'b, {
    self.bind(move |a| Self::pure(f(a)))
  }

  // pub fn fmap2<'b, 'c, B, C, F>(self, mut sb: State<'b, S, B>, f: F) -> State<'c, S, C>
  // where
  //   F: Fn(A, B) -> C + 'c,
  //   A: Clone + 'a,
  //   B: Clone + Default + 'b,
  //   C: Clone + 'c,
  //   S: Default + 'a,
  //   'a: 'c,
  //   'b: 'c, {
  //   self.bind(move |a| {
  //     let state_cloned = std::mem::replace(&mut sb, State::default());
  //     state_cloned.fmap(move |b| f(a.clone(), b)) } )
  // }

  pub fn bind<'b, B, F>(self, f: F) -> State<'b, S, B>
  where
    F: Fn(A) -> State<'b, S, B> + 'b,
    B: Clone + 'b,
    A: 'a,
    S: 'a,
    'a: 'b, {
    Self::new(Box::new(move |s| {
      let (a, s1) = self.run(s);
      f(a).run(s1)
    }))
  }

  pub fn modify<'b, T, F>(f: F) -> State<'b, T, ()>
  where
    F: Fn(T) -> T + 'b,
    T: Clone + 'b, {
    let s = Self::get();
    s.bind(move |t: T| Self::set(f(t)))
  }

  pub fn get<'b, T>() -> State<'b, T, T>
  where
    T: Clone + 'b, {
    Self::new(Box::new(move |t| (t.clone(), t)))
  }

  pub fn set<'b, T>(t: T) -> State<'b, T, ()>
  where
    T: Clone + 'b, {
    Self::new(Box::new(move |_| ((), t.clone())))
  }

  // pub fn sequence(sas: Vec<State<'a, S, A>>) -> State<'a, S, Vec<A>>
  //   where
  //     S: 'a,
  //     A: 'a, {
  //   Self::new(Box::new(move |s| {
  //     let mut s_ = s;
  //     let actions = sas;
  //     let mut acc: Vec<A> = vec![];
  //     for x in actions.into_iter() {
  //       let (a, s2) = x.run(s_);
  //       s_ = s2;
  //       acc.push(a);
  //     }
  //     (acc, s_)
  //   }))
  // }
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
