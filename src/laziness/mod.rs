use std::convert::TryInto;
use std::rc::Rc;

pub enum Stream<'a, A> {
  Empty,
  Cons {
    h: Rc<Box<dyn FnOnce() -> A + 'a>>,
    t: Rc<Box<dyn FnOnce() -> Stream<'a, A> + 'a>>,
  },
}

impl<'a, A> Stream<'a, A> {
  pub fn cons<HF, TF>(hf: HF, tf: TF) -> Stream<'a, A>
  where
    HF: FnOnce() -> A + 'a,
    TF: FnOnce() -> Stream<'a, A> + 'a, {
    Stream::Cons {
      h: Rc::new(Box::new(hf)),
      t: Rc::new(Box::new(tf)),
    }
  }

  pub fn empty() -> Stream<'a, A> {
    Stream::Empty
  }

  pub fn ones() -> Stream<'a, u32> {
    Stream::cons(move || 1, move || Stream::<'a, u32>::ones())
  }

  pub fn from<'b>(n: u32) -> Stream<'b, u32> {
    Stream::<'b, u32>::cons(move || n, move || Stream::<'b, u32>::from(n + 1))
  }

  pub fn unfold<'b, S, B>(z: S, f_rc: Rc<Box<dyn FnOnce(S) -> Option<(B, S)> + 'b>>) -> Stream<'b, B>
  where
    B: 'b,
    S: 'b, {
    let f_rc_cloned = f_rc.clone();
    let f_ = Rc::try_unwrap(f_rc_cloned).unwrap_or_else(|err| panic!());
    match f_(z) {
      Some((h, s)) => Stream::cons(move || h, move || Stream::<'b, B>::unfold(s, f_rc)),
      None => Stream::empty(),
    }
  }

  pub fn fold_right<B, ZF>(self, z: ZF, f_rc: Rc<Box<dyn FnOnce(A, B) -> B>>) -> B
  where
    ZF: FnOnce() -> B, {
    match self {
      Stream::Cons { h, t } => {
        let h_ = Rc::try_unwrap(h).unwrap_or_else(|err| panic!());
        let t_ = Rc::try_unwrap(t).unwrap_or_else(|err| panic!());
        let f_rc_cloned = f_rc.clone();
        let f_ = Rc::try_unwrap(f_rc_cloned).unwrap_or_else(|err| panic!());
        f_(h_(), t_().fold_right(z, f_rc))
      }
      Stream::Empty => z(),
    }
  }

  pub fn fmap<'b, B, F>(self, f: F) -> Stream<'b, B>
  where
    F: FnOnce(A) -> B + 'static,
    A: 'b,
    B: 'b, {
    self.fold_right(
      move || Stream::Empty,
      Rc::new(Box::new(move |h, t| Stream::cons(move || f(h), move || t))),
    )
  }

  pub fn find(self, f: Rc<Box<dyn FnOnce(A) -> bool>>) -> Option<A> where A: Clone {
    match self {
      Stream::Empty => None,
      Stream::Cons { h, t } => {
        let h_ = Rc::try_unwrap(h).unwrap_or_else(|err| panic!());
        let t_ = Rc::try_unwrap(t).unwrap_or_else(|err| panic!());
        let f_rc = f.clone();
        let f_ = Rc::try_unwrap(f_rc).unwrap_or_else(|err| panic!());
        let v = h_();
        if f_(v.clone()) {
          Some(v)
        } else {
          t_().find(f)
        }
      }
    }
  }

  pub fn take(self, n: u32) -> Stream<'a, A>
  where
    A: 'a, {
    match self {
      Stream::Cons { h, t } if n > 1 => {
        let h_ = Rc::try_unwrap(h).unwrap_or_else(|err| panic!());
        let t_ = Rc::try_unwrap(t).unwrap_or_else(|err| panic!());
        Stream::<'a, A>::cons(move || h_(), move || t_().take(n - 1))
      }
      Stream::Cons { h, .. } if n == 1 => {
        let h_ = Rc::try_unwrap(h).unwrap_or_else(|err| panic!());
        Stream::<'a, A>::cons(|| h_(), || Stream::Empty)
      }
      _ => Stream::Empty,
    }
  }

  pub fn zip<'b, 'c, B>(self, s2: Stream<'b, B>) -> Stream<'c, (A, B)>
  where
    A: 'a,
    B: 'b,
    'a: 'b,
    'b: 'c, {
    self.zip_with(s2, Rc::new(Box::new(move |a, b| (a, b))))
  }

  pub fn zip_with<'b, 'c, B, C>(self, s2: Stream<'b, B>, f: Rc<Box<dyn FnOnce(A, B) -> C + 'c>>) -> Stream<'c, C>
  where
    A: 'a,
    B: 'b,
    C: 'c,
    'a: 'b,
    'b: 'c, {
    let r = Self::unfold(
      (self, s2),
      Rc::new(Box::new(move |t| match t {
        (Stream::Cons { h: h1, t: t1 }, Stream::Cons { h: h2, t: t2 }) => {
          let f_ = Rc::try_unwrap(f.clone()).unwrap_or_else(|err| panic!());
          let h1_ = Rc::try_unwrap(h1).unwrap_or_else(|err| panic!());
          let h2_ = Rc::try_unwrap(h2).unwrap_or_else(|err| panic!());
          let t1_ = Rc::try_unwrap(t1).unwrap_or_else(|err| panic!());
          let t2_ = Rc::try_unwrap(t2).unwrap_or_else(|err| panic!());
          Some((f_(h1_(), h2_()), (t1_(), t2_())))
        }
        (..) => None,
      })),
    );
    r
  }
}
