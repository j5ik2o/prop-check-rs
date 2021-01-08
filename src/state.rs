pub trait NextRandValue {
  fn next_i32(&self) -> (i32, Self);

  fn next_u32(&self) -> (u32, Self) where Self: Sized {
    let (i, r) = self.next_i32();
    (if i < 0 { -(i + 1) as u32 } else { i as u32 }, r)
  }

  fn next_f32(&self) -> (f32, Self) where Self: Sized {
    let (i, r) = self.next_i32();
    (i as f32 / (std::i32::MAX as f32 + 1.0f32), r)
  }
}

#[derive(Clone, Debug)]
pub struct RNG {
  seed: i64,
}

type DynRand<'a, A> = dyn Fn(RNG) -> (A, RNG) + 'a;
type BoxRand<'a, A> = Box<DynRand<'a, A>>;

impl NextRandValue for RNG {
  fn next_i32(&self) -> (i32, Self) {
    let new_seed = self.seed.wrapping_mul(0x5DEECE66D) & 0xFFFFFFFFFFFF;
    let next_rng = RNG { seed: new_seed };
    let n = (new_seed >> 16) as i32;
    (n, next_rng)
  }
}

impl RNG {
  pub fn new() -> Self {
    RNG { seed: i64::MAX }
  }

  pub fn int_double(&self) -> ((i32, f32), Self) {
    let (i, r1) = self.next_i32();
    let (d, r2) = r1.next_f32();
    ((i, d), r2)
  }

  pub fn double_int(&self) -> ((f32, i32), Self) {
    let ((i, d), r) = self.int_double();
    ((d, i), r)
  }

  pub fn double_3(&self) -> ((f32, f32, f32), Self) {
    let (d1, r1) = self.next_f32();
    let (d2, r2) = r1.next_f32();
    let (d3, r3) = r2.next_f32();
    ((d1, d2, d3), r3)
  }

  pub fn ints1(self, count: u32) -> (Vec<i32>, Self) {
    if count == 0 {
      (vec![], self)
    } else {
      let (x, new_rng) = self.next_i32();
      let (mut acc, new_rng) = new_rng.ints1(count - 1);
      acc.push(x);
      (acc, new_rng)
    }
  }

  pub fn ints2(self, count: u32) -> (Vec<i32>, Self) {
    fn go(count: u32, rng: RNG, mut acc: Vec<i32>) -> (Vec<i32>, RNG) {
      if count == 0 {
        (acc, rng)
      } else {
        let (x, new_rng) = rng.next_i32();
        acc.push(x);
        go(count - 1, new_rng, acc)
      }
    }
    go(count, self, vec![])
  }

  pub fn ints3(self, count: u32) -> (Vec<i32>, Self) {
    let mut index = count;
    let mut acc = vec![];
    let mut current_rng = self;
    while index > 0 {
      let (x, new_rng) = current_rng.next_i32();
      acc.push(x);
      index -= 1;
      current_rng = new_rng;
    }
    (acc, current_rng)
  }

  pub fn ints_f<'a>(count: u32) -> BoxRand<'a, Vec<i32>> {
    let mut fs: Vec<Box<DynRand<i32>>> = Vec::with_capacity(count as usize);
    fs.resize_with(count as usize, || Self::int_value());
    Self::sequence(fs)
  }

  pub fn unit<'a, A: Clone + 'a>(a: A) -> BoxRand<'a, A> {
    Box::new(move |rng| (a.clone(), rng))
  }

  pub fn sequence<'a, A: Clone + 'a, F>(fs: Vec<F>) -> BoxRand<'a, Vec<A>>
    where
      F: Fn(RNG) -> (A, RNG) + 'a,
  {
    let unit = Self::unit(Vec::<A>::new());
    let result = fs.into_iter().fold(unit, |acc, e| {
      Self::map2(acc, e, |mut a, b| {
        a.push(b);
        a
      })
    });
    result
  }

  pub fn int_value<'a>() -> BoxRand<'a, i32> {
    Box::new(move |rng| rng.next_i32())
  }

  pub fn double_value<'a>() -> BoxRand<'a, f32> {
    Box::new(move |rng| rng.next_f32())
  }

  pub fn map<'a, A, B, F1, F2>(s: F1, f: F2) -> BoxRand<'a, B>
    where
      F1: Fn(RNG) -> (A, RNG) + 'a,
      F2: Fn(A) -> B + 'a,
  {
    Box::new(move |rng| {
      let (a, rng2) = s(rng);
      (f(a), rng2)
    })
  }

  pub fn map2<'a, F1, F2, F3, A, B, C>(ra: F1, rb: F2, f: F3) -> BoxRand<'a, C>
    where
      F1: Fn(RNG) -> (A, RNG) + 'a,
      F2: Fn(RNG) -> (B, RNG) + 'a,
      F3: Fn(A, B) -> C + 'a,
  {
    Box::new(move |rng| {
      let (a, r1) = ra(rng);
      let (b, r2) = rb(r1);
      (f(a, b), r2)
    })
  }

  pub fn both<'a, F1, F2, A, B>(ra: F1, rb: F2) -> BoxRand<'a, (A, B)>
    where
      F1: Fn(RNG) -> (A, RNG) + 'a,
      F2: Fn(RNG) -> (B, RNG) + 'a,
  {
    Self::map2(ra, rb, |a, b| (a, b))
  }

  pub fn rand_int_double<'a>() -> BoxRand<'a, (i32, f32)> {
    Self::both(Self::int_value(), Self::double_value())
  }

  pub fn rand_double_int<'a>() -> BoxRand<'a, (f32, i32)> {
    Self::both(Self::double_value(), Self::int_value())
  }

  pub fn flat_map<'a, A, B, F, GF, BF>(f: F, g: GF) -> BoxRand<'a, B>
    where
      F: Fn(RNG) -> (A, RNG) + 'a,
      BF: Fn(RNG) -> (B, RNG),
      GF: Fn(A) -> BF + 'a,
  {
    Box::new(move |rng| {
      let (a, r1) = f(rng);
      let f = g(a);
      f(r1)
    })
  }

  pub fn non_negative_less_than<'a>(n: u32) -> BoxRand<'a, u32> {
    Self::flat_map(
      |rng| rng.next_u32(),
      move |i| {
        let m = i % n;
        if i + (n - 1) - m >= 0 {
          Self::unit(m)
        } else {
          Self::non_negative_less_than(n)
        }
      },
    )
  }
}
mod hoge {
  #![feature(unboxed_closures)]

  struct State<'a, S, A> {
    // `Fn(S) -> (A, S)` is a trait for environment-immutable closures.
    // others include `FnMut` (environment-mutable) and `FnOnce` (can only be called once);
    // this is very similar (and in fact, equivalent) to `&self`, `&mut self` and `self` methods respectively.
    // `Box<...>` is required for making it a concrete (sized) type, allowing it to be stored to the struct.
    // `+ 'a` is required since the trait can contain references (similar to `|...|: 'a -> ...` in the boxed closure).
    runState: Box<dyn Fn(S) -> (A, S) + 'a>
  }

  impl<'a, S, A> State<'a, S, A> {
    // unlike old closures, new closures are generic, so you need a trait bound.
    fn and_then<'b, B, F>(&'b self, f: F) -> State<'b, S, B>
      where F: Fn(A) -> State<'b, S, B> + 'b
    {
      State {
        // `box` is for making `Box<...>`.
        // `move |...| { ... }` means that the closure moves its environments into itself,
        // this is required since we lose `f` after the return.
        // the borrowing counterpart is called `ref |...| { ... }`, and a bare `|...| { ... }` will be inferred to one of both.
        runState: box move |firstState| {
          // currently there is a caveat for calling new closures in a box:
          // you cannot directly use the call syntax. you need to explicitly write the method name out.
          // also note the "weird" tuple construction, this makes one-element tuple.
          let (result, nextState) = (self.runState)((firstState));

          (f(result).runState)((nextState))
        }
      }
    }
  }
}

mod state {
  use std::rc::Rc;

  use crate::state::RNG;

  struct State<'a, S, A> {
    run: Box<dyn Fn(S) -> (A, S) + 'a>,
  }

  impl<'a, S, A> State<'a, S, A> {
    pub fn pure<'b, X: Clone + 'b>(x: X) -> State<'b, S, X> {
      State {
        run: box move |s| { (x.clone(), s) },
      }
    }

    pub fn fmap<'b, B: Clone + 'b, F>(&'b self, f: F) -> State<'b, S, B> where F: Fn(A) -> B + 'b {
      self.bind(move |a| Self::pure(f(a)))
    }

    pub fn bind<'b, B, F>(&'b self, f: F) -> State<'b, S, B> where F: Fn(A) -> State<'b, S, B> + 'b {
      State {
        run: box move |s| {
          let (b, s1) = (self.run)(s);
          (f(b).run)(s1)
        },
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use crate::state::{NextRandValue, RNG};

  #[test]
  fn next_int() {
    let (v1, r1) = RNG::new().next_i32();
    println!("{:?}", v1);
    let (v2, _) = r1.next_u32();
    println!("{:?}", v2);
  }
}
