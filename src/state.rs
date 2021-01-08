pub trait RNG {
  fn next_int(&self) -> (i32, Self);
}

struct Simple {
  seed: i64,
}

type DynRand<A> = dyn Fn(Simple) -> (A, Simple);
type BoxRand<A> = Box<DynRand<A>>;

impl Simple {
  pub fn new() -> Self {
    Simple { seed: i64::MAX }
  }

  pub fn non_negative_int(&self) -> (i32, Self) {
    let (i, r) = self.next_int();
    (if i < 0 { -(i + 1) } else { i }, r)
  }

  pub fn double(&self) -> (f32, Self) {
    let (i, r) = self.next_int();
    (i as f32 / (std::i32::MAX as f32 + 1.0f32), r)
  }

  pub fn int_double(&self) -> ((i32, f32), Self) {
    let (i, r1) = self.next_int();
    let (d, r2) = r1.double();
    ((i, d), r2)
  }

  pub fn double_int(&self) -> ((f32, i32), Self) {
    let ((i, d), r) = self.int_double();
    ((d, i), r)
  }

  pub fn double_3(&self) -> ((f32, f32, f32), Self) {
    let (d1, r1) = self.double();
    let (d2, r2) = r1.double();
    let (d3, r3) = r2.double();
    ((d1, d2, d3), r3)
  }

  pub fn ints(self, count: i32) -> (Vec<i32>, Self) {
    if count == 0 {
      (vec![], self)
    } else {
      let (x, r1) = self.next_int();
      let (mut xs, r2) = r1.ints(count - 1);
      let mut xl = vec![x];
      xl.append(&mut xs);
      (xl, r2)
    }
  }

  pub fn ints2(self, count: i32) -> (Vec<i32>, Self) {
    fn go(count: i32, r: Simple, mut xs: Vec<i32>) -> (Vec<i32>, Simple) {
      if count == 0 {
        (xs, r)
      } else {
        let (x, r2) = r.next_int();
        let mut xl = vec![x];
        xl.append(&mut xs);
        go(count - 1, r2, xl)
      }
    }
    go(count, self, vec![])
  }

  pub fn int_value() -> BoxRand<i32> {
    Box::new(move |rng: Simple| rng.next_int())
  }

  pub fn map<A, B, F1, F2>(s: F1, f: F2) -> BoxRand<B>
    where
      F1: Fn(Simple) -> (A, Simple) + 'static,
      F2: Fn(A) -> B + 'static,
  {
    Box::new(move |rng| {
      let (a, rng2) = s(rng);
      (f(a), rng2)
    })
  }

  pub fn double_value() -> BoxRand<f32> {
    Self::map(
      |rng| {
        let result = Self::non_negative_int(&rng);
        result
      },
      |x| {
        let result = x as f32 / (i32::MAX as f32 + 1.0f32);
        result
      },
    )
  }

  pub fn map2<F1, F2, F3, A, B, C>(ra: F1, rb: F2, f: F3) -> BoxRand<C>
    where
      F1: Fn(Simple) -> (A, Simple) + 'static,
      F2: Fn(Simple) -> (B, Simple) + 'static,
      F3: Fn(A, B) -> C + 'static,
  {
    Box::new(move |rng| {
      let (a, r1) = ra(rng);
      let (b, r2) = rb(r1);
      (f(a, b), r2)
    })
  }

  pub fn both<F1, F2, A, B>(ra: F1, rb: F2) -> BoxRand<(A, B)>
    where
      F1: Fn(Simple) -> (A, Simple) + 'static,
      F2: Fn(Simple) -> (B, Simple) + 'static,
  {
    Self::map2(ra, rb, |a, b| (a, b))
  }

  pub fn rand_int_double() -> BoxRand<(i32, f32)> {
    Self::both(Self::int_value(), Self::double_value())
  }

  pub fn rand_double_int() -> BoxRand<(f32, i32)> {
    Self::both(Self::double_value(), Self::int_value())
  }

  pub fn unit<A: Clone + 'static>(a: A) -> BoxRand<A> {
    Box::new(move |rng| (a.clone(), rng))
  }

  pub fn sequence<A: Clone + 'static, F>(fs: Vec<F>) -> BoxRand<Vec<A>>
    where
      F: Fn(Simple) -> (A, Simple) + 'static,
  {
    let unit = Self::unit(Vec::<A>::new());
    let result = fs.into_iter().rev().fold(unit, |acc, e| {
      Self::map2(acc, e, |mut a: Vec<A>, b: A| {
        a.push(b);
        a
      })
    });
    result
  }

  pub fn _ints(count: i32) -> BoxRand<Vec<i32>> {
    let mut v: Vec<Box<DynRand<i32>>> = Vec::with_capacity(count as usize);
    v.resize_with(count as usize, || Self::int_value());
    Self::sequence(v)
  }

  pub fn flat_map<A, B, F, GF, BF>(f: F, g: GF) -> BoxRand<B>
    where
      F: Fn(Simple) -> (A, Simple) + 'static,
      BF: Fn(Simple) -> (B, Simple),
      GF: Fn(A) -> BF + 'static,
  {
    Box::new(move |rng| {
      let (a, r1): (A, Simple) = f(rng);
      let f = g(a);
      f(r1)
    })
  }

  pub fn non_negative_less_than(n: i32) -> BoxRand<i32> {
    Self::flat_map(
      |rng| rng.non_negative_int(),
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

  pub fn _map<A, B: Clone + 'static, AF, BF>(s: AF, f: BF) -> BoxRand<B>
    where
      AF: Fn(Simple) -> (A, Simple) + 'static,
      BF: Fn(A) -> B + 'static,
  {
    Self::flat_map(move |rng| s(rng), move |a| Self::unit(f(a)))
  }

  /*
    def _map2[A,B,C](ra: Rand[A], rb: Rand[B])(f: (A, B) => C): Rand[C] =
  flatMap(ra)(a => map(rb)(b => f(a, b)))
   */

  // pub fn _map2<A: Clone + 'static, B: Clone, C, AF, BF, CF>(ra: AF, rb: BF, f: CF) -> BoxRand<C>
  // where
  //     AF: Fn(Simple) -> (A, Simple) + 'static,
  //     BF: Fn(Simple) -> (B, Simple) + 'static,
  //     CF: Fn(A, B) -> C + 'static,
  // {
  //     Self::flat_map(
  //         move |rng| ra(rng),
  //         move |a| {
  //             let cf = Self::map(|rng| rb(rng), move |b| f(a.clone(), b.clone()));
  //             cf
  //         },
  //     )
  // }
}

impl RNG for Simple {
  fn next_int(&self) -> (i32, Self) {
    let new_seed = self.seed.wrapping_mul(0x5DEECE66D) & 0xFFFFFFFFFFFF;
    let next_rng = Simple { seed: new_seed };
    let n = (new_seed >> 16) as i32;
    (n, next_rng)
  }
}

mod state {
  use std::marker::PhantomData;

  use crate::state::{RNG, Simple};

  struct State<S, A> {
    run: Box<dyn Fn(S) -> (A, S)>,
  }

  type Rand<A> = State<Simple, A>;

  impl<S: Clone + 'static, A: Clone + 'static> State<S, A> {
    pub fn unit<X: Clone + 'static>(x: X) -> State<S, X> {
      State {
        run: Box::new(move |s| { (x.clone(), s) }),
      }
    }

    pub fn map<B: Clone + 'static, F>(self, f: F) -> State<S, B> where F: Fn(A) -> B + 'static {
      self.flat_map(move |a| Self::unit(f(a)))
    }

    pub fn flat_map<B, F>(self, f: F) -> State<S, B> where F: Fn(A) -> State<S, B> + 'static {
      let r1 = self.run;
      State {
        run: Box::new(move |s| {
          let (b, s1) = r1(s);
          let r2 = f(b).run;
          r2(s1)
        }),
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use crate::state::{RNG, Simple};

  #[test]
  fn next_int() {
    let (v1, r1) = Simple::new().next_int();
    println!("{:?}", v1);
    let (v2, r2) = r1.non_negative_int();
    println!("{:?}", v2);
  }

  #[test]
  fn it_works() {
    assert_eq!(2 + 2, 4);
  }
}
