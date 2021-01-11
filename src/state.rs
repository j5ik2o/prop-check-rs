pub trait NextRandValue {
  fn next_i32(&self) -> (i32, Self);

  fn next_u32(&self) -> (u32, Self)
    where
      Self: Sized, {
    let (i, r) = self.next_i32();
    (if i < 0 { -(i + 1) as u32 } else { i as u32 }, r)
  }

  fn next_f32(&self) -> (f32, Self)
    where
      Self: Sized, {
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
      F: Fn(RNG) -> (A, RNG) + 'a, {
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
      F2: Fn(A) -> B + 'a, {
    Box::new(move |rng| {
      let (a, rng2) = s(rng);
      (f(a), rng2)
    })
  }

  pub fn map2<'a, F1, F2, F3, A, B, C>(ra: F1, rb: F2, f: F3) -> BoxRand<'a, C>
    where
      F1: Fn(RNG) -> (A, RNG) + 'a,
      F2: Fn(RNG) -> (B, RNG) + 'a,
      F3: Fn(A, B) -> C + 'a, {
    Box::new(move |rng| {
      let (a, r1) = ra(rng);
      let (b, r2) = rb(r1);
      (f(a, b), r2)
    })
  }

  pub fn both<'a, F1, F2, A, B>(ra: F1, rb: F2) -> BoxRand<'a, (A, B)>
    where
      F1: Fn(RNG) -> (A, RNG) + 'a,
      F2: Fn(RNG) -> (B, RNG) + 'a, {
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
      GF: Fn(A) -> BF + 'a, {
    Box::new(move |rng| {
      let (a, r1) = f(rng);
      (g(a))(r1)
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

pub mod state {
  pub struct State<'a, S, A> {
    run_f: Box<dyn Fn(S) -> (A, S) + 'a>,
  }

  impl<'a, S, A> State<'a, S, A> {
    pub fn new<'b, T: Clone + 'b, B: Clone + 'b>(run_f: Box<dyn Fn(T) -> (B, T) + 'b>) -> State<'b, T, B>  {
      State {
        run_f
      }
    }

    pub fn run(&self, s: S) -> (A, S) {
      (self.run_f)(s)
    }

    pub fn pure<'b, T: Clone + 'b, B: Clone + 'b>(x: B) -> State<'b, T, B> {
      Self::new(
        box move |s| (x.clone(), s)
      )
    }

    pub fn fmap<'b, B: Clone + 'b, F>(&'a self, f: F) -> State<'b, S, B>
      where
        F: Fn(A) -> B + 'b, S: Clone, 'a: 'b {
      self.bind(move |a| Self::pure(f(a)))
    }

    pub fn bind<'b, B: Clone + 'b, F>(&'a self, f: F) -> State<'b, S, B>
      where
        F: Fn(A) -> State<'b, S, B> + 'b, S: Clone, 'a: 'b {
      Self::new(
        box move |s| {
          let (a, s1) = self.run(s);
          f(a).run(s1)
        },
      )
    }

    pub fn modify<'b, T: Clone + 'b, F>(f: F) -> State<'b, T, ()> where F: Fn(T) -> T + 'b {
      Self::get().bind(move |s| Self::set(f(s)))
    }

    pub fn get<'b, T: Clone + 'b>() -> State<'b, T, T> {
      Self::new(
        box move |s| (s.clone(), s)
      )
    }

    pub fn set<'b, T: Clone + 'b>(s: T) -> State<'b, T, ()> {
      Self::new(
        box move |_| ((), s.clone())
      )
    }

    // pub fn sequence<'b>(sas: &'b Vec<State<'b, S, A>>) -> State<'b, S, Vec<A>> {
    //   State::new(
    //     box move |s| {
    //       let mut s_ = s;
    //       let mut actions = sas;
    //       let mut acc: Vec<A> = vec![];
    //       for x in actions.into_iter() {
    //         let (a, s2) = x.run(s_);
    //         s_ = s2;
    //         acc.push(a);
    //       }
    //       (acc, s_)
    //     }
    //   )
    // }
  }
}

// #[cfg(test)]
// mod tests {
//   use crate::state::{NextRandValue, RNG};
//   use crate::state::state::*;
//
//   #[derive(Clone, Copy)]
//   enum Input {
//     Coin,
//     Turn,
//   }
//
//   #[derive(Clone, Copy)]
//   struct Machine { locked: bool, candies: i32, coins: i32 }
//
//   impl Machine {
//     fn simulate_machine<'a>(inputs: Vec<Input>) -> State<'a, Machine, (i32, i32)> {
//       let xs = inputs.into_iter().map(move |i| {
//         let uf: Box<dyn Fn(Input) -> Box<dyn Fn(Machine) -> Machine>> = Self::update();
//         let r: Box<dyn Fn(Machine) -> Machine> = uf(i);
//         State::<Machine, ()>::modify(move |m: Machine| r(m))
//       }).collect::<Vec<_>>();
//
//       let result = State::sequence(&xs);
//       result.bind(|_| {
//         State::<Machine, Machine>::get().fmap(|s: Machine| (s.coins, s.candies))
//       })
//     }
//
//     fn update() -> Box<dyn Fn(Input) -> Box<dyn Fn(Machine) -> Machine>> {
//       box move |i: Input| {
//         box move |s: Machine| {
//           match (i, s) {
//             (_, Machine { candies: 0, .. }) => s.clone(),
//             // (Coin, Machine { locked: false, .. }) => s.clone(),
//             // (Turn, Machine { locked: true, .. }) => s.clone(),
//             (Coin, Machine { locked: true, candies: candy, coins: coin }) =>
//               Machine { locked: false, candies: candy, coins: coin + 1 },
//             (Turn, Machine { locked: false, candies: candy, coins: coin }) =>
//               Machine { locked: true, candies: candy - 1, coins: coin },
//           }
//         }
//       }
//     }
//   }
//
//
//   #[test]
//   fn next_int() {
//     let (v1, r1) = RNG::new().next_i32();
//     println!("{:?}", v1);
//     let (v2, _) = r1.next_u32();
//     println!("{:?}", v2);
//   }
//
//   #[test]
//   fn state() {
//     let s = State::<i32, i32>::pure(10);
//     let r = s.run(10);
//     println!("{:?}", r);
//   }
// }
