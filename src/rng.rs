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

  fn next_bool(&self) -> (bool, Self)
  where
    Self: Sized, {
    let (i, r) = self.next_i32();
    ((i % 2) != 0, r)
  }
}

#[derive(Clone, Debug)]
pub struct RNG {
  seed: i64,
}

type DynRand<'a, A> = dyn FnOnce(RNG) -> (A, RNG) + 'a;
type BoxRand<'a, A> = Box<DynRand<'a, A>>;

impl Default for RNG {
  fn default() -> Self {
    Self::new()
  }
}

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

  pub fn i32_f32(&self) -> ((i32, f32), Self) {
    let (i, r1) = self.next_i32();
    let (d, r2) = r1.next_f32();
    ((i, d), r2)
  }

  pub fn f32_i32(&self) -> ((f32, i32), Self) {
    let ((i, d), r) = self.i32_f32();
    ((d, i), r)
  }

  pub fn f32_3(&self) -> ((f32, f32, f32), Self) {
    let (d1, r1) = self.next_f32();
    let (d2, r2) = r1.next_f32();
    let (d3, r3) = r2.next_f32();
    ((d1, d2, d3), r3)
  }

  pub fn i32s(self, count: u32) -> (Vec<i32>, Self) {
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

  pub fn unit<'a, A: Clone + 'a>(a: A) -> BoxRand<'a, A> {
    Box::new(|rng: RNG| (a, rng))
  }

  pub fn sequence<'a, A: Clone + 'a, F>(fs: Vec<F>) -> BoxRand<'a, Vec<A>>
  where
    F: FnOnce(RNG) -> (A, RNG) + 'a, {
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
    F1: FnOnce(RNG) -> (A, RNG) + 'a,
    F2: FnOnce(A) -> B + 'a, {
    Box::new(move |rng| {
      let (a, rng2) = s(rng);
      (f(a), rng2)
    })
  }

  pub fn map2<'a, F1, F2, F3, A, B, C>(ra: F1, rb: F2, f: F3) -> BoxRand<'a, C>
  where
    F1: FnOnce(RNG) -> (A, RNG) + 'a,
    F2: FnOnce(RNG) -> (B, RNG) + 'a,
    F3: FnOnce(A, B) -> C + 'a, {
    Box::new(move |rng| {
      let (a, r1) = ra(rng);
      let (b, r2) = rb(r1);
      (f(a, b), r2)
    })
  }

  pub fn both<'a, F1, F2, A, B>(ra: F1, rb: F2) -> BoxRand<'a, (A, B)>
  where
    F1: FnOnce(RNG) -> (A, RNG) + 'a,
    F2: FnOnce(RNG) -> (B, RNG) + 'a, {
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
    F: FnOnce(RNG) -> (A, RNG) + 'a,
    BF: FnOnce(RNG) -> (B, RNG),
    GF: FnOnce(A) -> BF + 'a, {
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

#[cfg(test)]
mod tests {
  use crate::rng::{NextRandValue, RNG};

  #[test]
  fn next_int() {
    let (v1, r1) = RNG::new().next_i32();
    println!("{:?}", v1);
    let (v2, _) = r1.next_u32();
    println!("{:?}", v2);
  }
}
