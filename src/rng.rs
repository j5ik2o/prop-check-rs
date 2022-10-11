pub trait NextRandValue
where
  Self: Sized, {
  fn next_i64(&self) -> (i64, Self);

  fn next_u64(&self) -> (u64, Self) {
    let (i, r) = self.next_i64();
    (if i < 0 { -(i + 1) as u64 } else { i as u64 }, r)
  }

  fn next_i32(&self) -> (i32, Self);

  fn next_u32(&self) -> (u32, Self) {
    let (i, r) = self.next_i32();
    (if i < 0 { -(i + 1) as u32 } else { i as u32 }, r)
  }

  fn next_i16(&self) -> (i16, Self);

  fn next_u16(&self) -> (u16, Self) {
    let (i, r) = self.next_i16();
    (if i < 0 { -(i + 1) as u16 } else { i as u16 }, r)
  }

  fn next_i8(&self) -> (i8, Self);

  fn next_u8(&self) -> (u8, Self) {
    let (i, r) = self.next_i8();
    (if i < 0 { -(i + 1) as u8 } else { i as u8 }, r)
  }

  fn next_f64(&self) -> (f64, Self) {
    let (i, r) = self.next_i64();
    (i as f64 / (i64::MAX as f64 + 1.0f64), r)
  }

  fn next_f32(&self) -> (f32, Self) {
    let (i, r) = self.next_i32();
    (i as f32 / (i32::MAX as f32 + 1.0f32), r)
  }

  fn next_bool(&self) -> (bool, Self) {
    let (i, r) = self.next_i32();
    ((i % 2) != 0, r)
  }
}

pub trait RandGen<T: NextRandValue>
where
  Self: Sized, {
  fn rnd_gen(rng: T) -> (Self, T);
}

impl<T: NextRandValue> RandGen<T> for i64 {
  fn rnd_gen(rng: T) -> (Self, T) {
    rng.next_i64()
  }
}

impl<T: NextRandValue> RandGen<T> for u32 {
  fn rnd_gen(rng: T) -> (Self, T) {
    rng.next_u32()
  }
}

impl<T: NextRandValue> RandGen<T> for i32 {
  fn rnd_gen(rng: T) -> (Self, T) {
    rng.next_i32()
  }
}

impl<T: NextRandValue> RandGen<T> for i16 {
  fn rnd_gen(rng: T) -> (Self, T) {
    rng.next_i16()
  }
}

impl<T: NextRandValue> RandGen<T> for f32 {
  fn rnd_gen(rng: T) -> (Self, T) {
    rng.next_f32()
  }
}

impl<T: NextRandValue> RandGen<T> for bool {
  fn rnd_gen(rng: T) -> (Self, T) {
    rng.next_bool()
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RNG {
  seed: i64,
}

type DynRand<A> = dyn FnMut(RNG) -> (A, RNG);
type BoxRand<A> = Box<DynRand<A>>;

impl Default for RNG {
  fn default() -> Self {
    Self::new()
  }
}

impl NextRandValue for RNG {
  fn next_i64(&self) -> (i64, Self) {
    let new_seed = self.seed.wrapping_mul(0x5DEECE66D) & 0xFFFFFFFFFFFF;
    let next_rng = RNG { seed: new_seed };
    let n = (new_seed >> 16) as i64;
    (n, next_rng)
  }

  fn next_i32(&self) -> (i32, Self) {
    let (n, next_rng) = self.next_i64();
    let n = n as i32;
    (n, next_rng)
  }

  fn next_i16(&self) -> (i16, Self) {
    let (n, next_rng) = self.next_i64();
    let n = n as i16;
    (n, next_rng)
  }

  fn next_i8(&self) -> (i8, Self) {
    let (n, next_rng) = self.next_i64();
    let n = n as i8;
    (n, next_rng)
  }
}

impl RNG {
  pub fn new() -> Self {
    Self { seed: i64::MAX }
  }

  pub fn new_with_seed(seed: i64) -> Self {
    Self { seed }
  }

  pub fn with_seed(&mut self, seed: i64) {
    self.seed = seed;
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

  pub fn unit<A>(a: A) -> BoxRand<A>
  where
    A: Clone + 'static, {
    Box::new(move |rng: RNG| (a.clone(), rng))
  }

  pub fn sequence<A, F>(fs: Vec<F>) -> BoxRand<Vec<A>>
  where
    A: Clone + 'static,
    F: FnMut(RNG) -> (A, RNG) + 'static, {
    let unit = Self::unit(Vec::<A>::new());
    let result = fs.into_iter().fold(unit, |acc, e| {
      Self::map2(acc, e, |mut a, b| {
        a.push(b);
        a
      })
    });
    result
  }

  pub fn int_value() -> BoxRand<i32> {
    Box::new(move |rng| rng.next_i32())
  }

  pub fn double_value() -> BoxRand<f32> {
    Box::new(move |rng| rng.next_f32())
  }

  pub fn map<A, B, F1, F2>(mut s: F1, mut f: F2) -> BoxRand<B>
  where
    F1: FnMut(RNG) -> (A, RNG) + 'static,
    F2: FnMut(A) -> B + 'static, {
    Box::new(move |rng| {
      let (a, rng2) = s(rng);
      (f(a), rng2)
    })
  }

  pub fn map2<F1, F2, F3, A, B, C>(mut ra: F1, mut rb: F2, mut f: F3) -> BoxRand<C>
  where
    F1: FnMut(RNG) -> (A, RNG) + 'static,
    F2: FnMut(RNG) -> (B, RNG) + 'static,
    F3: FnMut(A, B) -> C + 'static, {
    Box::new(move |rng| {
      let (a, r1) = ra(rng);
      let (b, r2) = rb(r1);
      (f(a, b), r2)
    })
  }

  pub fn both<F1, F2, A, B>(ra: F1, rb: F2) -> BoxRand<(A, B)>
  where
    F1: FnMut(RNG) -> (A, RNG) + 'static,
    F2: FnMut(RNG) -> (B, RNG) + 'static, {
    Self::map2(ra, rb, |a, b| (a, b))
  }

  pub fn rand_int_double() -> BoxRand<(i32, f32)> {
    Self::both(Self::int_value(), Self::double_value())
  }

  pub fn rand_double_int<'a>() -> BoxRand<(f32, i32)> {
    Self::both(Self::double_value(), Self::int_value())
  }

  pub fn flat_map<A, B, F, GF, BF>(mut f: F, mut g: GF) -> BoxRand<B>
  where
    F: FnMut(RNG) -> (A, RNG) + 'static,
    BF: FnMut(RNG) -> (B, RNG) + 'static,
    GF: FnMut(A) -> BF + 'static, {
    Box::new(move |rng| {
      let (a, r1) = f(rng);
      (g(a))(r1)
    })
  }

  // pub fn non_negative_less_than(n: u32) -> BoxRand<u32> {
  //   Self::flat_map(
  //     |rng| rng.next_u32(),
  //     move |i| {
  //       let m = i % n;
  //       //if i + (n - 1) - m >= 0 {
  //         Self::unit(m)
  //       // } else {
  //       //   Self::non_negative_less_than(n)
  //       // }
  //     },
  //   )
  // }
}

#[cfg(test)]
mod tests {
  use crate::rng::{RandGen, RNG};

  #[test]
  fn next_int() {
    let rng = RNG::new();
    let (v1, r1) = i32::rnd_gen(rng);
    println!("{:?}", v1);
    let (v2, _) = u32::rnd_gen(r1);
    println!("{:?}", v2);
  }
}
