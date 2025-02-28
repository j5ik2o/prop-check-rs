use crate::gen::{Gen, Gens};

/// Trait for types that can generate a single random value.
pub trait One
where
  Self: Sized, {
  /// Generate a single random value of this type.
  ///
  /// # Returns
  /// * `Gen<Self>` - A generator that produces a random value of this type.
  fn one() -> Gen<Self>;
}

impl One for i64 {
  /// Generate a single random i64 value.
  fn one() -> Gen<Self> {
    Gens::one_i64()
  }
}

impl One for u64 {
  /// Generate a single random u64 value.
  fn one() -> Gen<Self> {
    Gens::one_u64()
  }
}

impl One for i32 {
  /// Generate a single random i32 value.
  fn one() -> Gen<Self> {
    Gens::one_i32()
  }
}

impl One for u32 {
  /// Generate a single random u32 value.
  fn one() -> Gen<Self> {
    Gens::one_u32()
  }
}

impl One for i16 {
  /// Generate a single random i16 value.
  fn one() -> Gen<Self> {
    Gens::one_i16()
  }
}

impl One for u16 {
  /// Generate a single random u16 value.
  fn one() -> Gen<Self> {
    Gens::one_u16()
  }
}

impl One for i8 {
  /// Generate a single random i8 value.
  fn one() -> Gen<Self> {
    Gens::one_i8()
  }
}

impl One for u8 {
  /// Generate a single random u8 value.
  fn one() -> Gen<Self> {
    Gens::one_u8()
  }
}

impl One for char {
  /// Generate a single random char value.
  fn one() -> Gen<Self> {
    Gens::one_char()
  }
}

impl One for bool {
  /// Generate a single random bool value.
  fn one() -> Gen<Self> {
    Gens::one_bool()
  }
}

impl One for f64 {
  /// Generate a single random f64 value.
  fn one() -> Gen<Self> {
    Gens::one_f64()
  }
}

impl One for f32 {
  /// Generate a single random f32 value.
  fn one() -> Gen<Self> {
    Gens::one_f32()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::rng::RNG;
  use std::env;

  fn init() {
    env::set_var("RUST_LOG", "info");
    let _ = env_logger::builder().is_test(true).try_init();
  }

  fn new_rng() -> RNG {
    RNG::new()
  }

  #[test]
  fn test_one_i64() {
    init();
    let gen = <i64 as One>::one();
    let (value, _) = gen.run(new_rng());
    assert!(value >= i64::MIN && value <= i64::MAX);
  }

  #[test]
  fn test_one_u64() {
    init();
    let gen = <u64 as One>::one();
    let (value, _) = gen.run(new_rng());
    assert!(value <= u64::MAX);
  }

  #[test]
  fn test_one_i32() {
    init();
    let gen = <i32 as One>::one();
    let (value, _) = gen.run(new_rng());
    assert!(value >= i32::MIN && value <= i32::MAX);
  }

  #[test]
  fn test_one_u32() {
    init();
    let gen = <u32 as One>::one();
    let (value, _) = gen.run(new_rng());
    assert!(value <= u32::MAX);
  }

  #[test]
  fn test_one_i16() {
    init();
    let gen = <i16 as One>::one();
    let (value, _) = gen.run(new_rng());
    assert!(value >= i16::MIN && value <= i16::MAX);
  }

  #[test]
  fn test_one_u16() {
    init();
    let gen = <u16 as One>::one();
    let (value, _) = gen.run(new_rng());
    assert!(value <= u16::MAX);
  }

  #[test]
  fn test_one_i8() {
    init();
    let gen = <i8 as One>::one();
    let (value, _) = gen.run(new_rng());
    assert!(value >= i8::MIN && value <= i8::MAX);
  }

  #[test]
  fn test_one_u8() {
    init();
    let gen = <u8 as One>::one();
    let (value, _) = gen.run(new_rng());
    assert!(value <= u8::MAX);
  }

  #[test]
  fn test_one_char() {
    init();
    let gen = <char as One>::one();
    let (value, _) = gen.run(new_rng());
    assert!(value as u32 <= u8::MAX as u32);
  }

  #[test]
  fn test_one_bool() {
    init();
    let gen = <bool as One>::one();
    let (value, _) = gen.run(new_rng());
    assert!(value == true || value == false);
  }

  #[test]
  fn test_one_f64() {
    init();
    let gen = <f64 as One>::one();
    let (value, _) = gen.run(new_rng());
    assert!(!value.is_nan());
  }

  #[test]
  fn test_one_f32() {
    init();
    let gen = <f32 as One>::one();
    let (value, _) = gen.run(new_rng());
    assert!(!value.is_nan());
  }
}
