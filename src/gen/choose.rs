use crate::gen::{Gen, Gens};

/// Trait for types that can be randomly chosen from a range.<br/>
/// 範囲からランダムに選択できる型のトレイト。
pub trait Choose
where
  Self: Sized, {
  /// Choose a random value between min and max (inclusive).<br/>
  /// min以上max以下のランダムな値を選択する。
  ///
  /// # Arguments
  /// * `min` - The minimum value (inclusive).
  /// * `max` - The maximum value (inclusive).
  ///
  /// # Returns
  /// * `Gen<Self>` - A generator that produces random values in the specified range.
  fn choose(min: Self, max: Self) -> Gen<Self>;
}

impl<A> Choose for Option<A>
where
  A: Choose + Clone + 'static,
{
  /// Choose a random value between min and max for Option types.<br/>
  /// Option型の場合、min以上max以下のランダムな値を選択する。
  ///
  /// # Arguments
  /// * `min` - The minimum value (inclusive).
  /// * `max` - The maximum value (inclusive).
  ///
  /// # Returns
  /// * `Gen<Self>` - A generator that produces random Option values.
  fn choose(min: Self, max: Self) -> Gen<Self> {
    match (min, max) {
      (Some(mn), Some(mx)) => Gens::choose(mn, mx).map(Some),
      (none, _) if none.is_none() => Gens::pure(none),
      (_, none) if none.is_none() => Gens::pure(none),
      _ => panic!("occurred error"),
    }
  }
}

impl<A, B> Choose for Result<A, B>
where
  A: Choose + Clone + 'static,
  B: Clone + 'static,
{
  /// Choose a random value between min and max for Result types.<br/>
  /// Result型の場合、min以上max以下のランダムな値を選択する。
  ///
  /// # Arguments
  /// * `min` - The minimum value (inclusive).
  /// * `max` - The maximum value (inclusive).
  ///
  /// # Returns
  /// * `Gen<Self>` - A generator that produces random Result values.
  fn choose(min: Self, max: Self) -> Gen<Self> {
    match (min, max) {
      (Ok(mn), Ok(mx)) => Gens::choose(mn, mx).map(Ok),
      (err, _) if err.is_err() => Gens::pure(err),
      (_, err) if err.is_err() => Gens::pure(err),
      _ => panic!("occurred error"),
    }
  }
}

impl Choose for usize {
  /// Choose a random usize value between min and max.<br/>
  /// usize型の場合、min以上max以下のランダムな値を選択する。
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_u64(min as u64, max as u64).map(|v| v as usize)
  }
}

impl Choose for i64 {
  /// Choose a random i64 value between min and max.<br/>
  /// i64型の場合、min以上max以下のランダムな値を選択する。
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_i64(min, max)
  }
}

impl Choose for u64 {
  /// Choose a random u64 value between min and max.<br/>
  /// u64型の場合、min以上max以下のランダムな値を選択する。
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_u64(min, max)
  }
}

impl Choose for i32 {
  /// Choose a random i32 value between min and max.<br/>
  /// i32型の場合、min以上max以下のランダムな値を選択する。
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_i32(min, max)
  }
}

impl Choose for u32 {
  /// Choose a random u32 value between min and max.<br/>
  /// u32型の場合、min以上max以下のランダムな値を選択する。
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_u32(min, max)
  }
}

impl Choose for i16 {
  /// Choose a random i16 value between min and max.<br/>
  /// i16型の場合、min以上max以下のランダムな値を選択する。
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_i16(min, max)
  }
}

impl Choose for u16 {
  /// Choose a random u16 value between min and max.<br/>
  /// u16型の場合、min以上max以下のランダムな値を選択する。
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_u16(min, max)
  }
}

impl Choose for i8 {
  /// Choose a random i8 value between min and max.<br/>
  /// i8型の場合、min以上max以下のランダムな値を選択する。
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_i8(min, max)
  }
}

impl Choose for u8 {
  /// Choose a random u8 value between min and max.<br/>
  /// u8型の場合、min以上max以下のランダムな値を選択する。
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_u8(min, max)
  }
}

impl Choose for char {
  /// Choose a random char value between min and max.<br/>
  /// char型の場合、min以上max以下のランダムな値を選択する。
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_char(min, max)
  }
}

impl Choose for f64 {
  /// Choose a random f64 value between min and max.<br/>
  /// f64型の場合、min以上max以下のランダムな値を選択する。
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_f64(min, max)
  }
}

impl Choose for f32 {
  /// Choose a random f32 value between min and max.<br/>
  /// f32型の場合、min以上max以下のランダムな値を選択する。
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_f32(min, max)
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
  fn test_choose_option() {
    init();
    // Some値の場合
    let gen = Option::<i32>::choose(Some(1), Some(10));
    let (value, _) = gen.run(new_rng());
    assert!(value.is_some());
    if let Some(v) = value {
      assert!(v >= 1 && v <= 10);
    }

    // None値の場合
    let gen = Option::<i32>::choose(None, Some(10));
    let (value, _) = gen.run(new_rng());
    assert!(value.is_none());
  }

  #[test]
  fn test_choose_result() {
    init();
    // Ok値の場合
    let gen = Result::<i32, String>::choose(Ok(1), Ok(10));
    let (value, _) = gen.run(new_rng());
    assert!(value.is_ok());
    if let Ok(v) = value {
      assert!(v >= 1 && v <= 10);
    }

    // Err値の場合
    let error = "error".to_string();
    let gen = Result::<i32, String>::choose(Err(error.clone()), Ok(10));
    let (value, _) = gen.run(new_rng());
    assert!(value.is_err());
    if let Err(e) = value {
      assert_eq!(e, error);
    }
  }

  #[test]
  fn test_choose_usize() {
    init();
    let min = 1usize;
    let max = 100usize;
    let gen = usize::choose(min, max);
    let (value, _) = gen.run(new_rng());
    assert!(value >= min && value <= max);
  }

  #[test]
  fn test_choose_i64() {
    init();
    let min = 1i64;
    let max = 100i64;
    let gen = i64::choose(min, max);
    let (value, _) = gen.run(new_rng());
    assert!(value >= min && value <= max);
  }

  #[test]
  fn test_choose_u64() {
    init();
    let min = 1u64;
    let max = 100u64;
    let gen = u64::choose(min, max);
    let (value, _) = gen.run(new_rng());
    assert!(value >= min && value <= max);
  }

  #[test]
  fn test_choose_i32() {
    init();
    let min = 1i32;
    let max = 100i32;
    let gen = i32::choose(min, max);
    let (value, _) = gen.run(new_rng());
    assert!(value >= min && value <= max);
  }

  #[test]
  fn test_choose_u32() {
    init();
    let min = 1u32;
    let max = 100u32;
    let gen = u32::choose(min, max);
    let (value, _) = gen.run(new_rng());
    assert!(value >= min && value <= max);
  }

  #[test]
  fn test_choose_i16() {
    init();
    let min = 1i16;
    let max = 100i16;
    let gen = i16::choose(min, max);
    let (value, _) = gen.run(new_rng());
    assert!(value >= min && value <= max);
  }

  #[test]
  fn test_choose_u16() {
    init();
    let min = 1u16;
    let max = 100u16;
    let gen = u16::choose(min, max);
    let (value, _) = gen.run(new_rng());
    assert!(value >= min && value <= max);
  }

  #[test]
  fn test_choose_i8() {
    init();
    let min = 1i8;
    let max = 100i8;
    let gen = i8::choose(min, max);
    let (value, _) = gen.run(new_rng());
    assert!(value >= min && value <= max);
  }

  #[test]
  fn test_choose_u8() {
    init();
    let min = 1u8;
    let max = 100u8;
    let gen = u8::choose(min, max);
    let (value, _) = gen.run(new_rng());
    assert!(value >= min && value <= max);
  }

  #[test]
  fn test_choose_char() {
    init();
    let min = 'a';
    let max = 'z';
    let gen = char::choose(min, max);
    let (value, _) = gen.run(new_rng());
    assert!(value >= min && value <= max);
  }

  #[test]
  fn test_choose_f64() {
    init();
    let min = 1.0f64;
    let max = 100.0f64;
    let gen = f64::choose(min, max);
    let (value, _) = gen.run(new_rng());
    assert!(value >= min && value <= max);
  }

  #[test]
  fn test_choose_f32() {
    init();
    let min = 1.0f32;
    let max = 100.0f32;
    let gen = f32::choose(min, max);
    let (value, _) = gen.run(new_rng());
    assert!(value >= min && value <= max);
  }
}
