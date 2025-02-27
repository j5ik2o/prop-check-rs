//! # Prop-Check-RS
//!
//! A property-based testing library for Rust.
//!
//! This library provides tools for generating random test data and
//! defining properties that should hold for all inputs.

use crate::rng::RNG;

mod gen;
mod machine;
mod prop;
mod rng;
mod state;

/// Number of random integers to generate for the demo.
/// デモ用に生成するランダムな整数の数。
const COUNT: u32 = 100000000;

/// Generates a large number of random i32 values and returns the first and last.
/// 大量のランダムなi32値を生成し、最初と最後の値を返す。
///
/// # Returns
/// * `(u32, i32, i32)` - A tuple containing the count, first value, and last value.
pub fn generate_random_ints(count: u32) -> (u32, i32, i32) {
  // Create a new random number generator
  let rng: RNG = RNG::new();

  // Generate COUNT random i32 values
  let (rands, _) = rng.i32s(count);

  // Return the count, first value, and last value
  (count, *rands.get(0).unwrap(), *rands.get(rands.len() - 1).unwrap())
}

/// Generates a large number of random i32 values and prints the first and last.
/// 大量のランダムなi32値を生成し、最初と最後の値を表示する。
fn ints3() {
  let (count, first, last) = generate_random_ints(COUNT);
  println!("{}: {}...{}", count, first, last);
}

/// Main entry point for the application.
/// アプリケーションのメインエントリーポイント。
fn main() {
  ints3();
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::env;

  fn init() {
    env::set_var("RUST_LOG", "info");
    let _ = env_logger::builder().is_test(true).try_init();
  }

  #[test]
  fn test_generate_random_ints() {
    init();
    // 小さい値でテスト
    let count = 100;
    let (result_count, first, last) = generate_random_ints(count);

    // カウントが正しいことを確認
    assert_eq!(result_count, count);

    // 値が生成されていることを確認
    assert!(first >= i32::MIN && first <= i32::MAX);
    assert!(last >= i32::MIN && last <= i32::MAX);
  }
}
