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

/// Generates a large number of random i32 values and prints the first and last.
/// 大量のランダムなi32値を生成し、最初と最後の値を表示する。
fn ints3() {
  // Create a new random number generator
  let rng: RNG = RNG::new();
  
  // Generate COUNT random i32 values
  let (rands, _) = rng.i32s(COUNT);
  
  // Print the count, first value, and last value
  println!(
    "{}: {}...{}",
    COUNT,
    rands.get(0).unwrap(),
    rands.get(rands.len() - 1).unwrap()
  );
}

/// Main entry point for the application.
/// アプリケーションのメインエントリーポイント。
fn main() {
  ints3();
}
