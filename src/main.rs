#![feature(slice_fill_with)]
#![feature(box_syntax)]

use crate::rng::RNG;

mod machine;
mod prop;
mod rng;
mod state;

const COUNT: u32 = 100000000;

fn ints3() {
  let rng: RNG = RNG::new();
  let (rands, _) = rng.i32s(COUNT);
  println!(
    "{}: {}...{}",
    COUNT,
    rands.get(0).unwrap(),
    rands.get(rands.len() - 1).unwrap()
  );
}

fn main() {
  ints3();
}
