use crate::rng::RNG;

mod gen;
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
    rands.first().unwrap(),
    rands.last().unwrap()
  );
}

fn main() {
  ints3();
}
