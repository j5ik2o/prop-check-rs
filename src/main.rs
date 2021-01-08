use crate::state::RNG;

mod state;

const COUNT: u32 = 100000000;

fn ints1() {
  let rng: RNG = RNG::new();
  let (rands, _) = rng.ints1(COUNT); // stackが溢れる
  println!("{}: {}...{}", COUNT, rands.get(0).unwrap(), rands.get(rands.len() - 1).unwrap());
}

fn ints2() {
  let rng: RNG = RNG::new();
  let (rands, _) = rng.ints2(COUNT); // stackが溢れる
  println!("{}: {}...{}", COUNT, rands.get(0).unwrap(), rands.get(rands.len() - 1).unwrap());
}

fn ints3() {
  let rng: RNG = RNG::new();
  let (rands, _) = rng.ints3(COUNT);
  println!("{}: {}...{}", COUNT, rands.get(0).unwrap(), rands.get(rands.len() - 1).unwrap());
}

fn ints_f() {
  let rands_f = RNG::ints_f(COUNT);
  let rng: RNG = RNG::new();
  let (rands, _) = rands_f(rng); // stackが溢れる
  println!("{}: {}...{}", COUNT, rands.get(0).unwrap(), rands.get(rands.len() - 1).unwrap());
}

fn main() {
  ints3();
}