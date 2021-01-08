use crate::state::RNG;

mod state;

fn main() {
  let mut rng = RNG::new();
  let count = 100000000;
  //let (rands, new_rng) = rng.ints1(count); // stackが溢れる
  //let (rands, new_rng) = rng.ints2(count); // stackが溢れる
  let (rands, new_rng) = rng.ints3(count); // 問題なし
  println!("{}: {}...{}", count, rands.get(0).unwrap(), rands.get(rands.len() -1).unwrap());
}