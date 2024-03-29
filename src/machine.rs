use crate::state::State;

#[derive(Debug, Clone, Copy)]
enum Input {
  Coin,
  Turn,
}

#[derive(Debug, Clone, Copy, Default)]
struct Machine {
  locked: bool,
  candies: i32,
  coins: i32,
}

impl Machine {
  fn simulate_machine(inputs: Vec<Input>) -> State<Machine, (i32, i32)> {
    let xs = inputs
      .into_iter()
      .map(move |i| {
        let uf: Box<dyn Fn(Input) -> Box<dyn Fn(Machine) -> Machine>> = Self::update();
        let r: Box<dyn Fn(Machine) -> Machine> = uf(i);
        State::<Machine, ()>::modify(move |m: Machine| r(m))
      })
      .collect::<Vec<_>>();

    let result = State::sequence(xs);
    result.flat_map(|_| State::<Machine, Machine>::get().map(|s: Machine| (s.coins, s.candies)))
  }

  fn update() -> Box<dyn Fn(Input) -> Box<dyn Fn(Machine) -> Machine>> {
    Box::new(move |i: Input| {
      Box::new(move |s: Machine| {
        match (i, s) {
          // (Coin, Machine { locked: false, .. }) => s.clone(),
          // (Turn, Machine { locked: true, .. }) => s.clone(),
          (
            Input::Coin,
            Machine {
              locked: true,
              candies: candy,
              coins: coin,
            },
          ) => Machine {
            locked: false,
            candies: candy,
            coins: coin + 1,
          },
          (
            Input::Turn,
            Machine {
              locked: false,
              candies: candy,
              coins: coin,
            },
          ) => Machine {
            locked: true,
            candies: candy - 1,
            coins: coin,
          },
          (_, Machine { .. }) => s.clone(),
        }
      })
    })
  }
}

#[cfg(test)]
mod tests {
  use crate::machine::{Input, Machine};
  use std::env;

  #[ctor::ctor]
  fn init() {
    env::set_var("RUST_LOG", "info");
    let _ = env_logger::builder().is_test(true).try_init();
  }

  #[test]
  fn candy() {
    let state = Machine::simulate_machine(vec![Input::Coin, Input::Turn]);
    let result = state.run(Machine {
      locked: true,
      candies: 1,
      coins: 1,
    });
    println!("{:?}", result);
  }
}
