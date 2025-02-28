use crate::state::State;

/// Input for the candy machine.
#[derive(Debug, Clone, Copy)]
enum Input {
  /// Insert a coin into the machine.
  Coin,
  /// Turn the knob to get a candy.
  Turn,
}

/// Represents a candy machine state.
#[derive(Debug, Clone, Copy, Default)]
struct Machine {
  /// Whether the machine is locked.
  locked: bool,
  /// Number of candies in the machine.
  candies: i32,
  /// Number of coins in the machine.
  coins: i32,
}

impl Machine {
  /// Simulates the candy machine with a sequence of inputs.
  ///
  /// # Arguments
  /// - `inputs` - A vector of inputs to the machine.
  ///
  /// # Returns
  /// - `State<Machine, (i32, i32)>` - A state monad that returns a tuple of coins and candies.
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

  /// Creates a function that updates the machine state based on input.
  ///
  /// # Returns
  /// - `Box<dyn Fn(Input) -> Box<dyn Fn(Machine) -> Machine>>` - A function that takes an input and returns a function that updates the machine state.
  fn update() -> Box<dyn Fn(Input) -> Box<dyn Fn(Machine) -> Machine>> {
    Box::new(move |i: Input| {
      Box::new(move |s: Machine| {
        match (i, s) {
          // Inserting a coin into an unlocked machine does nothing
          // Inserting a coin into an unlocked machine does nothing
          // (Coin, Machine { locked: false, .. }) => s.clone(),

          // Turning the knob on a locked machine does nothing
          // Turning the knob on a locked machine does nothing
          // (Turn, Machine { locked: true, .. }) => s.clone(),

          // Inserting a coin into a locked machine unlocks it if there are candies
          // Inserting a coin into a locked machine unlocks it if there are candies
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

          // Turning the knob on an unlocked machine dispenses a candy and locks the machine
          // Turning the knob on an unlocked machine dispenses a candy and locks the machine
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

          // Any other action does nothing
          // Any other action does nothing
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
