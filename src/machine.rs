use crate::state::State;

/// Input for the candy machine.<br/>
/// キャンディマシンへの入力。
#[derive(Debug, Clone, Copy)]
enum Input {
  /// Insert a coin into the machine.<br/>
  /// マシンにコインを投入する。
  Coin,
  /// Turn the knob to get a candy.<br/>
  /// キャンディを取得するためにノブを回す。
  Turn,
}

/// Represents a candy machine state.<br/>
/// キャンディマシンの状態を表す。
#[derive(Debug, Clone, Copy, Default)]
struct Machine {
  /// Whether the machine is locked.<br/>
  /// マシンがロックされているかどうか。
  locked: bool,
  /// Number of candies in the machine.<br/>
  /// マシン内のキャンディの数。
  candies: i32,
  /// Number of coins in the machine.<br/>
  /// マシン内のコインの数。
  coins: i32,
}

impl Machine {
  /// Simulates the candy machine with a sequence of inputs.<br/>
  /// 一連の入力でキャンディマシンをシミュレートする。
  ///
  /// # Arguments
  /// * `inputs` - A vector of inputs to the machine.
  ///
  /// # Returns
  /// * `State<Machine, (i32, i32)>` - A state monad that returns a tuple of coins and candies.
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

  /// Creates a function that updates the machine state based on input.<br/>
  /// 入力に基づいてマシンの状態を更新する関数を作成する。
  ///
  /// # Returns
  /// * `Box<dyn Fn(Input) -> Box<dyn Fn(Machine) -> Machine>>` - A function that takes an input and returns a function that updates the machine state.
  fn update() -> Box<dyn Fn(Input) -> Box<dyn Fn(Machine) -> Machine>> {
    Box::new(move |i: Input| {
      Box::new(move |s: Machine| {
        match (i, s) {
          // Inserting a coin into an unlocked machine does nothing
          // ロックされていないマシンにコインを入れても何も起こらない
          // (Coin, Machine { locked: false, .. }) => s.clone(),
          
          // Turning the knob on a locked machine does nothing
          // ロックされたマシンのノブを回しても何も起こらない
          // (Turn, Machine { locked: true, .. }) => s.clone(),
          
          // Inserting a coin into a locked machine unlocks it if there are candies
          // ロックされたマシンにコインを入れると、キャンディがある場合はロックが解除される
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
          // ロックされていないマシンのノブを回すと、キャンディが出てマシンがロックされる
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
          // その他のアクションは何も起こらない
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
