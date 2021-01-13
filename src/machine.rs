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
  fn simulate_machine<'a>(inputs: Vec<Input>) -> State<'a, Machine, (i32, i32)> {
    let mut xs = inputs
      .into_iter()
      .map(move |i| {
        let uf: Box<dyn Fn(Input) -> Box<dyn Fn(Machine) -> Machine>> = Self::update();
        let r: Box<dyn Fn(Machine) -> Machine> = uf(i);
        State::<Machine, ()>::modify(move |m: Machine| r(m))
      })
      .collect::<Vec<_>>();

    let result = State::sequence(xs);
    result.bind(|_| State::<Machine, Machine>::get().fmap(|s: Machine| (s.coins, s.candies)))
  }

  fn update() -> Box<dyn Fn(Input) -> Box<dyn Fn(Machine) -> Machine>> {
    Box::new(move |i: Input| {
      Box::new(move |s: Machine| {
        match (i, s) {
          (_, Machine { candies: 0, .. }) => s.clone(),
          // (Coin, Machine { locked: false, .. }) => s.clone(),
          // (Turn, Machine { locked: true, .. }) => s.clone(),
          (
            Coin,
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
            Turn,
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
        }
      })
    })
  }
}

#[cfg(test)]
mod tests {
  use crate::machine::{Machine, Input};

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
