use crate::gen::{Gen, Gens};

pub trait Choose
where
  Self: Sized, {
  fn choose(min: Self, max: Self) -> Gen<Self>;
}

impl<A> Choose for Option<A>
where
  A: Choose + Clone + 'static,
{
  fn choose(min: Self, max: Self) -> Gen<Self> {
    match (min, max) {
      (Some(mn), Some(mx)) => Gens::choose(mn, mx).map(Some),
      (none, _) if none.is_none() => Gens::pure(none),
      (_, none) if none.is_none() => Gens::pure(none),
      _ => panic!("occurred error"),
    }
  }
}

impl<A, B> Choose for Result<A, B>
where
  A: Choose + Clone + 'static,
  B: Clone + 'static,
{
  fn choose(min: Self, max: Self) -> Gen<Self> {
    match (min, max) {
      (Ok(mn), Ok(mx)) => Gens::choose(mn, mx).map(Ok),
      (err, _) if err.is_err() => Gens::pure(err),
      (_, err) if err.is_err() => Gens::pure(err),
      _ => panic!("occurred error"),
    }
  }
}

impl Choose for usize {
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_u64(min as u64, max as u64).map(|v| v as usize)
  }
}

impl Choose for i64 {
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_i64(min, max)
  }
}

impl Choose for u64 {
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_u64(min, max)
  }
}

impl Choose for i32 {
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_i32(min, max)
  }
}

impl Choose for u32 {
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_u32(min, max)
  }
}

impl Choose for i16 {
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_i16(min, max)
  }
}

impl Choose for u16 {
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_u16(min, max)
  }
}

impl Choose for i8 {
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_i8(min, max)
  }
}

impl Choose for u8 {
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_u8(min, max)
  }
}

impl Choose for char {
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_char(min, max)
  }
}

impl Choose for f64 {
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_f64(min, max)
  }
}

impl Choose for f32 {
  fn choose(min: Self, max: Self) -> Gen<Self> {
    Gens::choose_f32(min, max)
  }
}
