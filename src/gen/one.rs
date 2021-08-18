use crate::gen::{Gen, Gens};

pub trait One where Self: Sized {
    fn one() -> Gen<Self>;
}

impl One for i64 {
    fn one() -> Gen<Self> {
        Gens::one_i64()
    }
}

impl One for u64 {
    fn one() -> Gen<Self> {
        Gens::one_u64()
    }
}
