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

impl One for i32 {
    fn one() -> Gen<Self> {
        Gens::one_i32()
    }
}

impl One for u32 {
    fn one() -> Gen<Self> {
        Gens::one_u32()
    }
}

impl One for i16 {
    fn one() -> Gen<Self> {
        Gens::one_i16()
    }
}

impl One for u16 {
    fn one() -> Gen<Self> {
        Gens::one_u16()
    }
}

impl One for i8 {
    fn one() -> Gen<Self> {
        Gens::one_i8()
    }
}

impl One for u8 {
    fn one() -> Gen<Self> {
        Gens::one_u8()
    }
}

impl One for char {
    fn one() -> Gen<Self> {
        Gens::one_char()
    }
}

impl One for bool {
    fn one() -> Gen<Self> {
        Gens::one_bool()
    }
}

impl One for f64 {
    fn one() -> Gen<Self> {
        Gens::one_f64()
    }
}

impl One for f32 {
    fn one() -> Gen<Self> {
        Gens::one_f32()
    }
}