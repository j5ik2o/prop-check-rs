use std::ops::{Add, Div, Mul, Neg};

use bigdecimal::{BigDecimal, Zero};
use iso_4217::CurrencyCode;
use num_bigint::BigInt;

#[derive(Debug, PartialEq)]
pub struct Money { amount: BigDecimal, currency: CurrencyCode }

#[derive(Debug, PartialEq)]
pub enum MoneyError {
    NotSameCurrencyError,
}

impl Money {
    pub fn new(amount: BigDecimal, currency: CurrencyCode) -> Self {
        let a = amount.with_scale(currency.digit().map(|v| v as i64).unwrap_or(0i64));
        Self { amount: a, currency }
    }

    pub fn from_bigint(amount: BigInt, currency: CurrencyCode) -> Self {
        let a = BigDecimal::from((amount, currency.digit().map(|v| v as i64).unwrap_or(0i64)));
        Self { amount: a, currency }
    }

    pub fn from_u64(amount: u64, currency: CurrencyCode) -> Self {
        let a = BigDecimal::from(amount).with_scale(currency.digit().map(|v| v as i64).unwrap_or(0i64));
        Self { amount: a, currency }
    }

    pub fn abs(&self) -> Self {
        Self { amount: self.amount.abs(), currency: self.currency }
    }

    pub fn is_positive(&self) -> bool {
        self.amount > BigDecimal::zero()
    }

    pub fn is_negative(&self) -> bool {
        self.amount < BigDecimal::zero()
    }

    pub fn is_zero(&self) -> bool {
        self.amount.is_zero()
    }

    pub fn negated(self) -> Self {
        Self { amount: self.amount.neg(), currency: self.currency }
    }

    pub fn add(self, other: Self) -> Result<Self, MoneyError> {
        if self.currency != other.currency {
            Err(MoneyError::NotSameCurrencyError)
        } else {
            Ok(Self { amount: self.amount.add(other.amount), currency: self.currency })
        }
    }

    pub fn subtract(self, other: Self) -> Result<Self, MoneyError> {
        self.add(other.negated())
    }

    pub fn times(self, factor: BigDecimal) -> Self {
        Self { amount: self.amount.mul(factor), currency: self.currency }
    }

    pub fn divided_by(self, divisor: BigDecimal) -> Self {
        Self { amount: self.amount.div(divisor), currency: self.currency }
    }
}

#[cfg(test)]
mod tests {
    use crate::money::{MoneyError, Money};
    use iso_4217::CurrencyCode;

    #[test]
    fn test_add() -> Result<(), MoneyError> {
        let m1 = Money::from_u64(1, CurrencyCode::JPY);
        let m2 = Money::from_u64(2, CurrencyCode::JPY);
        let m3 = m1.add(m2)?;
        println!("{:?}", m3);
        Ok(())
    }
}