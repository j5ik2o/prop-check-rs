use std::ops::{Add, Div, Mul, Neg};

use bigdecimal::{BigDecimal, Zero};
use iso_4217::CurrencyCode;
use num_bigint::BigInt;

#[derive(Debug, PartialEq)]
pub struct Money { amount: BigDecimal, currency: CurrencyCode }

pub enum MoneyError {
    NotSameCurrencyError,
}

impl Money {
    pub fn new(amount: BigInt, currency: CurrencyCode) -> Self {
        let a = BigDecimal::from((amount, currency.digit().map(|v| v as i64).unwrap_or(0i64)));
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
            let o = Self { amount: self.amount.add(other.amount), currency: self.currency };
            Ok(o)
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

