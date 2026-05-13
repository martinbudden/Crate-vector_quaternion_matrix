#![allow(unused)]

use core::cmp::Ordering;
use core::fmt;
use core::iter::FromIterator;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index};
use serde::{Deserialize, Serialize};

/// A memory-efficient 128-bit set for embedded environments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct BitSet128(u64, u64);

impl BitSet128 {
    /// Create a new empty bitset.
    pub const fn new() -> Self {
        Self(0, 0)
    }
}

impl Default for BitSet128 {
    fn default() -> Self {
        Self::new()
    }
}

impl BitSet128 {
    /// Resets all bits to 0.
    #[inline]
    pub fn reset_all(&mut self) {
        self.0 = 0;
        self.1 = 0;
    }

    /// Resets the bit at `index` to 0. Does nothing if the index is out of bounds.
    #[inline]
    pub fn reset(&mut self, index: u8) {
        if index < 64 {
            self.0 &= !(1 << index);
        } else if index < 128 {
            self.1 &= !(1 << (index - 64));
        }
    }

    /// Sets all bits to 1.
    #[inline]
    pub fn set_all(&mut self) {
        self.0 = u64::MAX;
        self.1 = u64::MAX;
    }

    /// Sets the bit at `index` to 1. Does nothing if the index is out of bounds.
    #[inline]
    pub fn set(&mut self, index: u8) {
        if index < 64 {
            self.0 |= 1 << index;
        } else if index < 128 {
            self.1 |= 1 << (index - 64);
        }
    }

    /// Tests if the bit at `index` is 1.
    /// Returns false if the bit is 0 or index is out of bounds.
    #[inline]
    pub fn test(&self, index: u8) -> bool {
        if index < 64 {
            (self.0 & (1 << index)) != 0
        } else if index < 128 {
            (self.1 & (1 << (index - 64))) != 0
        } else {
            false
        }
    }

    /// Returns the number of set bits (population count).
    #[inline]
    pub const fn count_ones(&self) -> u32 {
        self.0.count_ones() + self.1.count_ones()
    }

    /// Returns true if no bits are set.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.0 == 0 && self.1 == 0
    }
}

// **** Bit operations ****

impl BitOr for BitSet128 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0, self.1 | rhs.1)
    }
}

impl BitAnd for BitSet128 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0, self.1 & rhs.1)
    }
}

impl BitXor for BitSet128 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0, self.1 ^ rhs.1)
    }
}

impl BitOrAssign for BitSet128 {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
        self.1 |= rhs.1;
    }
}

impl BitAndAssign for BitSet128 {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
        self.1 &= rhs.1;
    }
}

impl BitXorAssign for BitSet128 {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
        self.1 ^= rhs.1;
    }
}

impl Index<u8> for BitSet128 {
    type Output = bool;

    fn index(&self, index: u8) -> &Self::Output {
        // We use static booleans because we must return a reference
        if self.test(index) { &true } else { &false }
    }
}

impl Index<usize> for BitSet128 {
    type Output = bool;

    #[allow(clippy::cast_possible_truncation)]
    fn index(&self, index: usize) -> &Self::Output {
        // We use static booleans because we must return a reference
        if self.test(index as u8) { &true } else { &false }
    }
}

/// `BitSet128` from `u32`.
impl From<u32> for BitSet128 {
    #[inline]
    fn from(a: u32) -> Self {
        Self(u64::from(a), 0)
    }
}

/// `BitSet64` from `(u32,u32)`.
impl From<(u32, u32)> for BitSet128 {
    #[inline]
    fn from((a, b): (u32, u32)) -> Self {
        Self(u64::from(a) << 32 | u64::from(b), 0)
    }
}

// **** Iter ****

#[derive(Debug, Default, PartialEq, Eq)]
pub struct BitSet128Iter(u64, u64);

// **** fmt ****

impl fmt::Binary for BitSet128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Handle the "0b" prefix if requested via {:#b}
        if f.alternate() {
            f.write_str("0b")?;
        }
        // Print from high bits to low bits (left-to-right)
        // High bits (127 down to 64)
        for i in (0..64).rev() {
            let val = (self.1 >> i) & 1;
            write!(f, "{val}")?;
        }
        // Low bits (63 down to 0)
        for i in (0..64).rev() {
            let val = (self.0 >> i) & 1;
            write!(f, "{val}")?;
        }

        Ok(())
    }
}

impl fmt::UpperHex for BitSet128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            f.write_str("0x")?;
        }
        write!(f, "{:016X}{:016X}", self.1, self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<BitSet128>();
    }
    #[test]
    fn new() {
        let mut bits = BitSet128::new();
        bits.set(42);
        assert!(bits[42u8]);
        assert!(bits.test(42));
    }
    #[test]
    fn assign() {
        let mut bits = BitSet128::new();
        bits.set(42);
        assert!(bits[42u8]);
        assert!(bits.test(42));
        let mask = bits;
        assert!(mask.test(42));
    }
    #[test]
    fn exercise() {
        let mut system_flags = BitSet128::new();
        let error_mask = BitSet128::new(); // imagine this has error bits set

        // Combine with OR-assign
        system_flags |= error_mask;

        // Toggle bits with XOR-assign
        system_flags ^= error_mask;

        // Mask out bits with AND-assign
        //system_flags &= BitSet128(0x0000_FFFF_FFFF_FFFF);

        let mut set_a = BitSet128::new();
        set_a.set(10);
        set_a.set(20);

        let mut set_b = BitSet128::new();
        set_b.set(20);
        set_b.set(30);

        // Intersection (AND): only bit 20 remains
        let common = set_a & set_b;
        assert!(!common.test(10));
        assert!(common.test(20));
        assert!(!common.test(30));

        // Union (OR): bits 10, 20, and 30 are set
        let all = set_a | set_b;
        assert!(all.test(10));
        assert!(all.test(20));
        assert!(all.test(30));

        // Difference (XOR): bits 10 and 30 are set (20 is cancelled out)
        let diff = set_a ^ set_b;
        assert!(diff.test(10));
        assert!(!diff.test(20));
        assert!(diff.test(30));
    }
}
