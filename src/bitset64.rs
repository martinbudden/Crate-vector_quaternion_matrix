use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index};

/// A memory-efficient 64-bit set for embedded environments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitSet64(u64);

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BitSet64Iter {
    bits: u64,
    current_bit: u8,
}

impl Default for BitSet64 {
    fn default() -> Self {
        Self::new()
    }
}

impl BitSet64 {
    /// Create a new empty bitset.
    pub const fn new() -> Self {
        Self(0)
    }

    /// Resets all bits to 0.
    pub fn reset_all(&mut self) {
        self.0 = 0;
    }

    /// Resets the bit at `index` to 0.
    /// Returns false if index is out of bounds (>= 64).
    pub fn reset(&mut self, index: u8) -> bool {
        if index < 64 {
            self.0 &= !(1 << index);
            true
        } else {
            false
        }
    }

    /// Sets the bit at `index` to 1.
    /// Returns false if index is out of bounds (>= 64).
    pub fn set(&mut self, index: u8) -> bool {
        if index < 64 {
            self.0 |= 1 << index;
            true
        } else {
            false
        }
    }

    /// Tests if the bit at `index` is 1.
    /// Returns false if the bit is 0 or index is out of bounds.
    pub fn test(&self, index: u8) -> bool {
        if index < 64 { (self.0 & (1 << index)) != 0 } else { false }
    }
    /// Returns an iterator over the indices of all bits that are set to 1.
    #[allow(clippy::iter_without_into_iter)] // TODO: fix this iterator
    pub fn iter(&self) -> BitSet64Iter {
        BitSet64Iter { bits: self.0, current_bit: 0 }
    }
}

impl BitOr for BitSet64 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitAnd for BitSet64 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitXor for BitSet64 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl BitOrAssign for BitSet64 {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAndAssign for BitSet64 {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitXorAssign for BitSet64 {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Index<u8> for BitSet64 {
    type Output = bool;

    fn index(&self, index: u8) -> &Self::Output {
        // We use static booleans because we must return a reference
        if self.test(index) { &true } else { &false }
    }
}
impl Index<usize> for BitSet64 {
    type Output = bool;

    #[allow(clippy::cast_possible_truncation)]
    fn index(&self, index: usize) -> &Self::Output {
        // We use static booleans because we must return a reference
        if self.test(index as u8) { &true } else { &false }
    }
}

/*impl Iterator for BitSet64Iter {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_bit < 64 {
            let bit = self.current_bit;
            self.current_bit += 1;

            if (self.bits & (1 << bit)) != 0 {
                return Some(bit);
            }
        }
        None
    }
}*/

#[allow(clippy::copy_iterator)] // TODO: fix this iterator
#[allow(clippy::cast_possible_truncation)]
impl Iterator for BitSet64Iter {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bits == 0 {
            return None;
        }

        // Find the index of the next 1-bit
        let bit = self.bits.trailing_zeros() as u8;

        if bit >= 64 {
            self.bits = 0;
            return None;
        }

        // Clear that bit so we can find the next one
        self.bits &= !(1 << bit);
        Some(bit)
    }
}

impl fmt::Binary for BitSet64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Handle the "0b" prefix if requested via {:#b}
        if f.alternate() {
            f.write_str("0b")?;
        }
        // Print from high bits to low bits (left-to-right)
        for i in (0..64).rev() {
            let val = (self.0 >> i) & 1;
            write!(f, "{val}")?;
        }

        Ok(())
    }
}

impl fmt::UpperHex for BitSet64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            f.write_str("0x")?;
        }
        write!(f, "{:016X}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<BitSet64>();
        is_normal::<BitSet64Iter>();
    }
    #[test]
    fn new() {
        let mut bits = BitSet64::new();
        _ = bits.set(42);
        assert!(bits[42u8]);
        assert!(bits.test(42));
    }
    #[test]
    fn assign() {
        let mut bits = BitSet64::new();
        _ = bits.set(42);
        assert!(bits[42u8]);
        assert!(bits.test(42));
        let mask = bits;
        assert!(mask.test(42));
    }
    #[test]
    fn exercise() {
        let mut system_flags = BitSet64::new();
        let error_mask = BitSet64::new(); // imagine this has error bits set

        // Combine with OR-assign
        system_flags |= error_mask;

        // Toggle bits with XOR-assign
        system_flags ^= error_mask;

        // Mask out bits with AND-assign
        system_flags &= BitSet64(0x0000_FFFF_FFFF_FFFF);

        let mut set_a = BitSet64::new();
        _ = set_a.set(10);
        _ = set_a.set(20);

        let mut set_b = BitSet64::new();
        _ = set_b.set(20);
        _ = set_b.set(30);

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
