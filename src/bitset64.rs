use core::cmp::Ordering;
use core::fmt;
use core::iter::FromIterator;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index};

/// A memory-efficient 64-bit set for embedded environments.
/// Note that it data is a one-tuple: this makes comparison with `BitSet128` clearer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitSet64(u64);

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
}

impl BitSet64 {
    /// Resets all bits to 0.
    #[inline]
    pub fn reset_all(&mut self) {
        self.0 = 0;
    }

    /// Resets the bit at `index` to 0. Does nothing if the index is out of bounds.
    #[inline]
    pub fn reset(&mut self, index: u8) {
        if index < 64 {
            self.0 &= !(1 << index);
        }
    }

    /// Sets all bits to 1.
    #[inline]
    pub fn set_all(&mut self) {
        self.0 = u64::MAX;
    }

    /// Sets the bit at `index` to 1. Does nothing if the index is out of bounds.
    #[inline]
    pub fn set(&mut self, index: u8) {
        if index < 64 {
            self.0 |= 1 << index;
        }
    }

    /// Tests if the bit at `index` is 1.
    /// Returns false if the bit is 0 or index is out of bounds.
    #[inline]
    pub fn test(&self, index: u8) -> bool {
        if index < 64 { (self.0 & (1 << index)) != 0 } else { false }
    }

    /// Returns the number of set bits (population count).
    #[inline]
    pub const fn count_ones(&self) -> u32 {
        self.0.count_ones()
    }

    /// Returns the number of leading zeros.
    #[inline]
    pub const fn leading_zeros(&self) -> u32 {
        self.0.leading_zeros()
    }

    /// Returns true if no bits are set.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Returns the highest index set, or None if the bitset is empty.
    /// Useful for finding the "top" of a priority queue or resource map.
    pub fn last_set(&self) -> Option<u8> {
        if self.is_empty() {
            None
        } else {
            #[allow(clippy::cast_possible_truncation)]
            Some(63 - self.0.leading_zeros() as u8)
        }
    }

    /// Returns true if this bitset contains all the bits set in `other`.
    #[inline]
    pub fn is_superset(&self, other: &BitSet64) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Returns true if this bitset is a subset of `other`.
    #[inline]
    pub fn is_subset(&self, other: &BitSet64) -> bool {
        other.is_superset(self)
    }
}

impl PartialOrd for BitSet64 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BitSet64 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // We delegate the comparison to the underlying u64
        self.0.cmp(&other.0)
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

/// `BitSet64` from `u32`.
/// ```
/// # use vqm::BitSet64;
/// ```
impl From<u32> for BitSet64 {
    #[inline]
    fn from(a: u32) -> Self {
        Self(u64::from(a))
    }
}

/// `BitSet64` from `(u32,u32)`.
/// ```
/// # use vqm::BitSet64;
/// ```
impl From<(u32, u32)> for BitSet64 {
    #[inline]
    fn from((a, b): (u32, u32)) -> Self {
        Self(u64::from(a) << 32 | u64::from(b))
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct BitSet64Iter(u64);

/// Consuming iterator.
impl Iterator for BitSet64Iter {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            None
        } else {
            // Find the index of the least significant bit set to 1
            #[allow(clippy::cast_possible_truncation)]
            let index = self.0.trailing_zeros() as u8;
            // Clear the least significant bit to prep for next iteration
            self.0 &= self.0 - 1;
            Some(index)
        }
    }
}

/// Non-consuming iterator.
impl IntoIterator for &BitSet64 {
    type Item = u8;
    type IntoIter = BitSet64Iter;

    fn into_iter(self) -> Self::IntoIter {
        // Since BitSet64 is Copy and just a u64,
        // we just pass the underlying value to the iterator.
        BitSet64Iter(self.0)
    }
}

impl BitSet64 {
    /// Returns an iterator over the indices of the set bits.
    pub fn iter(&self) -> BitSet64Iter {
        self.into_iter()
    }
}

impl FromIterator<u8> for BitSet64 {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        let mut bitset = Self::new();
        for index in iter {
            bitset.set(index);
        }
        bitset
    }
}

impl Extend<u8> for BitSet64 {
    fn extend<I: IntoIterator<Item = u8>>(&mut self, iter: I) {
        for index in iter {
            self.set(index);
        }
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
        bits.set(42);
        assert!(bits[42u8]);
        assert!(bits.test(42));
    }
    #[allow(unused)]
    #[test]
    fn const_new() {
        const FLAGS: BitSet64 = BitSet64::new();
        const EMPTY_CHECK: bool = FLAGS.is_empty(); // Evaluated at compile time
    }
    #[test]
    fn assign() {
        let mut bits = BitSet64::new();
        bits.set(42);
        assert!(bits[42u8]);
        assert!(bits.test(42));
        let mask = bits;
        assert!(mask.test(42));
    }
    #[test]
    fn from() {
        let _bits = BitSet64::from((0xab_u32, 0x12_u32));
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
        set_a.set(10);
        set_a.set(20);

        let mut set_b = BitSet64::new();
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
    #[test]
    fn iterator_consuming() {
        let mut bits = BitSet64::new();
        bits.set(2);
        bits.set(10);

        let mut sum = 0;
        let mut count = 0;
        for bit in &bits {
            sum += bit;
            count += 1;
        }
        assert_eq!(2, count);
        assert_eq!(12, sum);
    }
    #[test]
    fn into_iter_consuming() {
        let mut bits = BitSet64::new();
        bits.set(0);
        bits.set(10);
        bits.set(63);

        // Consuming iterator
        let mut iter = bits.into_iter();

        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(10));
        assert_eq!(iter.next(), Some(63));
        assert_eq!(iter.next(), None);

        // bits is moved here and can no longer be used
    }

    #[test]
    fn non_consuming_iterator() {
        let bits = BitSet64(0b1101); // Bits 0, 2, and 3 are set

        let mut sum = 0;
        let mut count = 0;

        // Use reference to bits rather than bits.iter()
        for bit in &bits {
            sum += bit;
            count += 1;
        }
        assert_eq!(3, count);
        assert_eq!(5, sum);

        let mut sum = 0;
        let mut count = 0;
        // Using a reference in a loop
        for bit in &bits {
            sum += bit;
            count += 1;
        }
        assert_eq!(3, count);
        assert_eq!(5, sum);
    }

    #[test]
    fn non_consuming_iterator2() {
        let mut bits = BitSet64::new();
        bits.set(5);
        bits.set(12);

        // Test using the .iter() method
        let count = bits.iter().count();
        assert_eq!(count, 2);

        // Test using & reference in a for-loop
        let mut last_val = 0;
        for bit in &bits {
            last_val = bit;
        }
        assert_eq!(last_val, 12);

        // test original bits is still valid
        assert!(bits.test(5));
    }

    #[test]
    fn from_iterator() {
        let indices = [1, 3, 5];
        // collect() uses FromIterator
        let bits: BitSet64 = indices.iter().copied().collect();

        assert!(bits.test(1));
        assert!(bits.test(3));
        assert!(bits.test(5));
        assert!(!bits.test(2));
    }

    #[test]
    fn empty_and_full() {
        let empty = BitSet64::new();
        assert_eq!(empty.iter().count(), 0);

        let mut full = BitSet64::new();
        full.set_all();
        assert_eq!(full.iter().count(), 64);
    }
}
