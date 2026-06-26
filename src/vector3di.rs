#[cfg(not(feature = "uom"))]
use core::ops::Mul;
use core::{error::Error, fmt};

use num_traits::MulAdd;

use crate::{Vector3d, Vector3df32};

/// 3-dimensional `{x, y, z}` vector of `i16` values<br>
pub type Vector3di16 = Vector3d<i16>;

#[cfg(not(feature = "uom"))]
impl Mul<f32> for Vector3d<i16> {
    type Output = Self;

    #[inline]
    fn mul(self, k: f32) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self { x: (f32::from(self.x) * k) as i16, y: (f32::from(self.y) * k) as i16, z: (f32::from(self.z) * k) as i16 }
    }
}

impl MulAdd<f32> for Vector3d<i16> {
    type Output = Self;

    #[inline]
    fn mul_add(self, k: f32, other: Self) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Vector3d {
            x: self.x * (k as i16) + other.x,
            y: self.y * (k as i16) + other.y,
            z: self.z * (k as i16) + other.z,
        }
    }
}

impl MulAdd<i32> for Vector3d<i16> {
    type Output = Self;

    #[inline]
    fn mul_add(self, k: i32, other: Self) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Vector3d {
            x: self.x * (k as i16) + other.x,
            y: self.y * (k as i16) + other.y,
            z: self.z * (k as i16) + other.z,
        }
    }
}

/// The error type returned when a slice is too short to parse a `Vector3di16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceTooShortError;

// Implement Display so it can be printed
impl fmt::Display for SliceTooShortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "source slice must contain at least 6 bytes")
    }
}

/// Implement the standard Error trait.
impl Error for SliceTooShortError {}

impl Vector3di16 {
    /// Creates a Vector3di16 from a 6-byte little-endian array.
    /// ```
    /// # use vqm::Vector3di16;
    /// let bytes = [0x01, 0x00, 0x00, 0x01, 0x2A, 0x00];
    /// let v = Vector3di16::from_le_bytes(bytes);
    /// assert_eq!(Vector3di16 { x: 1, y: 256, z: 42 }, v);
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_le_bytes(buf: [u8; 6]) -> Self {
        Self {
            x: i16::from_le_bytes([buf[0], buf[1]]),
            y: i16::from_le_bytes([buf[2], buf[3]]),
            z: i16::from_le_bytes([buf[4], buf[5]]),
        }
    }

    /// Creates a Vector3di16 from a little-endian slice.
    /// ```
    /// # use vqm::Vector3di16;
    /// let bytes = [0x01, 0x00, 0x00, 0x01, 0x2A, 0x00];
    /// let v = Vector3di16::from_le_slice(&bytes);
    /// assert_eq!(Vector3di16 { x: 1, y: 256, z: 42 }, v);
    /// ```
    /// # Panics
    /// Panics if `slice.len() < 6`.
    #[inline]
    #[must_use]
    pub fn from_le_slice(slice: &[u8]) -> Self {
        let chunk = &slice[0..6];
        Self {
            x: i16::from_le_bytes([chunk[0], chunk[1]]),
            y: i16::from_le_bytes([chunk[2], chunk[3]]),
            z: i16::from_le_bytes([chunk[4], chunk[5]]),
        }
    }
    /*
    suppose I have:

    let bytes = [0x01, 0x00, 0x00, 0x01, 0x2A, 0x00, 0x23, 0x45, 0x56, 0x78, 0x9a, 0xbc];

    How would I create two Vector3di16 from this using from_le_slice, the first vector starting at byte 0 and the second starting at byte 6
    */
    /// Creates a `Vector3di16` from a little-endian byte slice.
    /// ```
    /// # use vqm::Vector3di16;
    /// let bytes = [0x01, 0x00, 0x00, 0x01, 0x2A, 0x00, 0x23, 0x45, 0x56, 0x78, 0x9a, 0xbc];
    /// let v = Vector3di16::try_from_le_slice(&bytes);
    /// assert_eq!(Vector3di16 { x: 1, y: 256, z: 42 }, v.expect("REASON"));
    /// ```
    /// # Errors
    /// Returns an [`InvalidLengthError`] if the provided `slice` contains fewer than 6 bytes.
    pub fn try_from_le_slice(slice: &[u8]) -> Result<Self, SliceTooShortError> {
        if slice.len() < 6 {
            return Err(SliceTooShortError);
        }
        let chunk = &slice[0..6];
        Ok(Self {
            x: i16::from_le_bytes([chunk[0], chunk[1]]),
            y: i16::from_le_bytes([chunk[2], chunk[3]]),
            z: i16::from_le_bytes([chunk[4], chunk[5]]),
        })
    }

    /// Creates a 6-byte little-endian array from a Vector3di16.
    /// ```
    /// # use vqm::Vector3di16;
    /// let v = Vector3di16 { x: 1, y: 256, z: 42 };
    /// let bytes = v.to_le_bytes();
    /// assert_eq!([0x01, 0x00, 0x00, 0x01, 0x2A, 0x00], bytes);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_le_bytes(&self) -> [u8; 6] {
        let x = self.x.to_le_bytes();
        let y = self.y.to_le_bytes();
        let z = self.z.to_le_bytes();
        [x[0], x[1], y[0], y[1], z[0], z[1]]
    }

    /// Creates a Vector3di16 from a 6-byte big-endian array.
    /// ```
    /// # use vqm::Vector3di16;
    /// let bytes = [0x00, 0x01, 0x01, 0x00, 0x00, 0x2A];
    /// let v = Vector3di16::from_be_bytes(bytes);
    /// assert_eq!(Vector3di16 { x: 1, y: 256, z: 42 }, v);
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_be_bytes(buf: [u8; 6]) -> Self {
        Self {
            x: i16::from_be_bytes([buf[0], buf[1]]),
            y: i16::from_be_bytes([buf[2], buf[3]]),
            z: i16::from_be_bytes([buf[4], buf[5]]),
        }
    }

    /// Creates a Vector3di16 from a big-endian slice.
    /// ```
    /// # use vqm::Vector3di16;
    /// let bytes = [0x00, 0x01, 0x01, 0x00, 0x00, 0x2A];
    /// let v = Vector3di16::from_be_slice(&bytes);
    /// assert_eq!(Vector3di16 { x: 1, y: 256, z: 42 }, v);
    /// ```
    /// # Panics
    /// Panics if `slice.len() < 6`.
    #[inline]
    #[must_use]
    pub fn from_be_slice(slice: &[u8]) -> Self {
        let chunk = &slice[0..6];
        Self {
            x: i16::from_be_bytes([chunk[0], chunk[1]]),
            y: i16::from_be_bytes([chunk[2], chunk[3]]),
            z: i16::from_be_bytes([chunk[4], chunk[5]]),
        }
    }

    /// Creates a `Vector3di16` from a big-endian byte slice.
    /// ```
    /// # use vqm::Vector3di16;
    /// let bytes = [0x00, 0x01, 0x01, 0x00, 0x00, 0x2A];
    /// let v = Vector3di16::try_from_be_slice(&bytes);
    /// assert_eq!(Vector3di16 { x: 1, y: 256, z: 42 }, v.expect("REASON"));
    /// ```
    /// # Errors
    /// Returns an [`InvalidLengthError`] if the provided `slice` contains fewer than 6 bytes.
    pub fn try_from_be_slice(slice: &[u8]) -> Result<Self, SliceTooShortError> {
        if slice.len() < 6 {
            return Err(SliceTooShortError);
        }
        let chunk = &slice[0..6];
        Ok(Self {
            x: i16::from_be_bytes([chunk[0], chunk[1]]),
            y: i16::from_be_bytes([chunk[2], chunk[3]]),
            z: i16::from_be_bytes([chunk[4], chunk[5]]),
        })
    }

    /// Creates a 6-byte little-endian array from a Vector3di16.
    /// ```
    /// # use vqm::Vector3di16;
    /// let v = Vector3di16 { x: 1, y: 256, z: 42 };
    /// let bytes = v.to_be_bytes();
    /// assert_eq!([0x00, 0x01, 0x01, 0x00, 0x00, 0x2A], bytes);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_be_bytes(&self) -> [u8; 6] {
        let x = self.x.to_be_bytes();
        let y = self.y.to_be_bytes();
        let z = self.z.to_be_bytes();
        [x[0], x[1], y[0], y[1], z[0], z[1]]
    }
}

impl From<Vector3d<i16>> for Vector3d<f32> {
    /// `Vector3d<f32>` from `Vector3d<i16>`.
    /// ```
    /// # use vqm::{Vector3df32, Vector3di16};
    /// let v_i16 = Vector3di16{x: 2, y: 3, z: 5};
    /// let v_f32 = Vector3df32::from(v_i16);
    ///
    /// let w_f32 = Vector3df32{x: 7.0, y: 11.0, z: 13.0};
    /// let w_i16 : Vector3di16 = w_f32.into();
    ///
    /// assert_eq!(v_f32, Vector3df32 { x: 2.0, y: 3.0, z: 5.0 });
    /// assert_eq!(w_i16, Vector3di16 { x: 7, y: 11, z: 13 });
    /// ```
    #[inline]
    fn from(v: Vector3d<i16>) -> Self {
        Self { x: f32::from(v.x), y: f32::from(v.y), z: f32::from(v.z) }
    }
}

impl From<Vector3d<f32>> for Vector3d<i16> {
    #[inline]
    fn from(v: Vector3d<f32>) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self { x: v.x as i16, y: v.y as i16, z: v.z as i16 }
    }
}

impl Vector3df32 {
    /// Creates a Vector3df32 from a 6-byte little-endian array.
    /// ```
    /// # use vqm::Vector3df32;
    /// let bytes = [0x01, 0x00, 0x00, 0x01, 0x2A, 0x00];
    /// let v = Vector3df32::from_le_bytes_6(bytes);
    /// assert_eq!(Vector3df32 { x: 1.0, y: 256.0, z: 42.0 }, v);
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_le_bytes_6(buf: [u8; 6]) -> Self {
        let v = Vector3di16::from_le_bytes(buf);
        Self { x: v.x as f32, y: v.y as f32, z: v.z as f32 }
    }

    /// Creates a Vector3df32 from a  little-endian slice.
    /// ```
    /// # use vqm::Vector3df32;
    /// let bytes = [0x01, 0x00, 0x00, 0x01, 0x2A, 0x00];
    /// let v = Vector3df32::from_le_slice_6(&bytes);
    /// assert_eq!(Vector3df32 { x: 1.0, y: 256.0, z: 42.0 }, v);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_le_slice_6(slice: &[u8]) -> Self {
        let v = Vector3di16::from_le_slice(slice);
        Self { x: f32::from(v.x), y: f32::from(v.y), z: f32::from(v.z) }
    }

    /// Creates a Vector3df32 from a 6-byte big-endian array.
    /// ```
    /// # use vqm::Vector3df32;
    /// let bytes = [0x00, 0x01, 0x01, 0x00, 0x00, 0x2A];
    /// let v = Vector3df32::from_be_bytes_6(bytes);
    /// assert_eq!(Vector3df32 { x: 1.0, y: 256.0, z: 42.0 }, v);
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_be_bytes_6(buf: [u8; 6]) -> Self {
        let v = Vector3di16::from_be_bytes(buf);
        Self { x: v.x as f32, y: v.y as f32, z: v.z as f32 }
    }
}

// **** Vector3di32 ****

/// 3-dimensional `{x, y, z}` vector of `i32` values<br><br>
pub type Vector3di32 = Vector3d<i32>;

#[cfg(not(feature = "uom"))]
impl Mul<f32> for Vector3d<i32> {
    type Output = Self;

    #[inline]
    fn mul(self, k: f32) -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        Self { x: ((self.x as f32) * k) as i32, y: ((self.y as f32) * k) as i32, z: ((self.z as f32) * k) as i32 }
    }
}

impl From<Vector3d<i32>> for Vector3d<f32> {
    /// `Vector3d<f32>` from `Vector3d<i32>`.
    /// ```
    /// # use vqm::{Vector3df32,Vector3di32};
    /// let v_i32 = Vector3di32{x: 2, y: 3, z: 5};
    /// let v_f32 = Vector3df32::from(v_i32);
    ///
    /// let w_f32 = Vector3df32{x: 7.0, y: 11.0, z: 13.0};
    /// let w_i32 : Vector3di32 = w_f32.into();
    ///
    /// assert_eq!(v_f32, Vector3df32 { x: 2.0, y: 3.0, z: 5.0 });
    /// assert_eq!(w_i32, Vector3di32 { x: 7, y: 11, z: 13 });
    /// ```
    #[inline]
    fn from(v: Vector3d<i32>) -> Self {
        #[allow(clippy::cast_precision_loss)]
        Self { x: v.x as f32, y: v.y as f32, z: v.z as f32 }
    }
}

impl From<Vector3d<f32>> for Vector3d<i32> {
    #[inline]
    fn from(v: Vector3d<f32>) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self { x: v.x as i32, y: v.y as i32, z: v.z as i32 }
    }
}

/*pub fn acc_gyro(buf: [u8; 12]) -> (Vector3di16, Vector3di16) {
    // Split the 12-byte array into two fixed 6-byte arrays with zero room for error
    let (acc_bytes, gyro_bytes) = buf.split_at(6);

    // Convert the slices to fixed arrays securely without any manual index copying
    let acc = Vector3di16::from_le_slice(acc_bytes);
    let gyro = Vector3di16::from_le_slice(gyro_bytes);

    (acc, gyro)
}*/
