use core::ops::{Mul, Neg};
use num_traits::float::FloatCore;

use crate::{MathConstants, Quaternion, TrigonometricMethods, Vector2d, Vector3d};

/// `RollPitchYaw` `struct { roll: f32, pitch: f32, yaw: f32 }`<br>
pub type RollPitchYawf32 = RollPitchYaw<f32>;
/// `RollPitchYaw` `struct { roll: f64, pitch: f64, yaw: f64 }`<br>
pub type RollPitchYawf64 = RollPitchYaw<f64>;

/// `RollPitch` `struct { roll: f32, pitch: f32 }`<br>
pub type RollPitchf32 = RollPitch<f32>;
/// `RollPitch` `struct { roll: f64, pitch: f64 }`<br><br>
pub type RollPitchf64 = RollPitch<f64>;

/// Roll and Pitch bundled for convenience.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RollPitch<T> {
    pub roll: T,
    pub pitch: T,
}

impl<T> RollPitch<T>
where
    T: Copy,
{
    /// Create a `RollPitch`.
    #[inline]
    pub const fn new(roll: T, pitch: T) -> Self {
        Self { roll, pitch }
    }

    /// Create a `RollPitch` from a `Vector2d` assuming the North East Down (NED) convention.
    #[inline]
    pub fn from_vector_ned(v: Vector2d<T>) -> Self {
        Self { roll: v.y, pitch: v.x }
    }
}

impl<T> RollPitch<T>
where
    T: Copy + Mul<Output = T> + MathConstants,
{
    #[inline]
    pub fn to_degrees(self) -> Self {
        Self { roll: self.roll * T::RADIANS_TO_DEGREES, pitch: self.pitch * T::RADIANS_TO_DEGREES }
    }
    #[inline]
    pub fn to_radians(self) -> Self {
        Self { roll: self.roll * T::DEGREES_TO_RADIANS, pitch: self.pitch * T::DEGREES_TO_RADIANS }
    }
}

impl<T> RollPitch<T>
where
    T: Copy + FloatCore,
{
    #[inline]
    pub fn abs(self) -> Self {
        Self { roll: self.roll.abs(), pitch: self.pitch.abs() }
    }
    #[inline]
    pub fn clamp(self, min: T, max: T) -> Self {
        Self { roll: self.roll.clamp(min, max), pitch: self.pitch.clamp(min, max) }
    }
}

impl<T> From<RollPitch<T>> for Quaternion<T>
where
    T: Copy + TrigonometricMethods + FloatCore,
{
    #[inline]
    fn from(angles: RollPitch<T>) -> Self {
        Quaternion::from_roll_pitch_angles_radians(angles.roll, angles.pitch)
    }
}

/// Roll, Pitch, and Yaw bundled for convenience.<br><br>
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RollPitchYaw<T> {
    pub roll: T,
    pub pitch: T,
    pub yaw: T,
}

impl<T> RollPitchYaw<T>
where
    T: Copy + Neg<Output = T>,
{
    /// Create a `RollPitchYaw`.
    #[inline]
    pub const fn new(roll: T, pitch: T, yaw: T) -> Self {
        Self { roll, pitch, yaw }
    }

    /// Create a `RollPitchYaw` from a `Vector3d` assuming the North East Down (NED) convention.
    #[inline]
    pub fn from_vector_ned(v: Vector3d<T>) -> Self {
        Self { roll: v.y, pitch: v.x, yaw: -v.z }
    }
}

impl<T> RollPitchYaw<T>
where
    T: Copy + Mul<Output = T> + MathConstants,
{
    #[inline]
    pub fn to_degrees(self) -> Self {
        Self {
            roll: self.roll * T::RADIANS_TO_DEGREES,
            pitch: self.pitch * T::RADIANS_TO_DEGREES,
            yaw: self.yaw * T::RADIANS_TO_DEGREES,
        }
    }
    #[inline]
    pub fn to_radians(self) -> Self {
        Self {
            roll: self.roll * T::DEGREES_TO_RADIANS,
            pitch: self.pitch * T::DEGREES_TO_RADIANS,
            yaw: self.yaw * T::RADIANS_TO_DEGREES,
        }
    }
}

impl<T> RollPitchYaw<T>
where
    T: Copy + FloatCore,
{
    #[inline]
    pub fn abs(self) -> Self {
        Self { roll: self.roll.abs(), pitch: self.pitch.abs(), yaw: self.yaw.abs() }
    }
    #[inline]
    pub fn clamp(self, min: T, max: T) -> Self {
        Self { roll: self.roll.clamp(min, max), pitch: self.pitch.clamp(min, max), yaw: self.yaw.clamp(min, max) }
    }
}

impl<T> From<RollPitchYaw<T>> for Quaternion<T>
where
    T: Copy + TrigonometricMethods + FloatCore,
{
    #[inline]
    fn from(angles: RollPitchYaw<T>) -> Self {
        Quaternion::from_roll_pitch_yaw_angles_radians(angles.roll, angles.pitch, angles.yaw)
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use super::*;
    #[cfg(feature = "serde")]
    use serde::{Deserialize, Serialize};

    fn _is_normal<T: Sized + Send + Sync + Unpin>() {}
    fn is_full<T: Sized + Send + Sync + Unpin + Copy + Clone + Default + PartialEq>() {}

    #[test]
    fn normal_types() {
        is_full::<RollPitchf32>();
        is_full::<RollPitchYawf32>();
    }
}
