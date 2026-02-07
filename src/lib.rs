#![no_std]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(unused_must_use)]

mod math_methods;
mod matrix3x3;
mod quaternion;
mod vector3d;
mod vector3d_i16;
mod vector3d_i32;

pub use math_methods::MathMethods;
pub use matrix3x3::Matrix3x3;
pub use quaternion::Quaternion;
pub use vector3d::Vector3d;
pub use vector3d::Vector3di8;
pub use vector3d::Vector3di16;
pub use vector3d::Vector3di32;
pub use vector3d::Vector3df32;
pub use vector3d::Vector3df64;
pub use vector3d_i16::Vector3dI16;
pub use vector3d_i32::Vector3dI32;
