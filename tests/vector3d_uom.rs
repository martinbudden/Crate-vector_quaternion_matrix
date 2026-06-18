#![cfg(feature = "uom")]

#[cfg(test)]
mod tests {
    //use super::*;
    use uom::si::f32::{Area, Length, Ratio, Time, Velocity};
    use uom::si::ratio::ratio;
    use uom::si::time::second;
    use uom::si::velocity::meter_per_second;
    use uom::si::{area::square_meter, length::meter};
    use vqm::Vector3d;

    /*#[test]
    fn default() {
        let a: Vector3d<Length::new<meter>> = Vector3d::<Length::new<meter>>::default();
        let zero_meters = Length::new::<meter>(0.0);
        assert_eq!(a, Vector3d { x: zero_meters, y: zero_meters, z: zero_meters });
    }*/
    #[test]
    fn neg() {
        let a = Vector3d { x: Length::new::<meter>(2.0), y: Length::new::<meter>(5.0), z: Length::new::<meter>(11.0) };
        assert_eq!(
            -a,
            Vector3d { x: -Length::new::<meter>(2.0), y: -Length::new::<meter>(5.0), z: -Length::new::<meter>(11.0) }
        );

        let b = -a;
        assert_eq!(
            b,
            Vector3d { x: -Length::new::<meter>(2.0), y: -Length::new::<meter>(5.0), z: -Length::new::<meter>(11.0) }
        );
    }
    #[test]
    fn add() {
        let a = Vector3d { x: Length::new::<meter>(2.0), y: Length::new::<meter>(5.0), z: Length::new::<meter>(11.0) };
        let b = Vector3d { x: Length::new::<meter>(3.0), y: Length::new::<meter>(7.0), z: Length::new::<meter>(13.0) };
        let c = a + b;

        assert_eq!(
            c,
            Vector3d { x: Length::new::<meter>(5.0), y: Length::new::<meter>(12.0), z: Length::new::<meter>(24.0) }
        );
    }
    #[test]
    fn mul_scalar() {
        let a = Vector3d { x: Length::new::<meter>(2.0), y: Length::new::<meter>(5.0), z: Length::new::<meter>(11.0) };
        let k = Length::new::<meter>(3.0);
        let b = a * k;

        assert_eq!(
            b,
            Vector3d {
                x: Area::new::<square_meter>(6.0),
                y: Area::new::<square_meter>(15.0),
                z: Area::new::<square_meter>(33.0)
            }
        );
    }
    #[test]
    fn div_scalar() {
        let a = Vector3d { x: Length::new::<meter>(2.0), y: Length::new::<meter>(5.0), z: Length::new::<meter>(11.0) };
        let k = Length::new::<meter>(4.0);
        let b = a / k;

        assert_eq!(
            b,
            Vector3d { x: Ratio::new::<ratio>(0.5), y: Ratio::new::<ratio>(1.25), z: Ratio::new::<ratio>(2.75) }
        );
        let t = Time::new::<second>(4.0);
        let c = a / t;

        // The assertions must use Velocity and meter_per_second
        assert_eq!(
            c,
            Vector3d {
                x: Velocity::new::<meter_per_second>(0.5),
                y: Velocity::new::<meter_per_second>(1.25),
                z: Velocity::new::<meter_per_second>(2.75)
            }
        );
    }
}
