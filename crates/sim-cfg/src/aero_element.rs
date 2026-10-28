//! 3D **AeroElement** surface discretizer.
//!
//! Turns a [`WingGeometry`] planform into a grid of spanwise × chordwise
//! aerodynamic elements. Each element carries the data the 6-DOF solver
//! needs to evaluate a local lift/drag/moment contribution with a
//! lifting-line (strip) method: element area, local chord, centre point
//! in body coordinates, and the local effective incidence (root
//! incidence + linear washout + geometric twist at that station).
//!
//! Body axes: **+X forward, +Y right, +Z down** (classical aeronautical
//! body frame; positive pitch = nose up about +Y, positive roll = right
//! wing down about +X, positive yaw = nose right about +Z).

use crate::geometry::WingGeometry;
use glam::DVec3;

/// One discrete aerodynamic element (a strip patch of a lifting surface).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AeroElement {
    /// Index of the surface this element came from (stable per
    /// [`discretize`] call ordering).
    pub surface_index: u8,
    /// Element planform area, m².
    pub area_m2: f64,
    /// Local chord at the element centre, m.
    pub chord_m: f64,
    /// Spanwise width of the element (perpendicular to the body X axis), m.
    pub span_m: f64,
    /// Span station `t ∈ (0, 1)` measured root→tip (informational).
    pub span_frac: f64,
    /// +1 for the primary side, −1 for the mirrored side.
    pub side: f64,
    /// Unit span-axis of this element in body coordinates (root→tip
    /// direction after mirroring: `(0,1,0)` right wing, `(0,-1,0)` left
    /// wing, `(0,0,-1)` fin).
    pub axis: DVec3,
    /// Body-frame centre of the element's quarter chord, m.
    pub quarter_chord_center: DVec3,
    /// Body-frame centre of the element area (quarter-chord plane), m.
    pub center: DVec3,
    /// Local incidence: root incidence + twist at this station, radians.
    /// Positive raises the leading edge (increases local angle of attack).
    pub incidence_rad: f64,
    /// Spanwise station measured in metres from the symmetry plane.
    pub span_pos_m: f64,
}

/// Discretization controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Discretization {
    /// Chordwise elements per strip (≥ 1). More strips resolve the
    /// chordwise pressure distribution input to moment integration.
    pub chordwise: u32,
}

impl Default for Discretization {
    fn default() -> Self {
        Self { chordwise: 1 }
    }
}

/// Discretize `surface` into `spanwise` strips per generated side and
/// `disc.chordwise` chordwise elements per strip.
///
/// Axis dispatch (documented behaviour of this helper):
///
/// * `vtail` — the "span" is the fin height along **body −Z** (upward);
///   a single set of elements is produced on the symmetry plane.
/// * any other surface — the span runs along **±Y**; when `symmetric` is
///   `true` both sides are produced (right first, then the mirror),
///   otherwise only the +Y side.
///
/// Element total area always sums to the surface's authored trapezoid
/// area per generated side (equal-Δy strips with trapezoidal chord
/// interpolation integrate a linear taper exactly).
pub fn discretize(
    surface: &WingGeometry,
    surface_index: u8,
    spanwise: u32,
    disc: Discretization,
    symmetric: bool,
) -> Vec<AeroElement> {
    assert!(spanwise >= 1, "at least one spanwise strip");
    assert!(disc.chordwise >= 1, "at least one chordwise element");
    let (axis, mirror) = if surface.name == "vtail" {
        (DVec3::new(0.0, 0.0, -1.0), false)
    } else {
        (DVec3::new(0.0, 1.0, 0.0), symmetric)
    };

    // Full extent along the surface's own span axis.
    let extent = surface.span_m * if mirror { 0.5 } else { 1.0 };
    let chordwise = disc.chordwise as usize;
    let mut out =
        Vec::with_capacity(spanwise as usize * chordwise * if mirror { 2 } else { 1 });

    // Side multiplier: +1 primary, −1 mirrored (only for the ±Y axis).
    let sides: &[f64] = if mirror && axis.y != 0.0 { &[1.0, -1.0] } else { &[1.0] };
    for &side in sides {
        for i in 0..spanwise {
            let t0 = i as f64 / spanwise as f64;
            let t1 = (i + 1) as f64 / spanwise as f64;
            let tm = 0.5 * (t0 + t1);
            let chord = surface.chord_at(tm);
            let strip_width = extent / spanwise as f64;
            let strip_area = chord * strip_width;
            let s = tm * extent; // distance along the span axis
            let incidence = surface.incidence_rad + surface.washout_rad * tm;

            // Root attach point shifted along the (possibly mirrored)
            // span axis for this side.
            let eff_axis = axis * side;
            let attach = surface.position_m + eff_axis * s;
            // Sweep moves the quarter-chord aft; dihedral (±Y spans only)
            // moves the tip up (−Z).
            let x_qc = attach.x + surface.sweep_rad.tan() * s;
            let z_qc = attach.z - if axis.y != 0.0 { surface.dihedral_rad.tan() * s } else { 0.0 };
            let y_axis_offset = attach.y;

            for j in 0..chordwise {
                // element centre along the chord; LE sits at −0.25c from
                // the quarter-chord line, TE at +0.75c.
                let frac_along_chord = (j as f64 + 0.5) / chordwise as f64;
                let x_center = x_qc + chord * (frac_along_chord - 0.25);
                let elem_chord = chord / chordwise as f64;
                let elem_area = strip_area / chordwise as f64;
                out.push(AeroElement {
                    surface_index,
                    area_m2: elem_area,
                    chord_m: elem_chord,
                    span_m: strip_width,
                    span_frac: tm,
                    side,
                    axis: eff_axis,
                    quarter_chord_center: DVec3::new(x_qc, y_axis_offset, z_qc),
                    center: DVec3::new(x_center, y_axis_offset, z_qc),
                    incidence_rad: incidence,
                    span_pos_m: s,
                });
            }
        }
    }
    out
}

impl WingGeometry {
    /// Convenience: discretize this surface with the given resolution.
    pub fn discretize(
        &self,
        surface_index: u8,
        spanwise: u32,
        chordwise: u32,
        symmetric: bool,
    ) -> Vec<AeroElement> {
        discretize(
            self,
            surface_index,
            spanwise,
            Discretization { chordwise },
            symmetric,
        )
    }
}

/// Sum of element areas (verification helper).
pub fn total_area(elements: &[AeroElement]) -> f64 {
    elements.iter().map(|e| e.area_m2).sum()
}

/// Area-weighted centroid of all elements (verification helper).
pub fn area_centroid(elements: &[AeroElement]) -> DVec3 {
    let mut acc = DVec3::ZERO;
    let mut area = 0.0;
    for e in elements {
        acc += e.center * e.area_m2;
        area += e.area_m2;
    }
    if area > 0.0 {
        acc / area
    } else {
        DVec3::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect() -> WingGeometry {
        WingGeometry {
            name: "wing",
            span_m: 12.0,
            root_chord_m: 2.0,
            tip_chord_m: 2.0,
            sweep_rad: 0.0,
            dihedral_rad: 0.0,
            incidence_rad: 1.5f64.to_radians(),
            washout_rad: 0.0,
            area_m2: 24.0,
            position_m: DVec3::new(0.0, 0.0, 0.0),
        }
    }

    #[test]
    fn areas_sum_to_planform() {
        let w = rect();
        let els = discretize(&w, 0, 8, Discretization::default(), true);
        assert_eq!(els.len(), 16);
        assert!((total_area(&els) - w.area_m2).abs() < 1e-9);
        // trapezoidal taper integrates exactly to its own planform
        let trap = WingGeometry {
            tip_chord_m: 1.2,
            area_m2: (2.0 + 1.2) * 0.5 * 12.0, // 19.2 m^2 trapezoid
            ..rect()
        };
        let els = discretize(&trap, 0, 16, Discretization::default(), true);
        assert!((total_area(&els) - trap.area_m2).abs() < 1e-9);
    }

    #[test]
    fn chordwise_split_preserves_area_and_narrows_chords() {
        let w = rect();
        let els = discretize(&w, 0, 4, Discretization { chordwise: 3 }, true);
        assert_eq!(els.len(), 4 * 3 * 2);
        assert!((total_area(&els) - w.area_m2).abs() < 1e-9);
        for e in &els {
            assert!((e.chord_m - 2.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn mirror_sides_are_symmetric() {
        let w = rect();
        let els = discretize(&w, 0, 6, Discretization::default(), true);
        let right: Vec<_> = els.iter().filter(|e| e.side > 0.0).collect();
        let left: Vec<_> = els.iter().filter(|e| e.side < 0.0).collect();
        assert_eq!(right.len(), left.len());
        for (a, b) in right.iter().zip(left.iter()) {
            assert!((a.area_m2 - b.area_m2).abs() < 1e-12);
            assert!((a.center.y + b.center.y).abs() < 1e-12);
            assert!((a.center.x - b.center.x).abs() < 1e-12);
            assert!((a.center.z - b.center.z).abs() < 1e-12);
            assert!((a.incidence_rad - b.incidence_rad).abs() < 1e-15);
        }
    }

    #[test]
    fn sweep_and_dihedral_move_the_quarter_chord() {
        let w = WingGeometry {
            sweep_rad: 5.0f64.to_radians(),
            dihedral_rad: 3.0f64.to_radians(),
            ..rect()
        };
        let els = discretize(&w, 0, 12, Discretization::default(), true);
        let tip = els
            .iter()
            .filter(|e| e.side > 0.0)
            .max_by(|a, b| a.span_pos_m.partial_cmp(&b.span_pos_m).unwrap())
            .unwrap();
        // tip swept aft: x > root x
        assert!(tip.quarter_chord_center.x > 0.5, "sweep not applied: {}", tip.quarter_chord_center.x);
        // dihedral with +Z down: tip raised = z negative
        assert!(tip.quarter_chord_center.z < -0.1, "dihedral not applied: {}", tip.quarter_chord_center.z);
        // aspect ratio consistency: sum of spanwise extents = full span
        let span_sum: f64 = els
            .iter()
            .filter(|e| e.side > 0.0)
            .map(|e| e.span_m)
            .sum();
        assert!((span_sum - w.span_m * 0.5).abs() < 1e-12);
    }

    #[test]
    fn washout_and_incidence_interpolate_linearly() {
        let w = WingGeometry {
            washout_rad: -3.0f64.to_radians(),
            ..rect()
        };
        let els = discretize(&w, 0, 10, Discretization::default(), true);
        for e in els.iter().filter(|e| e.side > 0.0) {
            let expect = w.incidence_rad + w.washout_rad * e.span_frac;
            assert!((e.incidence_rad - expect).abs() < 1e-15, "station {}", e.span_frac);
        }
        let root_i = els
            .iter()
            .filter(|e| e.side > 0.0)
            .min_by(|a, b| a.span_frac.partial_cmp(&b.span_frac).unwrap())
            .unwrap()
            .incidence_rad;
        let tip_i = els
            .iter()
            .filter(|e| e.side > 0.0)
            .max_by(|a, b| a.span_frac.partial_cmp(&b.span_frac).unwrap())
            .unwrap()
            .incidence_rad;
        assert!(tip_i < root_i, "washout must reduce tip incidence");
    }

    #[test]
    fn asymmetric_surface_produces_one_side() {
        let fin = WingGeometry {
            name: "vtail",
            span_m: 5.0, // fin height
            root_chord_m: 3.0,
            tip_chord_m: 1.4,
            sweep_rad: 20f64.to_radians(),
            dihedral_rad: 0.0,
            incidence_rad: 0.0,
            washout_rad: 0.0,
            area_m2: (3.0 + 1.4) * 0.5 * 5.0, // trapezoid-integrated area
            position_m: DVec3::new(-15.0, 0.0, -0.5),
        };
        let els = discretize(&fin, 3, 5, Discretization::default(), false);
        assert_eq!(els.len(), 5);
        assert!((total_area(&els) - fin.area_m2).abs() < 1e-9);
        assert!(els.iter().all(|e| e.side > 0.0));
        // fin elements stay on the symmetry plane (y = 0)
        assert!(els.iter().all(|e| e.center.y.abs() < 1e-12));
    }

    #[test]
    fn element_centers_lie_within_planform() {
        let w = WingGeometry {
            tip_chord_m: 1.0,
            sweep_rad: 10f64.to_radians(),
            ..rect()
        };
        let els = discretize(&w, 0, 7, Discretization { chordwise: 2 }, true);
        for e in &els {
            // chordwise span: LE at x_qc - 0.25c, TE at x_qc + 0.75c
            let x_qc = w.position_m.x + w.sweep_rad.tan() * e.span_pos_m;
            let le = x_qc - 0.25 * w.chord_at(e.span_frac);
            let te = x_qc + 0.75 * w.chord_at(e.span_frac);
            assert!(e.center.x >= le - 1e-9 && e.center.x <= te + 1e-9);
            assert!(e.span_pos_m <= w.span_m * 0.5 + 1e-9);
        }
    }

    #[test]
    fn centroid_of_symmetric_wing_is_on_centerline() {
        let w = rect();
        let els = discretize(&w, 0, 9, Discretization { chordwise: 2 }, true);
        let c = area_centroid(&els);
        assert!(c.y.abs() < 1e-9, "centroid off centerline: {c}");
    }
}
