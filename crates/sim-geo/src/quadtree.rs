//! Six-face spherical quadtree (cube → sphere) for tile level-of-detail.
//!
//! The sphere is covered by the six faces of a cube; each face is
//! recursively split into four quadrants, giving a `2^level × 2^level`
//! grid of tiles per face (`TileId = face + level + (x, y)`). Direction
//! vectors are plain 64-bit unit [`glam::DVec3`]s — this crate never
//! leaves `f64`.
//!
//! The basis vectors of each face are chosen so that neighbouring faces
//! share *identical* boundary point sets (see the `face_edges_are_shared`
//! test), which makes the quadtree watertight: a tile's edge always meets
//! the neighbouring face's edge exactly, with no T-junctions.

use crate::geodesic::wrap_pi;
use crate::types::LatLonAlt;
use glam::DVec3;

/// Number of cube faces.
pub const FACES: u8 = 6;

/// Face center axes: +X, −X, +Y, −Y, +Z, −Z.
pub(crate) const FACE_C: [DVec3; FACES as usize] = [
    DVec3::new(1.0, 0.0, 0.0),
    DVec3::new(-1.0, 0.0, 0.0),
    DVec3::new(0.0, 1.0, 0.0),
    DVec3::new(0.0, -1.0, 0.0),
    DVec3::new(0.0, 0.0, 1.0),
    DVec3::new(0.0, 0.0, -1.0),
];

/// Face +u axes (right-handed with C and V: C × U = V).
pub(crate) const FACE_U: [DVec3; FACES as usize] = [
    DVec3::new(0.0, 1.0, 0.0),
    DVec3::new(0.0, -1.0, 0.0),
    DVec3::new(-1.0, 0.0, 0.0),
    DVec3::new(1.0, 0.0, 0.0),
    DVec3::new(1.0, 0.0, 0.0),
    DVec3::new(1.0, 0.0, 0.0),
];

/// Face +v axes.
pub(crate) const FACE_V: [DVec3; FACES as usize] = [
    DVec3::new(0.0, 0.0, 1.0),
    DVec3::new(0.0, 0.0, 1.0),
    DVec3::new(0.0, 0.0, 1.0),
    DVec3::new(0.0, 0.0, 1.0),
    DVec3::new(0.0, 1.0, 0.0),
    DVec3::new(0.0, -1.0, 0.0),
];

/// Angular radius (radians) of a root tile: half the diagonal of a cube
/// face seen from the sphere centre, `acos(1/sqrt(3)) ≈ 0.9553 rad`.
pub const ROOT_ANGULAR_RADIUS: f64 = 0.9553206878578596;

/// Identity of one tile of the quadtree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId {
    /// Cube face 0..=5 (+X, −X, +Y, −Y, +Z, −Z).
    pub face: u8,
    /// Level of detail (0 = whole face).
    pub level: u8,
    /// Column within the face, `0 <= x < 2^level`.
    pub x: u32,
    /// Row within the face, `0 <= y < 2^level`.
    pub y: u32,
}

impl TileId {
    /// Root tile of a face.
    pub fn root(face: u8) -> Self {
        assert!(face < FACES, "face must be 0..6");
        Self { face, level: 0, x: 0, y: 0 }
    }

    /// All six roots.
    pub fn roots() -> [TileId; FACES as usize] {
        let mut a = [TileId::root(0); FACES as usize];
        for (i, t) in a.iter_mut().enumerate() {
            *t = TileId::root(i as u8);
        }
        a
    }

    /// Number of tiles per edge at this level.
    pub fn grid(&self) -> u32 {
        1u32 << self.level
    }

    /// Parent tile (`None` at level 0).
    pub fn parent(&self) -> Option<TileId> {
        if self.level == 0 {
            None
        } else {
            Some(TileId {
                face: self.face,
                level: self.level - 1,
                x: self.x / 2,
                y: self.y / 2,
            })
        }
    }

    /// The four children in Z-order (SW, SE, NW, NE), or `None` if the
    /// subdivision would exceed the maximum level.
    pub fn children(&self) -> Option<[TileId; 4]> {
        if self.level >= 30 {
            return None;
        }
        let l = self.level + 1;
        let (x, y) = (self.x * 2, self.y * 2);
        let mk = |dx: u32, dy: u32| TileId { face: self.face, level: l, x: x + dx, y: y + dy };
        Some([mk(0, 0), mk(1, 0), mk(0, 1), mk(1, 1)])
    }

    /// True if `other` is this tile or one of its (transitive) ancestors.
    pub fn is_ancestor_of(&self, other: &TileId) -> bool {
        if self.face != other.face || other.level < self.level {
            return false;
        }
        let shift = other.level - self.level;
        other.x >> shift == self.x && other.y >> shift == self.y
    }

    /// Parametric (u, v) ∈ [-1, 1]² range of this tile on its face.
    pub fn uv_range(&self) -> (f64, f64, f64, f64) {
        let g = self.grid() as f64;
        let u0 = -1.0 + 2.0 * self.x as f64 / g;
        let u1 = -1.0 + 2.0 * (self.x + 1) as f64 / g;
        let v0 = -1.0 + 2.0 * self.y as f64 / g;
        let v1 = -1.0 + 2.0 * (self.y + 1) as f64 / g;
        (u0, u1, v0, v1)
    }

    /// Unit direction of parametric position (u, v) on this tile's face.
    pub fn direction_at(&self, u: f64, v: f64) -> DVec3 {
        let c = FACE_C[self.face as usize];
        let a = FACE_U[self.face as usize];
        let b = FACE_V[self.face as usize];
        (c + a * u + b * v).normalize()
    }

    /// Unit directions of the four corners, counter-clockwise seen from
    /// outside the sphere: SW, SE, NE, NW.
    pub fn corners(&self) -> [DVec3; 4] {
        let (u0, u1, v0, v1) = self.uv_range();
        [
            self.direction_at(u0, v0),
            self.direction_at(u1, v0),
            self.direction_at(u1, v1),
            self.direction_at(u0, v1),
        ]
    }

    /// Unit direction of the tile centre (projection of the cube-face
    /// centre onto the sphere).
    pub fn center(&self) -> DVec3 {
        let (u0, u1, v0, v1) = self.uv_range();
        self.direction_at((u0 + u1) * 0.5, (v0 + v1) * 0.5)
    }

    /// Geodetic position of the tile centre on the unit sphere
    /// (altitude 0; scale by the ellipsoid as needed).
    pub fn center_lat_lon(&self) -> LatLonAlt {
        let d = self.center();
        let lat = d.z.asin();
        let lon = wrap_pi(d.y.atan2(d.x));
        LatLonAlt::new(lat, lon, 0.0)
    }

    /// Angular radius (radians) from the tile centre to its farthest
    /// corner — a conservative bound for culling.
    pub fn angular_radius(&self) -> f64 {
        let c = self.center();
        self.corners()
            .iter()
            .map(|k| c.dot(*k).clamp(-1.0, 1.0).acos())
            .fold(0.0f64, f64::max)
    }

    /// Angular geometric error (radians) of this level: the root radius
    /// divided by `2^level`. Halves with every subdivision.
    pub fn geometric_error_rad(&self) -> f64 {
        ROOT_ANGULAR_RADIUS / (1u64 << self.level) as f64
    }

    /// World-space geometric error in metres on a sphere of `radius_m`.
    pub fn geometric_error_m(&self, radius_m: f64) -> f64 {
        self.geometric_error_rad() * radius_m
    }

    /// Whether this tile's centre covers the given unit direction
    /// (dominant-axis face assignment keeps this O(1)).
    pub fn contains(&self, dir: DVec3) -> bool {
        let (face, u, v) = face_and_uv(dir);
        if face != self.face {
            return false;
        }
        let g = self.grid() as f64;
        let fx = ((u + 1.0) * 0.5 * g).floor();
        let fy = ((v + 1.0) * 0.5 * g).floor();
        let fx = fx.clamp(0.0, g - 1.0) as u32;
        let fy = fy.clamp(0.0, g - 1.0) as u32;
        fx == self.x && fy == self.y
    }
}

/// Deterministically map a unit direction to (face, u, v) using the
/// dominant-axis rule.
pub fn face_and_uv(dir: DVec3) -> (u8, f64, f64) {
    let ax = dir.x.abs();
    let ay = dir.y.abs();
    let az = dir.z.abs();
    let face = if ax >= ay && ax >= az {
        if dir.x >= 0.0 { 0 } else { 1 }
    } else if ay >= az {
        if dir.y >= 0.0 { 2 } else { 3 }
    } else if dir.z >= 0.0 {
        4
    } else {
        5
    };
    let c = FACE_C[face as usize];
    let a = FACE_U[face as usize];
    let b = FACE_V[face as usize];
    let s = dir.dot(c);
    // s cannot be ~0: c was chosen as the dominant axis.
    (face, dir.dot(a) / s, dir.dot(b) / s)
}

/// The tile containing `dir` at `level`.
pub fn tile_for_direction(dir: DVec3, level: u8) -> TileId {
    let (face, u, v) = face_and_uv(dir.normalize());
    let g = 1u64 << level;
    let fx = (((u + 1.0) * 0.5 * g as f64).floor() as i64).clamp(0, g as i64 - 1) as u32;
    let fy = (((v + 1.0) * 0.5 * g as f64).floor() as i64).clamp(0, g as i64 - 1) as u32;
    TileId { face, level, x: fx, y: fy }
}

/// Parameters for [`select_tiles`].
#[derive(Debug, Clone, Copy)]
pub struct SelectionParams {
    /// Apparent geometric-error threshold (radians): a tile refines while
    /// `world_error / distance` exceeds this. Smaller = finer globe.
    pub tolerance_rad: f64,
    /// Hard cap on subdivision depth.
    pub max_level: u8,
    /// Refine only tiles whose *nearest* point is within this many metres
    /// of the camera. This bounds the total tile count (a cone of detail
    /// around the camera instead of uniform globe refinement). Pass
    /// `camera_altitude * 2 + 30_000` for typical flight regimes.
    pub refine_radius_m: f64,
}

impl Default for SelectionParams {
    fn default() -> Self {
        Self {
            // ~0.5° apparent error
            tolerance_rad: 0.01,
            max_level: 18,
            refine_radius_m: 80_000.0,
        }
    }
}

/// Camera-driven tile selection (the algorithm the sim-world streamer
/// runs every frame).
///
/// Starting from the six roots, a tile subdivides into four children when
/// all of the following hold:
///
/// 1. its *apparent* geometric error (world error ÷ distance to the
///    tile's nearest point) exceeds [`SelectionParams::tolerance_rad`],
/// 2. its nearest point is within [`SelectionParams::refine_radius_m`] of
///    the camera (local refinement, bounded tile counts), and
/// 3. the depth cap has not been reached.
///
/// The result is a watertight partition: no selected tile is an ancestor
/// of another, and all selected tiles together cover the sphere exactly.
pub fn select_tiles(
    camera_ecef: DVec3,
    sphere_radius_m: f64,
    params: SelectionParams,
) -> Vec<TileId> {
    let p = params;
    let mut out: Vec<TileId> = Vec::with_capacity(256);
    let mut stack: Vec<TileId> = TileId::roots().to_vec();
    while let Some(t) = stack.pop() {
        let center = t.center() * sphere_radius_m;
        let distance_center = (center - camera_ecef).length();
        let nearest = (distance_center - t.angular_radius() * sphere_radius_m).max(1.0);
        let apparent_err = t.geometric_error_m(sphere_radius_m) / distance_center.max(1.0);
        if t.level < p.max_level
            && nearest <= p.refine_radius_m
            && apparent_err > p.tolerance_rad
        {
            if let Some(kids) = t.children() {
                stack.extend_from_slice(&kids);
                continue;
            }
        }
        out.push(t);
    }
    out
}

/// Verify that `tiles` form a watertight partition of the sphere: no
/// duplicates and no tile that is an ancestor of another. Runs in
/// `O(n · depth)` using a hash set of the ancestor chain.
pub fn is_watertight_partition(tiles: &[TileId]) -> bool {
    let set: std::collections::HashSet<TileId> = tiles.iter().copied().collect();
    if set.len() != tiles.len() {
        return false;
    }
    for t in tiles {
        let mut p = t.parent();
        while let Some(a) = p {
            if set.contains(&a) {
                return false;
            }
            p = a.parent();
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WGS84;

    fn grid_points_on_edge(tile: &TileId, edge: u8, level: u8) -> Vec<DVec3> {
        // edge: 0 = u0 (west), 1 = u1 (east), 2 = v0 (south), 3 = v1 (north)
        let g = 1i64 << level;
        let mut pts = Vec::new();
        for i in 0..=g {
            let t = i as f64 / g as f64 * 2.0 - 1.0;
            let (u, v) = match edge {
                0 => (-1.0, t),
                1 => (1.0, t),
                2 => (t, -1.0),
                _ => (t, 1.0),
            };
            pts.push(tile.direction_at(u, v));
        }
        pts
    }

    #[test]
    fn roots_cover_six_faces() {
        let roots = TileId::roots();
        assert_eq!(roots.len(), 6);
        for (i, r) in roots.iter().enumerate() {
            assert_eq!(r.face, i as u8);
            assert_eq!(r.level, 0);
            let c = r.center();
            assert!((c.length() - 1.0).abs() < 1e-12);
        }
        // root centers point along the six axis directions
        assert!(roots[0].center().x > 0.999);
        assert!(roots[1].center().x < -0.999);
        assert!(roots[5].center().z < -0.999);
    }

    #[test]
    fn corners_are_unit_vectors() {
        for t in [TileId::root(0), TileId::root(3), TileId { face: 4, level: 7, x: 33, y: 100 }] {
            for c in t.corners() {
                assert!((c.length() - 1.0).abs() < 1e-12, "corner not unit: {c}");
            }
            assert!((t.center().length() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn children_tile_the_parent_exactly() {
        let parent = TileId { face: 0, level: 4, x: 5, y: 9 };
        let kids = parent.children().expect("level 4 can subdivide");
        let (pu0, pu1, pv0, pv1) = parent.uv_range();
        let mid_u = (pu0 + pu1) / 2.0;
        let mid_v = (pv0 + pv1) / 2.0;
        // each child covers one quadrant of the parent uv box
        let boxes = kids
            .iter()
            .map(|k| k.uv_range())
            .collect::<Vec<_>>();
        assert_eq!(boxes[0], (pu0, mid_u, pv0, mid_v)); // SW
        assert_eq!(boxes[1], (mid_u, pu1, pv0, mid_v)); // SE
        assert_eq!(boxes[2], (pu0, mid_u, mid_v, pv1)); // NW
        assert_eq!(boxes[3], (mid_u, pu1, mid_v, pv1)); // NE
        // shared corners of children = parent corners + midpoints
        for k in &kids {
            assert!(parent.is_ancestor_of(k));
            assert_eq!(k.parent(), Some(parent));
            for c in k.corners() {
                let ang = c.dot(parent.center()).acos();
                assert!(ang <= parent.angular_radius() + 1e-9);
            }
        }
    }

    #[test]
    fn face_edges_are_shared_watertight() {
        // For every pair of adjacent faces the grid points along the
        // common cube edge must coincide (as sets) at every level.
        // Adjacency derived from the cube geometry itself:
        let adjacent: [(u8, u8); 12] = [
            (0, 2), (0, 3), (0, 4), (0, 5), // +X touches +Y,-Y,+Z,-Z
            (1, 2), (1, 3), (1, 4), (1, 5), // -X touches +Y,-Y,+Z,-Z
            (2, 4), (2, 5),                 // +Y touches +Z,-Z
            (3, 4), (3, 5),                 // -Y touches +Z,-Z
        ];
        for level in 0..4u8 {
            let g = 1i64 << level;
            for (fa, fb) in adjacent {
                // collect all grid points on each face boundary
                let ta = TileId::root(fa);
                let tb = TileId::root(fb);
                let mut pa = Vec::new();
                for edge in 0..4 {
                    pa.extend(grid_points_on_edge(&ta, edge, level));
                }
                let mut pb = Vec::new();
                for edge in 0..4 {
                    pb.extend(grid_points_on_edge(&tb, edge, level));
                }
                // every boundary point of B must lie on SOME edge of A
                // within (grid resolution): the two faces share exactly
                // one edge — the cube edge whose points satisfy both.
                // Test the actual shared-edge correspondence directly:
                let shared = shared_edge_points(&ta, &tb, level);
                for p in &shared {
                    let on_a = pa.iter().any(|q| (q - *p).length() < 1e-9);
                    let on_b = pb.iter().any(|q| (q - *p).length() < 1e-9);
                    assert!(on_a && on_b, "level {level} faces {fa},{fb}: point {p} missing");
                }
                assert_eq!(shared.len(), g as usize + 1, "level {level} faces {fa},{fb}");
            }
        }
    }

    fn shared_edge_points(a: &TileId, b: &TileId, level: u8) -> Vec<DVec3> {
        // The shared edge of two cube faces: directions where both
        // dominant axes have equal |component| and the third is zero.
        // Easier: intersect boundary point sets geometrically.
        let g = 1i64 << level;
        let mut pts = Vec::new();
        let mut all_a = Vec::new();
        for edge in 0..4 {
            all_a.extend(grid_points_on_edge(a, edge, level));
        }
        for edge in 0..4 {
            for p in grid_points_on_edge(b, edge, level) {
                if all_a.iter().any(|q| (q - p).length() < 1e-9) {
                    pts.push(p);
                }
            }
        }
        pts.sort_by(|a, b| {
            a.x.partial_cmp(&b.x)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a.y.partial_cmp(&b.y)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.z.partial_cmp(&b.z).unwrap_or(std::cmp::Ordering::Equal))
        });
        pts.dedup_by(|x, y| (*x - *y).length() < 1e-9);
        // expect the full edge column count
        assert!(pts.len() as i64 > g || pts.is_empty());
        pts
    }

    #[test]
    fn children_share_exact_edges_with_siblings() {
        let p = TileId { face: 2, level: 6, x: 17, y: 42 };
        let k = p.children().unwrap();
        let ends_match = |a0: DVec3, a1: DVec3, b0: DVec3, b1: DVec3| {
            ((a0 - b0).length() < 1e-12 && (a1 - b1).length() < 1e-12)
                || ((a0 - b1).length() < 1e-12 && (a1 - b0).length() < 1e-12)
        };
        // horizontal neighbours: left child's east edge == right child's west edge
        // corners are (SW, SE, NE, NW)
        for (l, r) in [(0usize, 1usize), (2, 3)] {
            let cl = k[l].corners();
            let cr = k[r].corners();
            assert!(
                ends_match(cl[1], cl[2], cr[0], cr[3]),
                "east/west edge mismatch between children {l} and {r}"
            );
        }
        // vertical neighbours: lower child's north edge == upper child's south edge
        for (lo, up) in [(0usize, 2usize), (1, 3)] {
            let cl = k[lo].corners();
            let cu = k[up].corners();
            assert!(
                ends_match(cl[3], cl[2], cu[0], cu[1]),
                "north/south edge mismatch between children {lo} and {up}"
            );
        }
    }

    #[test]
    fn containment_and_lookup_agree() {
        let samples = [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.2, -0.5, 0.8).normalize(),
            DVec3::new(-0.7, 0.1, -0.3).normalize(),
            DVec3::new(0.01, -0.99, 0.02).normalize(),
            DVec3::new(0.6, 0.6, 0.6).normalize(),
        ];
        for dir in samples {
            for level in [0u8, 3, 8] {
                let t = tile_for_direction(dir, level);
                assert!(t.contains(dir), "tile {t:?} must contain {dir}");
                // children of other faces must not contain it
                for f in 0..6u8 {
                    if f != t.face {
                        assert!(!TileId { face: f, level, x: t.x, y: t.y }.contains(dir));
                    }
                }
            }
        }
    }

    #[test]
    fn parent_child_ancestor_logic() {
        let root = TileId::root(4);
        let deep = TileId { face: 4, level: 5, x: 17, y: 3 };
        assert!(root.is_ancestor_of(&deep));
        assert!(!deep.is_ancestor_of(&root));
        assert!(!TileId::root(5).is_ancestor_of(&deep));
        // walk up from deep to root
        let mut t = deep;
        let mut hops = 0;
        while let Some(p) = t.parent() {
            t = p;
            hops += 1;
        }
        assert_eq!(hops, 5);
        assert_eq!(t, root);
    }

    #[test]
    fn geometric_error_halves_per_level() {
        let a = TileId { face: 0, level: 3, x: 0, y: 0 };
        let b = TileId { face: 0, level: 4, x: 0, y: 0 };
        assert!((a.geometric_error_rad() / 2.0 - b.geometric_error_rad()).abs() < 1e-15);
        let err_m = a.geometric_error_m(WGS84.authalic_radius());
        // level 3 root error ≈ 0.9553/8 rad × R ≈ 761 km on Earth
        assert!((err_m - 760_794.0).abs() < 1_000.0, "err={err_m}");
    }

    #[test]
    fn selection_is_watertight_partition_and_refines_with_distance() {
        let r = WGS84.authalic_radius();
        let params = SelectionParams {
            tolerance_rad: 0.002,
            max_level: 12,
            refine_radius_m: 600_000.0,
        };
        // camera 400 km above (0°N, 0°E)
        let cam_dir = DVec3::new(1.0, 0.0, 0.0);
        let cam = cam_dir * (r + 400_000.0);
        let tiles = select_tiles(cam, r, params);
        assert!(tiles.len() >= 6, "must at least cover the 6 roots");
        assert!(
            is_watertight_partition(&tiles),
            "not a watertight partition ({} tiles)",
            tiles.len()
        );
        // nadir region refines deeply, antipodal region stays shallow
        let max_near = tiles
            .iter()
            .filter(|t| t.center().dot(cam_dir) > 0.99)
            .map(|t| t.level)
            .max()
            .unwrap_or(0);
        let max_far = tiles
            .iter()
            .filter(|t| t.center().dot(cam_dir) < -0.99)
            .map(|t| t.level)
            .max()
            .unwrap_or(0);
        assert!(max_near >= 10, "near region too coarse: L{max_near}");
        assert!(max_far <= 4, "far region over-refined: L{max_far}");

        // a much closer camera with a deeper cap refines further
        let cam_near = cam_dir * (r + 5_000.0);
        let p2 = SelectionParams { max_level: 16, refine_radius_m: 40_000.0, ..params };
        let tiles_near = select_tiles(cam_near, r, p2);
        assert!(is_watertight_partition(&tiles_near));
        let max_depth = |v: &[TileId]| v.iter().map(|t| t.level).max().unwrap_or(0);
        assert!(max_depth(&tiles_near) > max_depth(&tiles));

        // underground camera still yields a valid partition (roots only
        // when nothing is within the refine radius)
        let tiles_under = select_tiles(cam_dir * (r * 0.5), r, params);
        assert!(tiles_under.len() >= 6);
        assert!(is_watertight_partition(&tiles_under));
    }

    #[test]
    fn selection_respects_max_level_exactly() {
        let r = WGS84.authalic_radius();
        // unbounded refine radius + tiny tolerance: every leaf sits at the
        // depth cap, giving exactly 6·4^level tiles.
        let params = SelectionParams {
            tolerance_rad: 1e-9,
            max_level: 7,
            refine_radius_m: f64::INFINITY,
        };
        let cam = DVec3::new(0.0, 0.0, r + 1.0);
        let tiles = select_tiles(cam, r, params);
        assert_eq!(tiles.len(), 6 * 4usize.pow(7));
        assert!(tiles.iter().all(|t| t.level == 7));
        assert!(is_watertight_partition(&tiles));
    }

    #[test]
    fn selection_default_params_are_sane() {
        let p = SelectionParams::default();
        assert!(p.tolerance_rad > 0.0 && p.tolerance_rad < 0.1);
        assert!(p.max_level >= 10 && p.max_level <= 24);
        assert!(p.refine_radius_m > 1_000.0);
    }

    #[test]
    fn center_lat_lon_matches_direction() {
        let t = TileId { face: 0, level: 2, x: 1, y: 2 };
        let ll = t.center_lat_lon();
        let d = t.center();
        let back = DVec3::new(
            ll.latitude_rad().cos() * ll.longitude_rad().cos(),
            ll.latitude_rad().cos() * ll.longitude_rad().sin(),
            ll.latitude_rad().sin(),
        );
        assert!((back - d).length() < 1e-12);
    }
}
