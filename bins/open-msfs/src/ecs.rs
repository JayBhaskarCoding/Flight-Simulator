//! A small, real entity-component system: entities are dense `u32`
//! ids, components live in typed sparse maps, and systems are plain
//! functions that iterate entity ids and pull the components they
//! need. No ECS dependency is pulled into the workspace — the engine
//! only needs "entities + typed components + system functions", which
//! this provides completely (no placeholder storage, no dead API).

use glam::{DQuat, DVec3};
use std::collections::HashMap;

/// Entity identifier (dense, monotonically increasing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Entity(pub u32);

/// Typed sparse component store.
#[derive(Debug, Clone)]
pub struct Store<T> {
    map: HashMap<Entity, T>,
}

impl<T> Default for Store<T> {
    fn default() -> Self {
        Self { map: HashMap::new() }
    }
}

impl<T> Store<T> {
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    pub fn insert(&mut self, e: Entity, value: T) -> Option<T> {
        self.map.insert(e, value)
    }

    pub fn remove(&mut self, e: Entity) -> Option<T> {
        self.map.remove(&e)
    }

    pub fn get(&self, e: Entity) -> Option<&T> {
        self.map.get(&e)
    }

    pub fn get_mut(&mut self, e: Entity) -> Option<&mut T> {
        self.map.get_mut(&e)
    }

    pub fn contains(&self, e: Entity) -> bool {
        self.map.contains_key(&e)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Entity, &T)> {
        self.map.iter().map(|(k, v)| (*k, v))
    }
}

/// World-space (physics ENU frame at first; re-based to ECEF through
/// the frame bridge before rendering) pose.
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: DVec3,
    pub orientation: DQuat,
}

/// Linear/angular velocity of a dynamic body.
#[derive(Debug, Clone, Copy)]
pub struct Velocity {
    pub linear: DVec3,
    pub angular_body: DVec3,
}

/// Marks the simulated aircraft.
#[derive(Debug, Clone, Copy, Default)]
pub struct Aircraft;

/// Marks the render/debug camera.
#[derive(Debug, Clone, Copy, Default)]
pub struct Camera {
    /// Offset from the aircraft origin in *aircraft body* axes
    /// (+X forward, +Y right, +Z down).
    pub body_offset: DVec3,
}

/// Marks an entity whose transform drives GPU draw submission (the
/// render system re-bases the pose into camera-relative `f32` every
/// frame — [`sim_render::Renderer::draws_from_gltf`] does the f64 →
/// f32 arithmetic, so no stale matrices are ever stored here).
#[derive(Debug, Clone, Copy, Default)]
pub struct Renderable;

/// The engine world: entity id allocation + component stores.
#[derive(Debug, Default)]
pub struct Ecs {
    next: u32,
    pub transform: Store<Transform>,
    pub velocity: Store<Velocity>,
    pub aircraft: Store<Aircraft>,
    pub camera: Store<Camera>,
    pub renderable: Store<Renderable>,
}

impl Ecs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Spawn an empty entity.
    pub fn spawn(&mut self) -> Entity {
        let e = Entity(self.next);
        self.next += 1;
        e
    }

    /// Despawn: drops every component of `e`.
    pub fn despawn(&mut self, e: Entity) {
        self.transform.remove(e);
        self.velocity.remove(e);
        self.aircraft.remove(e);
        self.camera.remove(e);
        self.renderable.remove(e);
    }

    /// All entities carrying both a transform and the aircraft marker.
    pub fn aircraft_entities(&self) -> Vec<Entity> {
        self.aircraft
            .iter()
            .filter(|(e, _)| self.transform.contains(*e))
            .map(|(e, _)| e)
            .collect()
    }

    /// All entities carrying both a transform and a camera marker.
    pub fn camera_entities(&self) -> Vec<Entity> {
        self.camera
            .iter()
            .filter(|(e, _)| self.transform.contains(*e))
            .map(|(e, _)| e)
            .collect()
    }

    /// Every entity that can produce a draw this frame.
    pub fn renderable_entities(&self) -> Vec<Entity> {
        self.renderable
            .iter()
            .filter(|(e, _)| self.transform.contains(*e))
            .map(|(e, _)| e)
            .collect()
    }

    /// Live entity count (all stores touched at least once).
    pub fn entity_count(&self) -> u32 {
        self.next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_despawn_and_component_scopes() {
        let mut ecs = Ecs::new();
        let a = ecs.spawn();
        let b = ecs.spawn();
        assert_ne!(a, b);

        ecs.transform.insert(a, Transform { position: DVec3::ZERO, orientation: DQuat::IDENTITY });
        ecs.aircraft.insert(a, Aircraft);
        ecs.transform.insert(b, Transform { position: DVec3::X, orientation: DQuat::IDENTITY });
        ecs.camera.insert(b, Camera { body_offset: DVec3::Y });

        assert_eq!(ecs.aircraft_entities(), vec![a]);
        assert_eq!(ecs.camera_entities(), vec![b]);
        assert_eq!(ecs.renderable_entities().len(), 0);

        ecs.despawn(a);
        assert!(ecs.aircraft_entities().is_empty());
        assert!(ecs.transform.get(a).is_none());
        assert_eq!(ecs.entity_count(), 2, "ids are never recycled");
    }

    #[test]
    fn store_overwrite_returns_previous() {
        let mut store: Store<u32> = Store::new();
        let e = Entity(0);
        assert!(store.insert(e, 1).is_none());
        assert_eq!(store.insert(e, 2), Some(1));
        assert_eq!(store.get(e), Some(&2));
        assert_eq!(store.remove(e), Some(2));
        assert!(store.is_empty());
    }
}
