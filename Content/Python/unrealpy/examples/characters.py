"""The runtime examples' people: Epic's mannequins (see mannequins.py for where the assets come from) as
Python characters, animated by the template's own anim blueprint.

`DemoMannequin` is an ACharacter: the capsule and CharacterMovement move it, the template's ABP_Unarmed
reads the movement and blends idle, walk and jog (BS_Idle_Walk_Run). The blend space's samples are the
animations' own speeds, measured from the feet: walk 300 cm/s, jog 600 cm/s; a character moving at a
speed between them blends both, so the feet stay planted at any speed up to 600. It turns towards where it
is going at a limited rate (orient rotation to movement), never snapping.

    actor = ue.spawn_actor(DemoMannequin, ue.Vector(0, 0, 100))
    actor.dress(quinn=False, tint=(0.8, 0.82, 0.86))   # silver
    actor.steer_to((500.0, 0.0), speed=300.0)           # every tick: walks there, then stops
"""

from __future__ import annotations

import math
from typing import cast

import ue
from examples import mannequins

MOVABLE = ue.EComponentMobility.Movable
WALK_SPEED, JOG_SPEED = 300.0, 600.0  # the blend space's samples: the animations' own ground speeds
CAPSULE_HALF_HEIGHT = 96.0


@ue.uclass
class DemoMannequin(ue.Character):
    """A mannequin moved by CharacterMovement (an AI controller possesses it, so movement input works)."""

    use_controller_rotation_yaw = False
    auto_possess_ai = ue.EAutoPossessAI.PlacedInWorldOrSpawned

    def begin_play(self) -> None:
        if self.get_controller() is None:  # spawned while the game mode starts: no AI controller yet
            self.spawn_default_controller()
        movement = self.character_movement
        movement.run_physics_with_no_controller = True
        movement.set_movement_mode(ue.EMovementMode.MOVE_Walking, 0)
        movement.orient_rotation_to_movement = True
        movement.rotation_rate = ue.Rotator(0.0, 300.0, 0.0)  # degrees per second: smooth turns
        movement.max_walk_speed = WALK_SPEED
        movement.max_acceleration = 900.0
        movement.braking_deceleration_walking = 900.0
        movement.use_separate_braking_friction = True
        movement.braking_friction = 2.0
        mesh = self.mesh
        mesh.set_skeletal_mesh_asset(cast(ue.SkeletalMesh, ue.load_object(mannequins.MESH_MANNY)))
        mesh.set_relative_location_and_rotation(
            ue.Vector(0.0, 0.0, -CAPSULE_HALF_HEIGHT), ue.Rotator(0.0, -90.0, 0.0), False, True
        )
        anim = ue.load_object(mannequins.ANIM_CLASS)
        if isinstance(anim, type):
            mesh.set_anim_instance_class(cast(type[ue.Object], anim))
        self.dressed = False

    def dress(
        self,
        *,
        quinn: bool = False,
        tint: tuple[float, float, float] = (0.78, 0.8, 0.84),
        metallic: float = 1.0,
        roughness: float = 0.28,
    ) -> None:
        """Manny or Quinn, in a metal paint of the given tint (the mannequin material's own parameters,
        so the logos, panel lines and normal detail stay)."""
        mesh = self.mesh
        if quinn:
            mesh.set_skeletal_mesh_asset(cast(ue.SkeletalMesh, ue.load_object(mannequins.MESH_QUINN)))
        for index in range(mesh.get_num_materials()):
            base = mesh.get_material(index)
            if base is None:
                continue
            material = mesh.create_dynamic_material_instance(index, base)
            if material is None:
                continue
            material.set_vector_parameter_value("Paint Tint", ue.LinearColor(*tint, 1.0))
            material.set_scalar_parameter_value("MetalPaintMetallic", metallic)
            material.set_scalar_parameter_value("MetalPaintRoughness", roughness)
        self.dressed = True

    def steer_to(self, point: tuple[float, float], speed: float = WALK_SPEED, arrive: float = 40.0) -> bool:
        """Walk (or jog: up to 600 cm/s) towards a floor point; True once within `arrive` (then it stops
        and the anim blueprint settles into idle)."""
        here = self.get_actor_location()
        dx, dy = point[0] - here[0], point[1] - here[1]
        distance = math.hypot(dx, dy)
        if distance <= arrive:
            return True
        self.character_movement.max_walk_speed = speed
        self.add_movement_input(ue.Vector(dx / distance, dy / distance, 0.0), 1.0, False)
        return False

    def speed(self) -> float:
        velocity = self.get_velocity()
        return math.hypot(velocity[0], velocity[1])
