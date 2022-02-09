from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, Tf, UsdPhysics
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
from omni.usd import get_context
from omni.physx.scripts import utils
import numpy as np

def create_light(stage):
    sphereLight = UsdLux.SphereLight.Define(stage, "/World/sphereLight")
    sphereLight.CreateRadiusAttr(150)
    sphereLight.CreateIntensityAttr(5e4)
    sphereLight.AddTranslateOp().Set(Gf.Vec3f(0.0, 500.0, 600.0))


def create_wall(stage, size, gap):
    wallPath = "/World/wall"

    position = Gf.Vec3f(0.0, -size/2 - gap - size/10/2, 0.0)
    orientation = Gf.Quatf(1.0)
    color = Gf.Vec3f(165.0 / 255.0, 21.0 / 255.0, 21.0 / 255.0)
    size = 100.0
    scale = Gf.Vec3f(3.0, 0.1, 0.1)

    cubeGeom = UsdGeom.Cube.Define(stage, wallPath)
    cubePrim = stage.GetPrimAtPath(wallPath)
    cubeGeom.CreateSizeAttr(size)
    half_extent = size / 2
    cubeGeom.CreateExtentAttr([
        (-half_extent, -half_extent, -half_extent),
        (half_extent, half_extent, half_extent)
    ])
    cubeGeom.AddTranslateOp().Set(position)
    cubeGeom.AddOrientOp().Set(orientation)
    cubeGeom.AddScaleOp().Set(scale)
    cubeGeom.CreateDisplayColorAttr().Set([color])

    visAttr = cubePrim.GetAttribute("visibility")
    visAttr.Set("invisible")
    return cubePrim


def create_conveyor_plate(world, stage, size, gap, index):
    platePath = f"/World/plate_{index}"

    plate = DynamicCuboid(
        prim_path=platePath,
        name=f"plate_{index}",
        position=np.array([0, (index) * (size + gap), 0.0]),
        size=np.array([size, size, 10.0]),
        color=np.array([71.0 / 255.0, 165.0 / 255.0, 1.0])
    )
    world.scene.add(plate)

    # prismatic joint
    jointPath = f"/World/prismaticJoint_{index}"
    prismaticJoint = UsdPhysics.PrismaticJoint.Define(stage, jointPath)
    prismaticJoint.CreateAxisAttr("Y")
    prismaticJoint.CreateBody0Rel().SetTargets(["/World/wall"])
    prismaticJoint.CreateBody1Rel().SetTargets([platePath])
    prismaticJoint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, size/2 + gap, 0.0))
    prismaticJoint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    prismaticJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, -0.5, 0.0))
    prismaticJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    # add linear drive
    linearDriveAPI = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(jointPath), "linear")
    linearDriveAPI.CreateTypeAttr("force")
    linearDriveAPI.CreateMaxForceAttr(1000)
    linearDriveAPI.CreateTargetVelocityAttr(100.0)
    linearDriveAPI.CreateDampingAttr(1e10)
    linearDriveAPI.CreateStiffnessAttr(0)
    
    return plate


class Conveyor(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()

        self._plates = []
        self._num_plates = 10
        gap = 5.0
        size = 95.0
        self._max_plate_position = (size + gap) * self._num_plates
        self._widget_index = 0

        stage = get_context().get_stage()
        world.scene.add_ground_plane(z_position = -20)
        create_light(stage)
        create_wall(stage, size, gap)
        for i in range(self._num_plates):
            self._plates.append(create_conveyor_plate(self._world, stage, size, gap, i))

        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._world.add_physics_callback("sim_step", callback_fn=self.sim_step_callback) 
        return

    def sim_step_callback(self, step_size):
        for plate in self._plates:
            position, _ = plate.get_world_pose()
            if position[1] > self._max_plate_position:
                position[1] -= self._max_plate_position
                plate.set_world_pose(position)

                cube = DynamicCuboid(
                    prim_path=f"/World/widget_{self._widget_index}",
                    name=f"widget_{self._widget_index}",
                    position=np.array([0, 50, 100]),
                    size=np.array([10, 10, 10]),
                    color=np.array([1.0, 0.0, 0.0])
                )
                self.get_world().scene.add(cube)
                self._widget_index += 1

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
