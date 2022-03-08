import os
import glob
from pathlib import Path
import numpy as np
import random
import carb
from PIL import Image
from tensorflow import keras
from pxr import Usd, UsdGeom, Gf, UsdPhysics
import omni.kit
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.prims import create_prim, delete_prim
from omni.usd import get_context
from omni.kit.viewport import get_viewport_interface
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.materials import PreviewSurface
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.syntheticdata import sensors
import omni.syntheticdata._syntheticdata as sd


def setColliderSubtree(prim, approximationShape="none", execute_command_fn=None):
    pit = iter(Usd.PrimRange(prim))
    for p in pit:
        if p.GetMetadata("hide_in_stage_window"):
            pit.PruneChildren()
            continue
        if p.IsA(UsdGeom.Gprim) or p.IsInstanceable():
            if len(p.GetAttribute("faceVertexIndices").Get()) > 0:
                omni.physx.scripts.utils.setCollider(p, approximationShape, execute_command_fn)


def setRigidBody(prim, approximationShape, kinematic, custom_execute_fn=None):
    omni.physx.scripts.utils.setPhysics(prim, kinematic, custom_execute_fn)

    if prim.IsA(UsdGeom.Xformable):
        setColliderSubtree(prim, approximationShape, custom_execute_fn)
    else:
        omni.physx.scripts.utils.setCollider(prim, approximationShape, custom_execute_fn)


def create_light():
    create_prim(
        "/World/SphereLight",
        "SphereLight",
        position=np.array([0, 500, 500]),
        attributes={
            "radius": 150,
            "intensity": 5e4
        }
    )


def create_classification_camera():
    create_prim(
        "/World/ClassificationCamera",
        "Camera",
        orientation=np.array([0.33, 0.197, 0.464, 0.794]),
        position=np.array([151, 250, 135])
    )


def find_usd_assets(shapenet_dir, categories, max_asset_size=50):
    """Look for USD files under root/category for each category specified.
    For each category, generate a list of all USD files found and select
    assets up to split * len(num_assets) if `train=True`, otherwise select the
    remainder.
    """
    from omni.isaac.shapenet.utils import LABEL_TO_SYNSET

    references = {}
    for category in categories:
        category_id = LABEL_TO_SYNSET[category]
        all_assets = glob.glob(
            os.path.join(shapenet_dir, category_id, "*/*.usd"),
            recursive=True)
        if max_asset_size is None:
            assets_filtered = all_assets
        else:
            assets_filtered = []
            for a in all_assets:
                if os.stat(a).st_size > max_asset_size * 1e6:
                    carb.log_warn(
                        f"{a} skipped as it exceeded the max \
                        size {max_asset_size} MB.")
                else:
                    assets_filtered.append(a)
        num_assets = len(assets_filtered)
        if num_assets == 0:
            raise ValueError(
                f"No USDs found for category {category} \
                under max size {max_asset_size} MB.")

        references[category] = assets_filtered

    return references


def create_conveyor_anchor(plate_size):
    size = 5
    conveyor_anchor = create_prim(
        "/World/Conveyor/Anchor",
        "Cube",
        position=np.array([0.0, -plate_size/2 - size, 0.0]),
        scale=np.array([plate_size / 2, size, size]))
    conveyor_anchor.GetAttribute("visibility").Set("invisible")
    return conveyor_anchor


def create_conveyor_plate(stage, size, index):
    plate_path = f"/World/Conveyor/Plates/Plate{index + 1}"
    plate = DynamicCuboid(
        prim_path=plate_path,
        position=np.array([0, index * 100, 0.0]),
        size=np.array([size - 5, size - 5, 10.0]),
        color=np.array([0.28, 0.65, 1.0])
    )

    # prismatic joint
    joint_path = f"/World/Conveyor/Joints/PrismaticJoint{index + 1}"
    prismatic_joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    prismatic_joint.CreateAxisAttr("Y")
    prismatic_joint.CreateBody0Rel().SetTargets(["/World/Conveyor/Anchor"])
    prismatic_joint.CreateBody1Rel().SetTargets([plate_path])
    prismatic_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 1.0, 0.0))
    prismatic_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, -0.5, 0.0))

    # add linear drive
    driver = UsdPhysics.DriveAPI.Apply(
        prismatic_joint.GetPrim(),
        "linear")
    driver.CreateTypeAttr("force")
    driver.CreateMaxForceAttr(1000)
    driver.CreateTargetVelocityAttr(200.0)
    driver.CreateDampingAttr(1e10)
    driver.CreateStiffnessAttr(0)
    return plate


def create_pusher(stage, plate_size, index):
    actuator_path = f"/World/Pushers/Actuators/Actuator{index + 1}"
    anchor_path = f"/World/Pushers/Anchors/Anchor{index + 1}"
    depth = 10
    
    anchor = create_prim(
        anchor_path,
        "Cube",
        position=np.array([
            -plate_size/2 - depth - 5,
            (index + 2) * plate_size * 2,
            20.0]),
        scale=np.array([5, 5, 5]))
    anchor.GetAttribute("visibility").Set("invisible")

    pusher = DynamicCuboid(
        prim_path=actuator_path,
        position=np.array([
            -plate_size/2 - 5,
            (index + 2) * plate_size * 2,
            20.0]),
        size=np.array([depth, plate_size * 2, 30]),
        color=np.array([0.1, 0.1, 0.5])
    )

    mass_api = UsdPhysics.MassAPI.Apply(pusher.prim)
    mass_api.CreateMassAttr(1)

    # Prismatic joint 
    joint_path = f"/World/Pushers/Joints/Joint{index + 1}"
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateAxisAttr("X")
    joint.CreateBody0Rel().SetTargets([anchor_path])
    joint.CreateBody1Rel().SetTargets([actuator_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(1.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5, 0.0, 0.0))

    # Linear drive. No position target is set, only activated when needed.  
    driver = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    driver.CreateTypeAttr("force")
    driver.CreateMaxForceAttr(1000)
    driver.CreateDampingAttr(2e4)
    driver.CreateStiffnessAttr(1e5)

    return driver


def create_bucket(stage, plate_size, index):
    bucket_path = f"/World/Buckets/Bucket{index + 1}"

    width = plate_size * 2
    depth = width
    height = 20
    a = create_prim(
        f"{bucket_path}/a",
        "Cube",
        position=np.array([
            plate_size/2 + depth/2 - 10,
            (index + 2) * 2 * plate_size - width / 2,
            -height - 5
        ]),
        scale=np.array([depth/2, 5, height]),
        attributes={
            "primvars:displayColor": [(1.0, 1.0, 1.0)]
        }
    )
    b = create_prim(
        f"{bucket_path}/b",
        "Cube",
        position=np.array([
            plate_size/2 + depth/2 - 10,
            (index + 2) * 2 * plate_size + width / 2,
            -height - 5
        ]),
        scale=np.array([depth/2, 5, height]),
        attributes={
            "primvars:displayColor": [(1.0, 1.0, 1.0)]
        }
    )
    c = create_prim(
        f"{bucket_path}/c",
        "Cube",
        position=np.array([
            plate_size/2 + 5 - 10,
            (index + 2) * 2 * plate_size,
            -height - 5
        ]),
        scale=np.array([5, width/2 - 5, height]),
        attributes={
            "primvars:displayColor": [(1.0, 1.0, 1.0)]
        }
    )
    d = create_prim(
        f"{bucket_path}/d",
        "Cube",
        position=np.array([
            plate_size/2 + depth - 5 - 10,
            (index + 2) * 2 * plate_size,
            -height - 5
        ]),
        scale=np.array([5, width/2 - 5, height]),
        attributes={
            "primvars:displayColor": [(1.0, 1.0, 1.0)]
        }
    )
    UsdPhysics.CollisionAPI.Apply(a)
    UsdPhysics.CollisionAPI.Apply(b)
    UsdPhysics.CollisionAPI.Apply(c)
    UsdPhysics.CollisionAPI.Apply(d)
    

class Conveyor2(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()

        self.model = keras.models.load_model("/home/bjnortier/isaac/sorting/save_at_30-augmented-3.h5")
        self.categories = [
            "bus", "car", "plane", "rocket", "watercraft"
        ]
        shapenet_dir = Path(os.environ["SHAPENET_LOCAL_DIR"])
        self.asset_references = find_usd_assets(
            f"{shapenet_dir}_nomat",
            self.categories)

        self.num_classes = len(self.categories) 
        self.num_plates = self.num_classes * 2 + 4
        
        plate_size = 100.0
        self.max_plate_position = plate_size * self.num_plates
        self.widget_index = 0
        self.plate_reset_count = 0

        stage = get_context().get_stage()
        world.scene.add_ground_plane(z_position=-45.0)
        create_light()
        create_classification_camera()
        create_conveyor_anchor(plate_size)
        
        self.plates = []
        for i in range(self.num_plates):
            self.plates.append(create_conveyor_plate(stage, plate_size, i))

        self.pushers = []
        for i in range(self.num_classes):
            self.pushers.append(create_pusher(stage, plate_size, i))

        for i in range(self.num_classes):
            create_bucket(stage, plate_size, i)

        viewport_interface = get_viewport_interface()
        viewport_handle = viewport_interface.create_instance()
        vp = viewport_interface.get_viewport_window(viewport_handle)
        vp.set_active_camera("/World/ClassificationCamera")
        vp.set_texture_resolution(299, 299)
        self.classification_viewport = vp

        self.sd_interface = sd.acquire_syntheticdata_interface()
        self.is_sensor_initialized = False

        # # Create the first widget
        self.drop_widget(y_position=100.0)

        return

    def drop_widget(self, y_position=0.0):
        category = random.choice(self.categories)
        asset_reference = random.choice(self.asset_references[category])
        widget_path = f"/World/widget_{self.widget_index}"
        widget_prim = create_prim(
            widget_path,
            "Xform",
            scale=np.array([50.0, 50.0, 50.0]),
            orientation=euler_angles_to_quat(
                np.array([90.0, 0.0, 0.0]),
                degrees=True),
            position=np.array([0.0, y_position, 50.0]),
            usd_path=asset_reference,
            semantic_label=category)
        self.current_widget_category = category

        widget = XFormPrim(widget_path)
        material = PreviewSurface(
            prim_path="/World/Looks/ShapeMaterial",
            color=np.array([0.1, 0.6, 0.1]))
        widget.apply_visual_material(material)

        # Determine bounds and translate to sit on the Z=0 plane
        orientation_on_plane = euler_angles_to_quat(
            np.array([90.0, 0.0, 0.0]),
            degrees=True)
        widget.set_local_pose(
            np.array([0.0, 0.0, 0.0]),
            orientation_on_plane)
        bounds = UsdGeom.Mesh(widget_prim).ComputeWorldBound(0.0, "default")
        new_position = np.array([0.0, 0.0, -bounds.GetBox().GetMin()[2] + 5.0])
        widget.set_local_pose(new_position)

        mass_api = UsdPhysics.MassAPI.Apply(widget_prim)
        mass_api.CreateMassAttr(1)

        setRigidBody(widget_prim, "convexHull", False)        

        self.widget = widget
        self.widget_index += 1
        self.widget_class = None
        self.classification_requested = False
        self.classification_complete = False
        self.arm_activated = False
        for pusher in self.pushers:
            pusher.CreateTargetPositionAttr(0.0)

    async def setup_post_load(self):
        self._world = self.get_world()
        self._world.add_physics_callback("sim_step", callback_fn=self.sim_step_callback) 
        return

    def sim_step_callback(self, step_size):
        if not self.is_sensor_initialized:
            print("Waiting for sensor to initialize")
            sensor = sensors.create_or_retrieve_sensor(
                self.classification_viewport, sd.SensorType.Rgb)
            self.is_sensor_initialized = \
                self.sd_interface.is_sensor_initialized(sensor)
            if self.is_sensor_initialized:
                print("Sensor initialized!")

        for plate in self.plates:
            # When a plate reaches the end ov the conveyour belt,
            # reset it's position to the start. Drop a widget if it's
            # the first plate
            plate_position, _ = plate.get_world_pose()
            if plate_position[1] > self.max_plate_position:
                plate_position[1] -= self.max_plate_position
                plate.set_world_pose(plate_position)
                self.plate_reset_count += 1
                if self.plate_reset_count == self.num_plates:
                    self.plate_reset_count = 0
                    self.drop_widget()

        # Classify the widget when it passes under the camera
        if not self.classification_requested:
            widget_position, _ = self.widget.get_world_pose()
            if widget_position[1] > 100:
                self.capture_gt()
                self.classification_requested = True

        if self.classification_complete and not self.arm_activated:
            widget_position, _ = self.widget.get_world_pose()
            if widget_position[1] > (self.widget_class + 1) * 200 + 100:
                self.arm_activated = True                
                self.pushers[self.widget_class].CreateTargetPositionAttr(120.0)

    def capture_gt(self):
        rgb = sensors.get_rgb(self.classification_viewport)
        # Discard alpha channel
        rgb = rgb[:, :, :3]
        input = np.expand_dims(rgb, axis=0)
        prediction = self.model.predict(input)
        self.widget_class = np.argmax(prediction)
        
        print(f"actual:predicted {self.current_widget_category}:{self.categories[self.widget_class]}")
        image = Image.fromarray(rgb)
        image.save("/tmp/rgb.png")
        self.classification_complete = True
             
    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return

