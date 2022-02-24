import numpy as np
from PIL import Image
import os
import glob
import carb
from pathlib import Path
import random
from omni.isaac.kit import SimulationApp
from omni.isaac.shapenet.utils import LABEL_TO_SYNSET

random.seed(335899)


def dr_seed():
    return random.randint(0, 2 ** 16 - 1)    


def find_usd_assets(shapenet_dir, categories, max_asset_size=50):
    """Look for USD files under root/category for each category specified.
    For each category, generate a list of all USD files found and select
    assets up to split * len(num_assets) if `train=True`, otherwise select the
    remainder.
    """

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


class TrainingDataGenerator(object):
    def __init__(self, categories):
        self.app = SimulationApp({"headless": True})

        from omni.isaac.core import World
        import omni.isaac.dr as dr
        
        self.world = World(stage_units_in_meters=0.01)
        self.create_environment()

        dr.commands.ToggleManualModeCommand().do()

        shapenet_dir = Path(os.environ["SHAPENET_LOCAL_DIR"])
        self.asset_references = find_usd_assets(
            f"{shapenet_dir}_nomat",
            categories)

    def create_environment(self):
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        import omni.isaac.dr as dr

        # Camera and rig
        create_prim(
            "/World/CameraRig",
            "Xform"
        )
        create_prim(
            "/World/CameraRig/Camera",
            "Camera",
            position=np.array([0.0, 0.0, 100.0])
        )

        # Light source
        create_prim("/World/LightRig", "Xform")
        create_prim(
            "/World/LightRig/Light",
            "SphereLight",
            position=np.array([100, 0, 200]),
            attributes={
                "radius": 50,
                "intensity": 5e4,
                "color": (1.0, 1.0, 1.0)
            }
        )

        # Conveyor "belt"
        create_prim(
            "/World/Belt",
            "Cube",
            position=np.array([0.0, 0.0, -1.0]),
            scale=np.array([100.0, 100.0, 1.0]),
            orientation=euler_angles_to_quat(
                np.array([0.0, 0.0, 0.0]),
                degrees=True
            ),
            attributes={"primvars:displayColor": [(0.28, 0.65, 1.0)]},
        )

        # DR commands for light and camera
        dr.commands.CreateLightComponentCommand(
            light_paths=["/World/LightRig/Light"],
            intensity_range=(5e3, 5e4),
            seed=dr_seed()
        ).do()
        dr.commands.CreateRotationComponentCommand(
            prim_paths=["/World/LightRig"],
            min_range=(0.0, 15.0, -180.0),
            max_range=(0.0, 60.0, 180.0),
            seed=dr_seed()
        ).do()
        dr.commands.CreateRotationComponentCommand(
            prim_paths=["/World/CameraRig"],
            min_range=(0.0, 0.0, -180.0),
            max_range=(0.0, 60.0, 180.0),
            seed=dr_seed()
        ).do()

    def load_random_shape(self, category):
        from pxr import UsdGeom
        import omni.isaac.dr as dr
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.core.prims.xform_prim import XFormPrim
        from omni.isaac.core.materials import PreviewSurface

        asset_reference = random.choice(self.asset_references[category])

        create_prim("/World/Shape", "Xform")
        create_prim("/World/Shape/rig", "Xform")
        prim = create_prim(
            "/World/Shape/rig/Cube",
            "Xform",
            scale=np.array([20.0, 20.0, 20.0]),
            orientation=euler_angles_to_quat(
                np.array([90.0, 0.0, 0.0]),
                degrees=True),
            usd_path=asset_reference,
            semantic_label=category)

        primX = XFormPrim("/World/Shape/rig/Cube")
        orientation_on_plane = euler_angles_to_quat(
            np.array([90.0, 0.0, 0.0]),
            degrees=True)
        primX.set_local_pose(
            np.array([0.0, 0.0, 0.0]),
            orientation_on_plane)
        # Determine bounds and translate to sit on the Z=0 plane
        bounds = UsdGeom.Mesh(prim).ComputeWorldBound(0.0, "default")
        new_position = np.array([0.0, 0.0, -bounds.GetBox().GetMin()[2]])
        primX.set_local_pose(new_position)

        material = PreviewSurface(
            prim_path="/World/Looks/shape_material",
            color=np.array([0.2, 0.7, 0.2]))
        primX.apply_visual_material(material)

        dr.commands.CreateMovementComponentCommand(
            prim_paths=["/World/Shape/rig"],
            min_range=(-5.0, -5.0, 0.0),
            max_range=(5.0, 5.0, 0.0),
            seed=dr_seed()
        ).do()
        dr.commands.CreateRotationComponentCommand(
            prim_paths=["/World/Shape/rig"],
            min_range=(0.0, 0.0, 0.0),
            max_range=(0.0, 0.0, 360.0),
            seed=dr_seed()
        ).do()

    def clear_shape(self):
        from omni.isaac.core.utils.prims import delete_prim
        delete_prim("/World/Shape")

    def capture_rgb_groundtruth(self, filename):
        import omni.isaac.dr as dr
        from omni.kit.viewport import get_default_viewport_window
        from omni.isaac.synthetic_utils import SyntheticDataHelper

        viewport = get_default_viewport_window()
        viewport.set_active_camera("/World/CameraRig/Camera")
        viewport.set_texture_resolution(1024, 1024)
        sd_helper = SyntheticDataHelper()

        sd_helper.initialize(sensor_names=["rgb"], viewport=viewport)

        dr.commands.RandomizeOnceCommand().do()
        generator.world.step()
        generator.world.step()
        generator.world.step()

        ground_truth = sd_helper.get_groundtruth(
            ["rgb"],
            viewport, verify_sensor_init=True
        )
        image = Image.fromarray(ground_truth["rgb"])
        image.save(filename)

    def close(self):
        self.app.close()


if __name__ == "__main__":
    categories = [
        "watercraft", "plane", "car", "bus", "rocket"
    ]
    generator = TrainingDataGenerator(categories)

    for category in [categories[2]]:
        generator.load_random_shape(category)
        for i in range(9):
            generator.capture_rgb_groundtruth(f"/home/bjnortier/isaac/sorting/training_data/rgb_{category}_{i}.png")
        generator.clear_shape()

    generator.close()
