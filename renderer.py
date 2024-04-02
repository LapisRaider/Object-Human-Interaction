import math
import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags
from Rendering.lib.models.smpl import get_smpl_faces

class Renderer:
    def __init__(self, resolution=(224,224), wireframe=False):
        self.resolution = resolution

        self.faces = get_smpl_faces()
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

        # [VIBE-Object Start]
        self.cam_node = None
        self.camera_pose = np.eye(4)
        self.fov = 0.0
        self.object_nodes = []
        self.human_nodes = []
        # [VIBE-Object End]

        self.whiteBackground = np.full((self.resolution[1], self.resolution[0], 3), 255, dtype=np.uint8)

    def push_persp_cam(self, yfov, cam_pose=np.eye(4)):
        self.fov = yfov
        camera = pyrender.PerspectiveCamera(yfov, 0.1, 1000.0)
        self.camera_pose = cam_pose
        self.cam_node = self.scene.add(camera, pose=self.camera_pose)
        
    def push_human(self, verts, color=[1.0, 1.0, 0.9], translation=[0.0, 0.0, 0.0]):
        # Build mesh from vertices.
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]) # Harcode another rotation because the human model loaded is using a different coordinate system.
        T = trimesh.transformations.translation_matrix(translation)

        # Apply transformations.
        mesh.apply_transform(Rx)
        mesh.apply_transform(T)

        # Material
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.5,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        self.human_nodes.append(self.scene.add(mesh))

    def push_obj(self,
                   mesh_file,
                   translation_offset = [0.0, 0.0, 0.0],
                   translation=[0.0, 0.0, 0.0],
                   angle=0.0, # Rotation Angle (Radians)
                   axis=[1.0, 0.0, 0.0], # Rotation Axis (Right-Hand System: X points right. Y points up. Z points out.)
                   scale=[1.0, 1.0, 1.0],
                   color=[0.3, 1.0, 0.3]):
        # Load mesh from file.
        mesh = trimesh.load(mesh_file)

        T_Offset = trimesh.transformations.translation_matrix(translation_offset)

        # Apply transformations.
        Sx = trimesh.transformations.scale_matrix(scale[0], origin=[0,0, 0.0, 0.0], direction=[1.0, 0.0, 0.0])
        Sy = trimesh.transformations.scale_matrix(scale[1], origin=[0,0, 0.0, 0.0], direction=[0.0, 1.0, 0.0])
        Sz = trimesh.transformations.scale_matrix(scale[2], origin=[0,0, 0.0, 0.0], direction=[0.0, 0.0, 1.0])
        
        # prevent divide by zero error when angle is 0
        if angle == 0:
            R = trimesh.transformations.identity_matrix()
        else:
            R = trimesh.transformations.rotation_matrix(angle, axis)

        T = trimesh.transformations.translation_matrix(translation)
        
        mesh.apply_transform(Sx)
        mesh.apply_transform(Sy)
        mesh.apply_transform(Sz)
        mesh.apply_transform(T_Offset)
        mesh.apply_transform(R)
        mesh.apply_transform(T)

        # Setup material.
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.5,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        # Attach material to mesh and add it to scene.
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        self.object_nodes.append(self.scene.add(mesh))

    def pop_and_render(self, img = None, _renderWhite = False):
        # Render triangles or wireframe.
        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        # background will just be white
        if _renderWhite:
            img = self.whiteBackground

        # Combine current rendered scene with input image.
        # Allows multiple objects to be rendered by combining their resultant output.
        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        # Remove nodes
        self.scene.remove_node(self.cam_node)
        for n in self.object_nodes:
            self.scene.remove_node(n)
        for n in self.human_nodes:
            self.scene.remove_node(n)
        
        self.cam_node = None
        self.object_nodes.clear()
        self.human_nodes.clear()

        return image