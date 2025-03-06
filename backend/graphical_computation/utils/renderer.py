import glm
import moderngl
import numpy as np

from backend.graphical_computation.utils.image_texture import ImageTexture
from backend.graphical_computation.utils.loaders import load_shader_source
from backend.graphical_computation.utils.mesh import Mesh
from backend.graphical_computation.utils.model_geometry import ModelGeometry
from backend.trajectories.trajectory import Trajectory


class Renderer:
    def __init__(
            self,
            height: int,
            width: int,
            obj_path: str,
            texture_path: str,
            trajectory: Trajectory,
            camera_fov=45.0
    ):
        self.debug = 0

        self.context = moderngl.create_standalone_context()

        self.height = height
        self.width = width

        rgb_vertex_shader_source = load_shader_source('shaders/rgb_vertex_shader.glsl')
        rgb_fragment_shader_source = load_shader_source('shaders/rgb_fragment_shader.glsl')

        flow_vertex_shader_source = load_shader_source('shaders/flow_vertex_shader.glsl')
        flow_fragment_shader_source = load_shader_source('shaders/flow_fragment_shader.glsl')

        self.rgb_program = self.context.program(
            vertex_shader=rgb_vertex_shader_source,
            fragment_shader=rgb_fragment_shader_source,
        )

        self.flow_program = self.context.program(
            vertex_shader=flow_vertex_shader_source,
            fragment_shader=flow_fragment_shader_source
        )

        self.texture = ImageTexture(self.context, texture_path)
        self.room_geometry = ModelGeometry(self.context, obj_path)
        self.room = Mesh(self.context, self.room_geometry, self.texture)

        # Create textures for color and depth
        self.color_texture = self.context.texture((width, height), 4, dtype='f1')
        self.depth_texture = self.context.depth_texture((width, height))

        self.flow_texture = self.context.texture((self.width, self.height), 4, dtype='f4')
        self.flow_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Framebuffer with both color and depth attachments
        self.fbo = self.context.framebuffer(
            color_attachments=[self.color_texture],
            depth_attachment=self.depth_texture
        )
        self.flow_fbo = self.context.framebuffer(
            color_attachments=[self.flow_texture]
        )

        self.color_data = np.zeros((height, width, 4), dtype=np.float32)
        self.depth_data = np.zeros((height, width), dtype=np.float32)
        self.flow_data = np.zeros((height, width, 4), dtype=np.float32)

        self.context.enable(self.context.DEPTH_TEST)

        self.near = 0.1
        self.far = 100.0

        self.proj = glm.perspective(
            glm.radians(camera_fov),
            self.width / self.height,
            0.1,
            100.0
        )

        self.trajectory = trajectory
        self.prev_camera_matrix = self.get_camera_matrix(
            camera_pos=self.trajectory.camera_pos,
            camera_front=self.trajectory.camera_front
        )

    def get_camera_matrix(self, camera_pos, camera_front):
        look = glm.lookAt(camera_pos, camera_pos + camera_front, glm.vec3(0.0, 1.0, 0.0))

        return self.proj * look

    def linearize_depth(self, depth):
        """
        Convert non-linear depth [0,1] to linear depth.

        Parameters:
        depth (numpy array or float): Non-linear depth values in range [0,1]
        near (float): Near clipping plane distance
        far (float): Far clipping plane distance

        Returns:
        numpy array or float: Linearized depth values
        """

        ndc_depth = depth * 2.0 - 1.0  # Convert to NDC [-1,1]
        lin_depth = (2.0 * self.near * self.far) / (self.far + self.near - ndc_depth * (self.far - self.near))

        return np.where(lin_depth > 0.95 * self.far, 0, lin_depth)

    def compute_optical_flow(self, current_camera_matrix):
        self.flow_program['prev_camera'].write(self.prev_camera_matrix)
        self.flow_program['cur_camera'].write(current_camera_matrix)

        self.flow_program['screen_size'].value = (self.width, self.height)

        self.flow_fbo.use()
        self.flow_fbo.clear(red=0.0, green=0.0, blue=0.0, alpha=0.0)
        self.room.render(self.flow_program, use_uv=False)

        self.flow_data[:] = np.flipud(
            np.frombuffer(self.flow_texture.read(), dtype=np.float32).reshape(self.height, self.width, 4)
        )

    def render(self, current_time, timestep):
        camera_pos, camera_front = self.trajectory.get_next_point(
            current_time=current_time,
            timestep=timestep
        )

        camera_matrix = self.get_camera_matrix(camera_pos=camera_pos, camera_front=camera_front)
        self.rgb_program['camera'].write(camera_matrix)

        self.fbo.use()
        self.fbo.clear(red=0.0, green=0.0, blue=0.0, alpha=0.0, depth=1.0)
        self.room.render(self.rgb_program)

        self.color_data[:] = np.flipud(
            np.frombuffer(self.color_texture.read(), dtype=np.uint8).reshape(self.height, self.width, 4)
        )
        self.depth_data[:] = np.flipud(
            np.frombuffer(self.depth_texture.read(), dtype=np.float32).reshape(self.height, self.width)
        )

        linearized_depth = self.linearize_depth(self.depth_data)

        self.compute_optical_flow(current_camera_matrix=camera_matrix)
        self.prev_camera_matrix = camera_matrix

        return self.color_data, linearized_depth, self.flow_data
