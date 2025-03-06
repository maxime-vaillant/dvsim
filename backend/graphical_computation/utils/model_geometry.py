from objloader import Obj


class ModelGeometry:
    def __init__(self, context, path):
        self.context = context
        try:
            obj = Obj.open(path)
            # Prepare VBOs for different configurations
            self.vbo_with_uv = self.context.buffer(obj.pack('vx vy vz tx ty'))
            self.vbo_without_uv = self.context.buffer(obj.pack('vx vy vz'))
        except FileNotFoundError:
            raise Exception(f"Model file not found: {path}")

    def vertex_array(self, program, use_uv=True):
        if use_uv:
            # Use the buffer with UV mapping
            return self.context.vertex_array(program, [(self.vbo_with_uv, '3f 2f', 'in_vertex', 'in_uv')])
        else:
            # Use the buffer without UV mapping
            return self.context.vertex_array(program, [(self.vbo_without_uv, '3f', 'in_vertex')])

