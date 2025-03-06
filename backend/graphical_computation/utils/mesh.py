class Mesh:
    def __init__(self, context, geometry, texture):
        self.context = context
        self.geometry = geometry
        self.texture = texture

    def render(self, program, use_uv=True):
        self.texture.use()

        vao = self.geometry.vertex_array(program, use_uv=use_uv)
        vao.render()
