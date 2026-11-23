"""Editor-side builder for the fluid's assets: M_FluidSurface (the water's surface), M_FluidParticle (the
spray above it) and the linear texture their parameters default to.

Run once in the editor (demo.py does it when the asset is missing):

    unrealpy run --editor examples.fluid.assets:main

The repository ships no Unreal assets: the material is made from engine expressions with
MaterialEditingLibrary into Content/Fluid (Config/DefaultGame.ini cooks that folder).

Each particle is one instance of a quad (an InstancedStaticMeshComponent, the instance's custom data 0 is its
index). In the vertex shader the material reads the particle's texel from the "Positions" texture (x, y, z
in cm, speed in m/s; the game writes it every frame), moves the quad there and turns it to the camera. In the
pixel shader the quad becomes a little sphere: a disc (opacity mask) with a sphere's normal, deep blue water
that turns to white foam where the particle is fast.
"""

from __future__ import annotations

from typing import Any

import ue

FOLDER = "/Game/Fluid"
NAME = "M_FluidParticle"
PATH = f"{FOLDER}/{NAME}.{NAME}"
P = ue.EMaterialProperty

SAMPLE = """
float i = floor(Index + 0.5);
float2 uv = (float2(fmod(i, Dims.x), floor(i / Dims.x)) + 0.5) / Dims.xy;
return Texture2DSampleLevel(Tex, TexSampler, uv, 0);
"""
OFFSET = """
float2 c = Corner * 2.0 - 1.0;
float size = Size * lerp(1.0, 0.5, saturate((Texel.w - 2.5) / 3.0));  // spray: smaller drops
size = Texel.w < 0.0 ? 0.0 : size;  // under the surface: not drawn (a zero-size quad)
float3 target = Texel.xyz + (normalize(Right) * c.x - normalize(Up) * c.y) * (size * 0.5);
return target - WorldPos;
"""
SURFACE = """
float2 g = (WorldPos.xy - Grid.xy) / Grid.z - 0.5;
float2 f = frac(g);
float2 i0 = floor(g) + 0.5;
float2 inv = 1.0 / Dims.xy;
float4 a = Texture2DSampleLevel(Tex, TexSampler, i0 * inv, 0);
float4 b = Texture2DSampleLevel(Tex, TexSampler, (i0 + float2(1, 0)) * inv, 0);
float4 c = Texture2DSampleLevel(Tex, TexSampler, (i0 + float2(0, 1)) * inv, 0);
float4 d = Texture2DSampleLevel(Tex, TexSampler, (i0 + float2(1, 1)) * inv, 0);
return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
"""
LIFT = """
float h = Cell.x > 1.0 ? Cell.x : -6.0;  // dry: under the floor
return float3(0.0, 0.0, h - WorldPos.z);
"""
WATER_NORMAL = """
float2 n = Cell.zw;
return normalize(float3(n, sqrt(saturate(1.0 - dot(n, n)))));
"""
SPHERE = """
float2 c = Corner * 2.0 - 1.0;
float r2 = dot(c, c);
float3 front = normalize(ToCamera) * sqrt(saturate(1.0 - r2));
float3 n = normalize(normalize(Right) * c.x - normalize(Up) * c.y + front);
return float4(n, 1.0 - r2);
"""


class Graph:
    """Nodes and links over MaterialEditingLibrary."""

    def __init__(self, material: ue.Material) -> None:
        self.material = material
        self.edit = ue.MaterialEditingLibrary
        self.edit.delete_all_material_expressions(material)
        self.y = 0

    def node(self, cls: Any, **props: Any) -> Any:
        self.y += 140
        expression = self.edit.create_material_expression(self.material, cls, -700, self.y)
        for key, value in props.items():
            setattr(expression, key, value)
        return expression

    def link(self, source: Any, target: Any, pin: str = "", out: str = "") -> None:
        assert self.edit.connect_material_expressions(source, out, target, pin), (source, out, target, pin)

    def custom(self, code: str, output: Any, inputs: dict[str, Any], description: str) -> Any:
        node = self.node(
            ue.MaterialExpressionCustom,
            code=code.strip(),
            output_type=output,
            description=description,
            inputs=[ue.CustomInput(input_name=name) for name in inputs],
        )
        for name, source in inputs.items():
            self.link(source, node, name)
        return node

    def vector(self, r: float, g: float, b: float) -> Any:
        return self.node(ue.MaterialExpressionConstant3Vector, constant=ue.LinearColor(r, g, b, 1.0))

    def scalar(self, name: str, value: float) -> Any:
        return self.node(ue.MaterialExpressionScalarParameter, parameter_name=name, default_value=value)

    def mask(self, source: Any, channels: str) -> Any:
        node = self.node(
            ue.MaterialExpressionComponentMask,
            r="r" in channels,
            g="g" in channels,
            b="b" in channels,
            a="a" in channels,
        )
        self.link(source, node)
        return node

    def camera_axis(self, axis: tuple[float, float, float]) -> Any:
        node = self.node(
            ue.MaterialExpressionTransform,
            transform_source_type=ue.EMaterialVectorCoordTransformSource.TRANSFORMSOURCE_View,
            transform_type=ue.EMaterialVectorCoordTransform.TRANSFORM_World,
        )
        self.link(self.vector(*axis), node)
        return node

    def output(self, source: Any, prop: Any, out: str = "") -> None:
        assert self.edit.connect_material_property(source, out, prop), prop


def linear_texture() -> ue.Texture:
    """The Positions parameter's default: a copy of the engine's black texture, made linear (not sRGB), as
    the float texture the game puts there is."""
    path = f"{FOLDER}/T_FluidDefault"
    texture = ue.load_object(f"{path}.T_FluidDefault")
    if texture is None:
        texture = ue.EditorAssetLibrary.duplicate_asset("/Engine/EngineResources/Black", path)
    assert isinstance(texture, ue.Texture2D), path
    texture.srgb = False
    texture.compression_settings = ue.TextureCompressionSettings.TC_HDR
    texture.filter = ue.TextureFilter.TF_Nearest
    assert ue.EditorAssetLibrary.save_loaded_asset(texture, False)
    return texture


def build() -> ue.Material:
    material = ue.load_object(PATH)
    if material is None:
        tools = ue.AssetToolsHelpers.get_asset_tools()
        assert tools is not None
        material = tools.create_asset(NAME, FOLDER, ue.Material, ue.new_object(ue.MaterialFactoryNew))
    assert isinstance(material, ue.Material)
    material.blend_mode = ue.EBlendMode.BLEND_Masked
    material.shading_model = ue.EMaterialShadingModel.MSM_DefaultLit
    material.two_sided = True
    material.tangent_space_normal = False
    material.opacity_mask_clip_value = 0.02
    material.used_with_instanced_static_meshes = True
    g = Graph(material)
    texture = g.node(
        ue.MaterialExpressionTextureObjectParameter,
        parameter_name="Positions",
        texture=linear_texture(),
        sampler_type=ue.EMaterialSamplerType.SAMPLERTYPE_LinearColor,
    )
    index = g.node(ue.MaterialExpressionPerInstanceCustomData, data_index=0)
    dims = g.node(
        ue.MaterialExpressionVectorParameter,
        parameter_name="TextureSize",
        default_value=ue.LinearColor(512.0, 512.0, 0.0, 0.0),
    )
    corner = g.node(ue.MaterialExpressionTextureCoordinate)
    right, up = g.camera_axis((1.0, 0.0, 0.0)), g.camera_axis((0.0, 1.0, 0.0))
    texel = g.custom(
        SAMPLE,
        ue.ECustomMaterialOutputType.CMOT_Float4,
        {"Tex": texture, "Index": index, "Dims": g.mask(dims, "rg")},
        "Particle texel",
    )
    world = g.node(
        ue.MaterialExpressionWorldPosition,
        world_position_shader_offset=ue.EWorldPositionIncludedOffsets.WPT_ExcludeAllShaderOffsets,
    )
    offset = g.custom(
        OFFSET,
        ue.ECustomMaterialOutputType.CMOT_Float3,
        {
            "Texel": texel,
            "Corner": corner,
            "Right": right,
            "Up": up,
            "WorldPos": world,
            "Size": g.scalar("Size", 11.0),
        },
        "Billboard at the particle",
    )
    g.output(offset, P.MP_WorldPositionOffset)
    height_speed = g.node(ue.MaterialExpressionVertexInterpolator)  # to the pixel shader: z (cm) and speed
    g.link(g.mask(texel, "ba"), height_speed)
    speed = g.mask(height_speed, "g")
    height = g.mask(height_speed, "r")
    to_camera = g.node(ue.MaterialExpressionCameraVectorWS)
    sphere = g.custom(
        SPHERE,
        ue.ECustomMaterialOutputType.CMOT_Float4,
        {"Corner": corner, "Right": right, "Up": up, "ToCamera": to_camera},
        "Sphere impostor",
    )
    g.output(g.mask(sphere, "rgb"), P.MP_Normal)
    g.output(g.mask(sphere, "a"), P.MP_OpacityMask)
    foam_amount = g.node(ue.MaterialExpressionSmoothStep)
    g.link(g.scalar("FoamFrom", 3.0), foam_amount, "Min")
    g.link(g.scalar("FoamTo", 6.0), foam_amount, "Max")
    g.link(speed, foam_amount, "Value")
    depth = g.node(ue.MaterialExpressionLinearInterpolate)  # deep blue low down, brighter on the crests
    g.link(
        g.node(
            ue.MaterialExpressionVectorParameter,
            parameter_name="Water",
            default_value=ue.LinearColor(0.002, 0.03, 0.12, 1.0),
        ),
        depth,
        "A",
    )
    g.link(
        g.node(
            ue.MaterialExpressionVectorParameter,
            parameter_name="Crest",
            default_value=ue.LinearColor(0.02, 0.22, 0.42, 1.0),
        ),
        depth,
        "B",
    )
    crest = g.node(ue.MaterialExpressionSmoothStep)
    g.link(g.scalar("CrestFrom", 15.0), crest, "Min")
    g.link(g.scalar("CrestTo", 90.0), crest, "Max")
    g.link(height, crest, "Value")
    g.link(crest, depth, "Alpha")
    colour = g.node(ue.MaterialExpressionLinearInterpolate)
    g.link(depth, colour, "A")
    g.link(
        g.node(
            ue.MaterialExpressionVectorParameter,
            parameter_name="Foam",
            default_value=ue.LinearColor(0.7, 0.82, 0.92, 1.0),
        ),
        colour,
        "B",
    )
    g.link(foam_amount, colour, "Alpha")
    g.output(colour, P.MP_BaseColor)
    g.output(g.scalar("Roughness", 0.06), P.MP_Roughness)
    g.output(g.scalar("Specular", 0.7), P.MP_Specular)
    ue.MaterialEditingLibrary.recompile_material(material)
    assert ue.EditorAssetLibrary.save_loaded_asset(material, False)
    return material


def surface() -> ue.Material:
    """M_FluidSurface: each corner of a cell's quad rises to the height field (bilinear, in the vertex
    shader); the pixel shader takes the normal from it, colours the water by its depth (clear and light where
    it is shallow, deep blue where it is deep) and turns it white where it is fast or steep."""
    name = "M_FluidSurface"
    material = ue.load_object(f"{FOLDER}/{name}.{name}")
    if material is None:
        tools = ue.AssetToolsHelpers.get_asset_tools()
        assert tools is not None
        material = tools.create_asset(name, FOLDER, ue.Material, ue.new_object(ue.MaterialFactoryNew))
    assert isinstance(material, ue.Material)
    material.blend_mode = ue.EBlendMode.BLEND_Opaque
    material.shading_model = ue.EMaterialShadingModel.MSM_DefaultLit
    material.two_sided = False
    material.tangent_space_normal = False
    material.used_with_instanced_static_meshes = True
    g = Graph(material)
    texture = g.node(
        ue.MaterialExpressionTextureObjectParameter,
        parameter_name="Surface",
        texture=linear_texture(),
        sampler_type=ue.EMaterialSamplerType.SAMPLERTYPE_LinearColor,
    )
    grid = g.node(
        ue.MaterialExpressionVectorParameter,
        parameter_name="Grid",
        default_value=ue.LinearColor(-600.0, -400.0, 5.0, 0.0),
    )
    dims = g.node(
        ue.MaterialExpressionVectorParameter,
        parameter_name="GridSize",
        default_value=ue.LinearColor(240.0, 160.0, 0.0, 0.0),
    )
    vertex_world = g.node(
        ue.MaterialExpressionWorldPosition,
        world_position_shader_offset=ue.EWorldPositionIncludedOffsets.WPT_ExcludeAllShaderOffsets,
    )
    inputs = {"Tex": texture, "Grid": g.mask(grid, "rgb"), "Dims": g.mask(dims, "rg")}
    corner = g.custom(
        SURFACE,
        ue.ECustomMaterialOutputType.CMOT_Float4,
        {**inputs, "WorldPos": vertex_world},
        "Height field at the vertex",
    )
    lift = g.custom(
        LIFT,
        ue.ECustomMaterialOutputType.CMOT_Float3,
        {"Cell": corner, "WorldPos": vertex_world},
        "Lift to the surface",
    )
    g.output(lift, P.MP_WorldPositionOffset)
    pixel_world = g.node(ue.MaterialExpressionWorldPosition)
    here = g.custom(
        SURFACE,
        ue.ECustomMaterialOutputType.CMOT_Float4,
        {**inputs, "WorldPos": pixel_world},
        "Height field at the pixel",
    )
    normal = g.custom(WATER_NORMAL, ue.ECustomMaterialOutputType.CMOT_Float3, {"Cell": here}, "Normal")
    g.output(normal, P.MP_Normal)
    depth = g.node(ue.MaterialExpressionSmoothStep)
    g.link(g.scalar("ShallowAt", 4.0), depth, "Min")
    g.link(g.scalar("DeepAt", 70.0), depth, "Max")
    g.link(g.mask(here, "r"), depth, "Value")
    water = g.node(ue.MaterialExpressionLinearInterpolate)
    g.link(
        g.node(
            ue.MaterialExpressionVectorParameter,
            parameter_name="Shallow",
            default_value=ue.LinearColor(0.03, 0.22, 0.3, 1.0),
        ),
        water,
        "A",
    )
    g.link(
        g.node(
            ue.MaterialExpressionVectorParameter,
            parameter_name="Deep",
            default_value=ue.LinearColor(0.004, 0.035, 0.11, 1.0),
        ),
        water,
        "B",
    )
    g.link(depth, water, "Alpha")
    fast = g.node(ue.MaterialExpressionSmoothStep)
    g.link(g.scalar("FoamFrom", 3.2), fast, "Min")
    g.link(g.scalar("FoamTo", 6.0), fast, "Max")
    g.link(g.mask(here, "g"), fast, "Value")
    steep = g.node(ue.MaterialExpressionSmoothStep)
    g.link(g.scalar("SteepFrom", 0.55), steep, "Min")
    g.link(g.scalar("SteepTo", 0.92), steep, "Max")
    g.link(
        g.custom("return length(N.xy);", ue.ECustomMaterialOutputType.CMOT_Float1, {"N": normal}, "Slope"),
        steep,
        "Value",
    )
    foam = g.node(ue.MaterialExpressionMax)
    g.link(fast, foam, "A")
    g.link(steep, foam, "B")
    colour = g.node(ue.MaterialExpressionLinearInterpolate)
    g.link(water, colour, "A")
    g.link(
        g.node(
            ue.MaterialExpressionVectorParameter,
            parameter_name="Foam",
            default_value=ue.LinearColor(0.75, 0.85, 0.92, 1.0),
        ),
        colour,
        "B",
    )
    g.link(foam, colour, "Alpha")
    g.output(colour, P.MP_BaseColor)
    roughness = g.node(ue.MaterialExpressionLinearInterpolate)
    g.link(g.scalar("Roughness", 0.03), roughness, "A")
    g.link(g.scalar("FoamRoughness", 0.6), roughness, "B")
    g.link(foam, roughness, "Alpha")
    g.output(roughness, P.MP_Roughness)
    g.output(g.scalar("Specular", 0.9), P.MP_Specular)
    ue.MaterialEditingLibrary.recompile_material(material)
    assert ue.EditorAssetLibrary.save_loaded_asset(material, False)
    return material


def main(_args: list[str]) -> dict[str, Any]:
    return {"materials": [build().get_path_name(), surface().get_path_name()]}
