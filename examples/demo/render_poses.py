"""Render the README demo inputs: one rigged character, several actions, three angles.

Usage (Blender 4.2+ in background mode):

    blender -b --python examples/demo/render_poses.py -- animated-woman.glb ./renders

The demo used Quaternius' "Animated Woman" (CC0), https://poly.pizza/m/9kF7eTDbhO.
Five frames are sampled from each action (Death/Sitting/SitIdle/Idle skipped) and
rendered from -40, 0 and +40 degrees, giving 90 PNGs on a plain backdrop.
"""

import math
import os
import sys

import bpy
import mathutils

SKIP_ACTIONS = ("Death", "Sitting", "SitIdle", "Idle")
FRAMES_PER_ACTION = 5
CAMERA_ANGLES = (-40, 0, 40)
SIZE = (768, 1280)
BACKDROP = (0.93, 0.90, 0.85, 1.0)

glb_path, out_dir = sys.argv[sys.argv.index("--") + 1 :][:2]
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_path)
armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
for obj in bpy.data.objects:
    if obj.name.startswith("Icosphere"):  # helper mesh shipped with the model
        obj.hide_render = True
body = next(o for o in bpy.data.objects if o.type == "MESH" and not o.name.startswith("Icosphere"))

scene = bpy.context.scene
engines = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = SIZE

world = bpy.data.worlds.new("backdrop")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = BACKDROP
world.node_tree.nodes["Background"].inputs[1].default_value = 0.9

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
scene.collection.objects.link(sun)
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(50), 0, math.radians(30))

# Frame the character from its rest-pose height.
scene.frame_set(1)
evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
corners = [evaluated.matrix_world @ mathutils.Vector(c) for c in evaluated.bound_box]
bottom, top = min(p.z for p in corners), max(p.z for p in corners)
height = top - bottom
target = mathutils.Vector((0, 0, bottom + height * 0.5))
distance = height * 2.7

camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
scene.collection.objects.link(camera)
scene.camera = camera
camera.data.lens = 50

armature.animation_data_create()
count = 0
for action in bpy.data.actions:
    name = action.name.split("|")[-1]
    if name in SKIP_ACTIONS:
        continue
    armature.animation_data.action = action
    start, end = (int(f) for f in action.frame_range)
    for index in range(FRAMES_PER_ACTION):
        scene.frame_set(round(start + (end - start) * (index + 0.5) / FRAMES_PER_ACTION))
        for angle in CAMERA_ANGLES:
            yaw = math.radians(angle)
            camera.location = (
                target.x + distance * math.sin(yaw),
                target.y - distance * math.cos(yaw),
                target.z + height * 0.05,
            )
            camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = os.path.join(out_dir, f"{name}-{index}-{angle:+d}.png")
            bpy.ops.render.render(write_still=True)
            count += 1
print(f"rendered {count} images to {out_dir}")
