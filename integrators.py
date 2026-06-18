from __future__ import annotations

import math
import random
from pathlib import Path

import glm
from PIL import Image, ImageFilter

from camera import Camera
from materials import Material
from scene import AreaLight, HitRecord, Ray, Scene, build_cornell_box
from utils import EPSILON, balance_heuristic, clamp, luminance, ray_color_to_rgb, reflect, vec3


def _visible(scene: Scene, origin: glm.vec3, target: glm.vec3) -> bool:
    direction = target - origin
    distance = glm.length(direction)
    if distance <= 0.0:
        return False
    direction = direction / distance
    ray = Ray(origin + direction * EPSILON, direction)
    return scene.hit(ray, EPSILON, distance - EPSILON) is None


def _sample_light_contribution(
    scene: Scene,
    light: AreaLight,
    hit: HitRecord,
    incoming: glm.vec3,
    rng: random.Random,
) -> glm.vec3:
    target, light_normal, pdf_area, emission = light.sample(rng)
    direction = target - hit.p
    dist2 = glm.dot(direction, direction)
    if dist2 <= 0.0:
        return vec3(0.0, 0.0, 0.0)
    distance = math.sqrt(dist2)
    direction = direction / distance
    cos_surface = max(0.0, glm.dot(hit.normal, direction))
    cos_light = max(0.0, glm.dot(-direction, light_normal))
    if cos_surface <= 0.0 or cos_light <= 0.0:
        return vec3(0.0, 0.0, 0.0)
    if not _visible(scene, hit.p, target):
        return vec3(0.0, 0.0, 0.0)

    pdf_light = light.pdf_from(hit.p, target)
    pdf_bsdf = hit.material.pdf(-incoming, direction, hit.normal)
    weight = balance_heuristic(pdf_light, pdf_bsdf)
    if pdf_light <= 0.0:
        return vec3(0.0, 0.0, 0.0)
    brdf = hit.material.eval(-incoming, direction, hit.normal)
    return emission * brdf * cos_surface * weight / pdf_light


def _forced_light_closure(
    scene: Scene,
    light: AreaLight,
    hit: HitRecord,
    incoming: glm.vec3,
    rng: random.Random,
) -> glm.vec3:
    target, light_normal, pdf_area, emission = light.sample(rng)
    direction = target - hit.p
    dist2 = glm.dot(direction, direction)
    if dist2 <= 0.0:
        return vec3(0.0, 0.0, 0.0)
    distance = math.sqrt(dist2)
    direction = direction / distance
    cos_surface = max(0.0, glm.dot(hit.normal, direction))
    cos_light = max(0.0, glm.dot(-direction, light_normal))
    if cos_surface <= 0.0 or cos_light <= 0.0:
        return vec3(0.0, 0.0, 0.0)
    if not _visible(scene, hit.p, target):
        return vec3(0.0, 0.0, 0.0)
    pdf_light = light.pdf_from(hit.p, target)
    if pdf_light <= 0.0:
        return vec3(0.0, 0.0, 0.0)
    brdf = hit.material.eval(-incoming, direction, hit.normal)
    return emission * brdf * cos_surface / pdf_light


def _trace_path(
    scene: Scene,
    light: AreaLight,
    ray: Ray,
    rng: random.Random,
    d_max: int,
    step: int,
) -> glm.vec3:
    radiance = vec3(0.0, 0.0, 0.0)
    throughput = vec3(1.0, 1.0, 1.0)
    current_ray = ray
    last_vertex: HitRecord | None = None
    last_pdf = 0.0
    last_specular = False

    for bounce in range(max(1, d_max + 3)):
        hit = scene.hit(current_ray, EPSILON, float("inf"))
        if hit is None:
            radiance += throughput * scene.ambient
            break

        emitted = hit.material.emitted(hit, -current_ray.direction)
        if glm.length(emitted) > 0.0:
            if step >= 3 and last_vertex is not None and not last_specular:
                to_light = glm.normalize(hit.p - last_vertex.p)
                light_pdf = light.pdf_from(last_vertex.p, hit.p)
                weight = balance_heuristic(last_pdf, light_pdf)
                radiance += throughput * emitted * weight
            else:
                radiance += throughput * emitted
            break

        if step >= 3:
            radiance += throughput * _sample_light_contribution(scene, light, hit, current_ray.direction, rng)
        if step >= 5:
            radiance += throughput * _bidirectional_probe(scene, light, hit, current_ray.direction, rng)

        if bounce >= d_max:
            radiance += throughput * _forced_light_closure(scene, light, hit, current_ray.direction, rng)
            break

        if step >= 2 and bounce >= 2:
            q = max(0.05, min(0.8, 1.0 - luminance(throughput)))
            if rng.random() < q:
                break
            throughput /= 1.0 - q

        sample = hit.material.sample(-current_ray.direction, hit.normal, rng)
        if sample.pdf <= 0.0:
            break

        if sample.specular:
            throughput *= sample.attenuation
            current_ray = Ray(hit.p + hit.normal * EPSILON, glm.normalize(sample.direction))
            last_vertex = hit
            last_pdf = 1.0
            last_specular = True
            continue

        outgoing = glm.normalize(sample.direction)
        cos_term = max(0.0, glm.dot(hit.normal, outgoing))
        brdf = hit.material.eval(-current_ray.direction, outgoing, hit.normal)
        throughput *= brdf * cos_term / sample.pdf
        current_ray = Ray(hit.p + hit.normal * EPSILON, outgoing)
        last_vertex = hit
        last_pdf = sample.pdf
        last_specular = False

    return radiance


def _bidirectional_probe(
    scene: Scene,
    light: AreaLight,
    hit: HitRecord,
    incoming: glm.vec3,
    rng: random.Random,
) -> glm.vec3:
    target, light_normal, pdf_area, emission = light.sample(rng)
    if not _visible(scene, hit.p, target):
        return vec3(0.0, 0.0, 0.0)
    direction = glm.normalize(target - hit.p)
    cos_surface = max(0.0, glm.dot(hit.normal, direction))
    cos_light = max(0.0, glm.dot(-direction, light_normal))
    if cos_surface <= 0.0 or cos_light <= 0.0:
        return vec3(0.0, 0.0, 0.0)
    pdf_light = light.pdf_from(hit.p, target)
    pdf_bsdf = hit.material.pdf(-incoming, direction, hit.normal)
    weight = balance_heuristic(pdf_light, pdf_bsdf)
    brdf = hit.material.eval(-incoming, direction, hit.normal)
    return emission * brdf * cos_surface * weight / max(pdf_light, 1e-8)


def _render(
    scene: Scene,
    light: AreaLight,
    width: int,
    height: int,
    spp: int,
    d_max: int,
    step: int,
    use_filter: bool,
) -> Image.Image:
    camera = Camera.look_at(
        lookfrom=vec3(0.5, 0.5, -2.2),
        lookat=vec3(0.5, 0.5, 0.5),
        vup=vec3(0.0, 1.0, 0.0),
        vfov=40.0,
        aspect_ratio=width / height,
    )

    pixels: list[tuple[int, int, int]] = []
    for j in range(height - 1, -1, -1):
        for i in range(width):
            rng = random.Random((step * 1000003) + j * 1009 + i)
            color = vec3(0.0, 0.0, 0.0)
            for _ in range(spp):
                u = (i + rng.random()) / max(1, width - 1)
                v = (j + rng.random()) / max(1, height - 1)
                ray = camera.get_ray(u, v)
                color += _trace_path(scene, light, ray, rng, d_max=d_max, step=step)
            pixels.append(ray_color_to_rgb(color, spp))

    image = Image.new("RGB", (width, height))
    image.putdata(pixels)
    if use_filter:
        image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
    return image


def render_step(step: int, width: int, height: int, spp: int, d_max: int, use_filter: bool) -> Image.Image:
    scene, light, _ = build_cornell_box()
    return _render(scene, light, width, height, spp, d_max, step, use_filter)


def render_all_steps(width: int, height: int, spp: int, d_max: int, use_filter: bool) -> dict[int, Image.Image]:
    scene, light, _ = build_cornell_box()
    images: dict[int, Image.Image] = {}
    for step in range(1, 6):
        images[step] = _render(scene, light, width, height, spp, d_max, step, use_filter)
    return images
