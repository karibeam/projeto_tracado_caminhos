from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import glm
from PIL import Image, ImageFilter

from camera import Camera
from materials import Material
from scene import AreaLight, HitRecord, Ray, Scene, build_cornell_box
from utils import (
    EPSILON,
    PI,
    balance_heuristic,
    clamp,
    luminance,
    power_heuristic,
    random_cosine_direction,
    ray_color_to_rgb,
    reflect,
    to_world,
    vec3,
)


def _mis_weight(pdf_a: float, pdf_b: float, step: int) -> float:
    if step >= 6:
        return power_heuristic(pdf_a, pdf_b)
    return balance_heuristic(pdf_a, pdf_b)


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
    step: int,
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
    weight = _mis_weight(pdf_light, pdf_bsdf, step)
    if pdf_light <= 0.0:
        return vec3(0.0, 0.0, 0.0)
    brdf = hit.material.eval(-incoming, direction, hit.normal)
    return emission * brdf * cos_surface * weight / pdf_light


def _sample_direct_lighting(
    scene: Scene,
    light: AreaLight,
    hit: HitRecord,
    incoming: glm.vec3,
    rng: random.Random,
    step: int,
) -> glm.vec3:
    samples = 4 if step >= 6 else 1
    acc = vec3(0.0, 0.0, 0.0)
    for _ in range(samples):
        acc += _sample_light_contribution(scene, light, hit, incoming, rng, step)
    return acc / float(samples)


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


class LightVertex:
    __slots__ = ["p", "normal", "material", "incoming", "throughput"]
    def __init__(self, p: glm.vec3, normal: glm.vec3, material: Any, incoming: glm.vec3, throughput: glm.vec3):
        self.p = p
        self.normal = normal
        self.material = material
        self.incoming = incoming
        self.throughput = throughput


def _sample_light_path(scene: Scene, light: AreaLight, rng: random.Random) -> LightVertex | None:
    target, light_normal, pdf_area, emission = light.sample(rng)
    if pdf_area <= 0.0:
        return None
    # A radiância inicial leva em conta a área da luz e compensa a amostragem de cosseno da direção
    throughput = (emission * PI) / pdf_area

    # Amostra uma direção com base em cosseno em relação à normal negativa da luz (apontando para baixo)
    local_dir = random_cosine_direction(rng)
    direction = to_world(local_dir, -light_normal)

    ray = Ray(target + -light_normal * EPSILON, direction)
    hit = scene.hit(ray, EPSILON, float("inf"))
    if hit is None:
        return None

    return LightVertex(
        p=hit.p,
        normal=hit.normal,
        material=hit.material,
        incoming=-direction,
        throughput=throughput
    )


def _trace_path(
    scene: Scene,
    light: AreaLight,
    ray: Ray,
    rng: random.Random,
    d_max: int,
    step: int,
    light_vertex: LightVertex | None = None,
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
            radiance += throughput * (scene.ambient + scene.infinite_light)
            break

        emitted = hit.material.emitted(hit, -current_ray.direction)
        if glm.length(emitted) > 0.0:
            if step >= 3 and last_vertex is not None and not last_specular:
                to_light = glm.normalize(hit.p - last_vertex.p)
                light_pdf = light.pdf_from(last_vertex.p, hit.p)
                weight = _mis_weight(last_pdf, light_pdf, step)
                radiance += throughput * emitted * weight
            else:
                radiance += throughput * emitted
            break

        if step >= 3:
            radiance += throughput * _sample_direct_lighting(scene, light, hit, current_ray.direction, rng, step)

        if step >= 5 and light_vertex is not None and not hit.material.is_specular() and not light_vertex.material.is_specular():
            # Conexão bidirecional entre o vértice da câmera `hit` e o vértice da luz `light_vertex`
            w_conn = light_vertex.p - hit.p
            dist2 = glm.dot(w_conn, w_conn)
            if dist2 > 0.0:
                distance = math.sqrt(dist2)
                w_conn_normalized = w_conn / distance
                if _visible(scene, hit.p, light_vertex.p):
                    cos_camera = max(0.0, glm.dot(hit.normal, w_conn_normalized))
                    cos_light = max(0.0, glm.dot(light_vertex.normal, -w_conn_normalized))
                    if cos_camera > 0.0 and cos_light > 0.0:
                        brdf_camera = hit.material.eval(-current_ray.direction, w_conn_normalized, hit.normal)
                        brdf_light = light_vertex.material.eval(light_vertex.incoming, -w_conn_normalized, light_vertex.normal)
                        G = (cos_camera * cos_light) / dist2
                        connection_radiance = throughput * brdf_camera * G * brdf_light * light_vertex.throughput * 0.5
                        radiance += connection_radiance

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
        if step >= 6:
            throughput = glm.clamp(throughput, vec3(0.0, 0.0, 0.0), vec3(12.0, 12.0, 12.0))
        current_ray = Ray(hit.p + hit.normal * EPSILON, outgoing)
        last_vertex = hit
        last_pdf = sample.pdf
        last_specular = False

    return radiance


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
        completed_rows = height - j
        progress = int((completed_rows / max(1, height)) * 100)
        if completed_rows == 1 or completed_rows == height or progress % 5 == 0:
            sys.stdout.write(f"\rRender {progress:3d}% ({completed_rows}/{height} linhas)")
            sys.stdout.flush()
        for i in range(width):
            rng = random.Random((step * 1000003) + j * 1009 + i)
            color = vec3(0.0, 0.0, 0.0)
            for _ in range(spp):
                u = (i + rng.random()) / max(1, width - 1)
                v = (j + rng.random()) / max(1, height - 1)
                ray = camera.get_ray(u, v)
                light_vertex = _sample_light_path(scene, light, rng) if step >= 5 else None
                color += _trace_path(scene, light, ray, rng, d_max=d_max, step=step, light_vertex=light_vertex)
            pixels.append(ray_color_to_rgb(color, spp))

    sys.stdout.write("\rRender 100% (concluído)\n")
    sys.stdout.flush()

    image = Image.new("RGB", (width, height))
    image.putdata(pixels)
    if use_filter:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.4))
    return image


def render_step(step: int, width: int, height: int, spp: int, d_max: int, use_filter: bool) -> Image.Image:
    scene, light, _ = build_cornell_box(render_step=step)
    render_spp = spp
    return _render(scene, light, width, height, render_spp, d_max, step, use_filter)


def render_all_steps(width: int, height: int, spp: int, d_max: int, use_filter: bool) -> dict[int, Image.Image]:
    images: dict[int, Image.Image] = {}
    for step in range(1, 7):
        images[step] = render_step(step, width, height, spp, d_max, use_filter)
    return images
