from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Optional

import glm

from utils import EPSILON, vec3


@dataclass(slots=True)
class Ray:
    origin: glm.vec3
    direction: glm.vec3

    def at(self, t: float) -> glm.vec3:
        return self.origin + t * self.direction


@dataclass(slots=True)
class HitRecord:
    p: glm.vec3
    normal: glm.vec3
    t: float
    front_face: bool
    material: Any
    u: float = 0.0
    v: float = 0.0


def _set_face_normal(ray: Ray, outward_normal: glm.vec3) -> tuple[bool, glm.vec3]:
    front_face = glm.dot(ray.direction, outward_normal) < 0.0
    normal = outward_normal if front_face else -outward_normal
    return front_face, normal


class Hittable:
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        raise NotImplementedError


@dataclass(slots=True)
class Translated(Hittable):
    hittable: Hittable
    offset: glm.vec3

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        moved_ray = Ray(ray.origin - self.offset, ray.direction)
        hit = self.hittable.hit(moved_ray, t_min, t_max)
        if hit is None:
            return None
        return HitRecord(
            p=hit.p + self.offset,
            normal=hit.normal,
            t=hit.t,
            front_face=hit.front_face,
            material=hit.material,
            u=hit.u,
            v=hit.v,
        )


@dataclass(slots=True)
class RotateY(Hittable):
    hittable: Hittable
    angle_degrees: float

    def __post_init__(self) -> None:
        radians = glm.radians(self.angle_degrees)
        self.sin_theta = math.sin(radians)
        self.cos_theta = math.cos(radians)

    def _rotate_point(self, point: glm.vec3, inverse: bool = False) -> glm.vec3:
        if inverse:
            x = self.cos_theta * point.x + self.sin_theta * point.z
            z = -self.sin_theta * point.x + self.cos_theta * point.z
        else:
            x = self.cos_theta * point.x - self.sin_theta * point.z
            z = self.sin_theta * point.x + self.cos_theta * point.z
        return vec3(x, point.y, z)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        rotated_origin = self._rotate_point(ray.origin, inverse=True)
        rotated_direction = self._rotate_point(ray.direction, inverse=True)
        rotated_ray = Ray(rotated_origin, rotated_direction)
        hit = self.hittable.hit(rotated_ray, t_min, t_max)
        if hit is None:
            return None

        world_p = self._rotate_point(hit.p)
        world_normal = glm.normalize(self._rotate_point(hit.normal))
        front_face, normal = _set_face_normal(ray, world_normal)
        return HitRecord(
            p=world_p,
            normal=normal,
            t=hit.t,
            front_face=front_face,
            material=hit.material,
            u=hit.u,
            v=hit.v,
        )


@dataclass(slots=True)
class Sphere(Hittable):
    center: glm.vec3
    radius: float
    material: Any

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        oc = ray.origin - self.center
        a = glm.dot(ray.direction, ray.direction)
        half_b = glm.dot(oc, ray.direction)
        c = glm.dot(oc, oc) - self.radius * self.radius
        discriminant = half_b * half_b - a * c
        if discriminant < 0.0:
            return None
        root = discriminant ** 0.5
        temp = (-half_b - root) / a
        if temp < t_min or temp > t_max:
            temp = (-half_b + root) / a
            if temp < t_min or temp > t_max:
                return None
        p = ray.at(temp)
        outward = (p - self.center) / self.radius
        front_face, normal = _set_face_normal(ray, outward)
        return HitRecord(p=p, normal=normal, t=temp, front_face=front_face, material=self.material)


@dataclass(slots=True)
class Quad(Hittable):
    origin: glm.vec3
    u: glm.vec3
    v: glm.vec3
    material: Any

    def __post_init__(self) -> None:
        self.normal = glm.normalize(glm.cross(self.u, self.v))
        self.area = glm.length(glm.cross(self.u, self.v))
        self.u_len2 = glm.dot(self.u, self.u)
        self.v_len2 = glm.dot(self.v, self.v)

    def sample_point(self, rng) -> tuple[glm.vec3, glm.vec3, float]:
        su = rng.random()
        sv = rng.random()
        point = self.origin + su * self.u + sv * self.v
        pdf_area = 1.0 / max(self.area, 1e-8)
        return point, self.normal, pdf_area

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        denom = glm.dot(ray.direction, self.normal)
        if abs(denom) < 1e-8:
            return None
        t = glm.dot(self.origin - ray.origin, self.normal) / denom
        if t < t_min or t > t_max:
            return None
        p = ray.at(t)
        rel = p - self.origin
        u_coord = glm.dot(rel, self.u) / self.u_len2
        v_coord = glm.dot(rel, self.v) / self.v_len2
        if u_coord < 0.0 or u_coord > 1.0 or v_coord < 0.0 or v_coord > 1.0:
            return None
        front_face, normal = _set_face_normal(ray, self.normal)
        return HitRecord(p=p, normal=normal, t=t, front_face=front_face, material=self.material, u=u_coord, v=v_coord)


@dataclass(slots=True)
class Box(Hittable):
    minimum: glm.vec3
    maximum: glm.vec3
    material: Any

    def __post_init__(self) -> None:
        minp = self.minimum
        maxp = self.maximum
        dx = vec3(maxp.x - minp.x, 0.0, 0.0)
        dy = vec3(0.0, maxp.y - minp.y, 0.0)
        dz = vec3(0.0, 0.0, maxp.z - minp.z)
        self.sides = [
            Quad(vec3(minp.x, minp.y, minp.z), dx, dy, self.material),
            Quad(vec3(minp.x, minp.y, minp.z), dx, dz, self.material),
            Quad(vec3(minp.x, minp.y, minp.z), dy, dz, self.material),
            Quad(vec3(maxp.x, minp.y, minp.z), dy, dz, self.material),
            Quad(vec3(minp.x, maxp.y, minp.z), dx, dz, self.material),
            Quad(vec3(minp.x, minp.y, maxp.z), dx, dy, self.material),
        ]

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        closest = t_max
        result = None
        for side in self.sides:
            hit = side.hit(ray, t_min, closest)
            if hit is not None and hit.t < closest:
                closest = hit.t
                result = hit
        return result


class Scene(Hittable):
    def __init__(self, objects: Iterable[Hittable], ambient: glm.vec3, infinite_light: glm.vec3 | None = None) -> None:
        self.objects = list(objects)
        self.ambient = ambient
        self.infinite_light = infinite_light if infinite_light is not None else vec3(0.0, 0.0, 0.0)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        closest = t_max
        result = None
        for obj in self.objects:
            hit = obj.hit(ray, t_min, closest)
            if hit is not None and hit.t < closest:
                closest = hit.t
                result = hit
        return result


@dataclass(slots=True)
class AreaLight:
    quad: Quad
    emission: glm.vec3

    @property
    def area(self) -> float:
        return self.quad.area

    @property
    def normal(self) -> glm.vec3:
        return self.quad.normal

    def sample(self, rng) -> tuple[glm.vec3, glm.vec3, float, glm.vec3]:
        point, normal, pdf_area = self.quad.sample_point(rng)
        return point, normal, pdf_area, self.emission

    def pdf_from(self, point: glm.vec3, target: glm.vec3) -> float:
        direction = target - point
        dist2 = glm.dot(direction, direction)
        if dist2 <= 0.0:
            return 0.0
        direction = glm.normalize(direction)
        cos_light = max(0.0, glm.dot(-direction, self.normal))
        if cos_light <= 0.0:
            return 0.0
        return dist2 / (cos_light * self.area)


def build_cornell_box(use_infinite_light: bool = False, render_step: int | None = None) -> tuple[Scene, AreaLight, list[Hittable]]:
    from materials import EmissiveMaterial, LambertianMaterial, Material, MicrofacetMaterial, MirrorMaterial

    white = LambertianMaterial(vec3(0.73, 0.73, 0.73))
    red = LambertianMaterial(vec3(0.65, 0.05, 0.05))
    green = LambertianMaterial(vec3(0.12, 0.45, 0.15))
    metal_plastic = MicrofacetMaterial(base_color=vec3(0.86, 0.84, 0.80), metallic=0.10, roughness=0.34)
    plastic = MicrofacetMaterial(base_color=vec3(0.88, 0.86, 0.82), metallic=0.0, roughness=0.30)
    light_material = EmissiveMaterial(vec3(15.0, 15.0, 15.0))

    floor = Quad(vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), white)
    ceiling = Quad(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), white)
    back = Quad(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), white)
    left_wall = Quad(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), red)
    right_wall = Quad(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), green)

    light_quad = Quad(vec3(0.34, 0.999, 0.34), vec3(0.32, 0.0, 0.0), vec3(0.0, 0.0, 0.32), light_material)
    area_light = AreaLight(quad=light_quad, emission=vec3(15.0, 15.0, 15.0))

    if render_step == 5:
        sphere_material = Material(color=glm.vec3(0.9, 0.9, 0.9), specular=0.9,
                                   shininess=100.0, reflectivity=0.95)
        sphere_radius = 0.12
        box_material = plastic
    elif render_step == 7:
        # Esfera dourada metálica: reflecte vividamente as paredes vermelha e verde
        sphere_material = MicrofacetMaterial(base_color=vec3(1.0, 0.78, 0.34), metallic=0.95, roughness=0.05)
        sphere_radius = 0.20
        # Caixa branca suave: mostra color bleeding por iluminação indirecta difusa
        box_material = MicrofacetMaterial(base_color=vec3(0.93, 0.93, 0.93), metallic=0.0, roughness=0.15)
    else:
        sphere_material = metal_plastic
        sphere_radius = 0.22
        box_material = plastic

    sphere = Sphere(vec3(0.33, sphere_radius, 0.35), sphere_radius, sphere_material)
    local_box = Box(vec3(-0.13, 0.0, -0.135), vec3(0.13, 0.55, 0.135), box_material)
    box = Translated(RotateY(local_box, 18.0), vec3(0.75, 0.0, 0.685))

    objects = [floor, ceiling, back, left_wall, right_wall, light_quad, sphere, box]
    infinite_light = vec3(0.15, 0.15, 0.17) if use_infinite_light else vec3(0.0, 0.0, 0.0)
    # Passo 7: sem luz ambiente artificial — a iluminação indirecta (color bleeding) domina
    ambient = vec3(0.0, 0.0, 0.0) if render_step == 7 else vec3(0.015, 0.015, 0.018)
    scene = Scene(objects, ambient=ambient, infinite_light=infinite_light)
    return scene, area_light, [sphere, box]
