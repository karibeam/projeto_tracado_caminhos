from __future__ import annotations

from dataclasses import dataclass
import math

import glm

from scene import Ray


@dataclass(slots=True)
class Camera:
    origin: glm.vec3
    lower_left_corner: glm.vec3
    horizontal: glm.vec3
    vertical: glm.vec3
    u: glm.vec3
    v: glm.vec3
    w: glm.vec3

    @classmethod
    def look_at(
        cls,
        lookfrom: glm.vec3,
        lookat: glm.vec3,
        vup: glm.vec3,
        vfov: float,
        aspect_ratio: float,
    ) -> "Camera":
        theta = math.radians(vfov)
        h = math.tan(theta / 2.0)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        w = glm.normalize(lookfrom - lookat)
        u = glm.normalize(glm.cross(vup, w))
        v = glm.cross(w, u)

        origin = lookfrom
        horizontal = viewport_width * u
        vertical = viewport_height * v
        lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - w

        return cls(
            origin=origin,
            lower_left_corner=lower_left_corner,
            horizontal=horizontal,
            vertical=vertical,
            u=u,
            v=v,
            w=w,
        )

    def get_ray(self, s: float, t: float) -> Ray:
        direction = self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin
        return Ray(self.origin, glm.normalize(direction))
