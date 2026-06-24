from __future__ import annotations

from dataclasses import dataclass
import math
import random

import glm

from scene import HitRecord, Ray
from utils import (
    PI,
    balance_heuristic,
    cosine_hemisphere_pdf,
    ggx_distribution,
    luminance,
    near_zero,
    random_cosine_direction,
    reflect,
    schlick_fresnel,
    smith_ggx_masking,
    to_world,
    vec3,
)


@dataclass(slots=True)
class BsdfSample:
    direction: glm.vec3
    attenuation: glm.vec3
    pdf: float
    specular: bool = False


class Material:
    def emitted(self, hit: HitRecord, outgoing: glm.vec3) -> glm.vec3:
        return vec3(0.0, 0.0, 0.0)

    def sample(self, incoming: glm.vec3, normal: glm.vec3, rng: random.Random) -> BsdfSample:
        raise NotImplementedError

    def eval(self, incoming: glm.vec3, outgoing: glm.vec3, normal: glm.vec3) -> glm.vec3:
        raise NotImplementedError

    def pdf(self, incoming: glm.vec3, outgoing: glm.vec3, normal: glm.vec3) -> float:
        raise NotImplementedError

    def is_specular(self) -> bool:
        return False


@dataclass(slots=True)
class LambertianMaterial(Material):
    albedo: glm.vec3

    def sample(self, incoming: glm.vec3, normal: glm.vec3, rng: random.Random) -> BsdfSample:
        local = random_cosine_direction(rng)
        direction = to_world(local, normal)
        pdf = cosine_hemisphere_pdf(max(0.0, glm.dot(normal, glm.normalize(direction))))
        return BsdfSample(direction=glm.normalize(direction), attenuation=self.albedo, pdf=pdf)

    def eval(self, incoming: glm.vec3, outgoing: glm.vec3, normal: glm.vec3) -> glm.vec3:
        return self.albedo / PI

    def pdf(self, incoming: glm.vec3, outgoing: glm.vec3, normal: glm.vec3) -> float:
        cosine = max(0.0, glm.dot(normal, glm.normalize(outgoing)))
        return cosine_hemisphere_pdf(cosine)


@dataclass(slots=True)
class EmissiveMaterial(Material):
    emission: glm.vec3

    def emitted(self, hit: HitRecord, outgoing: glm.vec3) -> glm.vec3:
        return self.emission

    def sample(self, incoming: glm.vec3, normal: glm.vec3, rng: random.Random) -> BsdfSample:
        return BsdfSample(direction=normal, attenuation=vec3(0.0, 0.0, 0.0), pdf=0.0, specular=True)

    def eval(self, incoming: glm.vec3, outgoing: glm.vec3, normal: glm.vec3) -> glm.vec3:
        return vec3(0.0, 0.0, 0.0)

    def pdf(self, incoming: glm.vec3, outgoing: glm.vec3, normal: glm.vec3) -> float:
        return 0.0

    def is_specular(self) -> bool:
        return True


@dataclass(slots=True)
class MirrorMaterial(Material):
    albedo: glm.vec3

    def sample(self, incoming: glm.vec3, normal: glm.vec3, rng: random.Random) -> BsdfSample:
        direction = reflect(glm.normalize(incoming), normal)
        return BsdfSample(direction=glm.normalize(direction), attenuation=self.albedo, pdf=1.0, specular=True)

    def eval(self, incoming: glm.vec3, outgoing: glm.vec3, normal: glm.vec3) -> glm.vec3:
        return self.albedo

    def pdf(self, incoming: glm.vec3, outgoing: glm.vec3, normal: glm.vec3) -> float:
        return 1.0

    def is_specular(self) -> bool:
        return True


@dataclass(slots=True)
class MicrofacetMaterial(Material):
    base_color: glm.vec3
    metallic: float
    roughness: float

    def _sample_ggx_half_vector(self, normal: glm.vec3, rng: random.Random) -> glm.vec3:
        normal = glm.normalize(normal)
        alpha = max(0.001, self.roughness * self.roughness)
        u1 = max(1e-6, rng.random())
        u2 = rng.random()
        tan2_theta = (alpha * alpha) * u1 / max(1e-6, 1.0 - u1)
        cos_theta = 1.0 / math.sqrt(1.0 + tan2_theta)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi = 2.0 * PI * u2
        local = glm.vec3(math.cos(phi) * sin_theta, math.sin(phi) * sin_theta, cos_theta)
        return glm.normalize(to_world(local, normal))

    def _ggx_specular_pdf(self, incoming: glm.vec3, outgoing: glm.vec3, normal: glm.vec3) -> float:
        incoming = glm.normalize(incoming)
        outgoing = glm.normalize(outgoing)
        normal = glm.normalize(normal)
        ndotl = max(0.0, glm.dot(normal, outgoing))
        if ndotl <= 0.0:
            return 0.0
        half_vector = glm.normalize(incoming + outgoing)
        ndoth = max(0.0, glm.dot(normal, half_vector))
        vdoth = max(1e-6, abs(glm.dot(incoming, half_vector)))
        alpha = max(0.001, self.roughness * self.roughness)
        alpha2 = alpha * alpha
        denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0
        d = alpha2 / (PI * denom * denom + 1e-12)
        return (d * ndoth) / max(1e-6, 4.0 * vdoth)

    def sample(self, incoming: glm.vec3, normal: glm.vec3, rng: random.Random) -> BsdfSample:
        incoming = glm.normalize(incoming)
        normal = glm.normalize(normal)
        diffuse_weight = max(0.15, 1.0 - self.metallic)
        specular_weight = min(0.85, 0.25 + 0.5 * self.metallic + 0.2 * (1.0 - self.roughness))
        total_weight = diffuse_weight + specular_weight
        specular_prob = specular_weight / max(1e-6, total_weight)

        if rng.random() < specular_prob:
            half_vector = self._sample_ggx_half_vector(normal, rng)
            direction = reflect(-incoming, half_vector)
            if glm.dot(direction, normal) <= 0.0:
                direction = reflect(-incoming, normal)
        else:
            local = random_cosine_direction(rng)
            direction = to_world(local, normal)

        direction = glm.normalize(direction)
        pdf = self.pdf(incoming, direction, normal)
        return BsdfSample(direction=direction, attenuation=self.base_color, pdf=pdf)

    def eval(self, incoming: glm.vec3, outgoing: glm.vec3, normal: glm.vec3) -> glm.vec3:
        incoming = glm.normalize(incoming)
        outgoing = glm.normalize(outgoing)
        normal = glm.normalize(normal)
        ndotl = max(0.0, glm.dot(normal, outgoing))
        ndotv = max(0.0, glm.dot(normal, incoming))
        if ndotl <= 0.0 or ndotv <= 0.0:
            return vec3(0.0, 0.0, 0.0)

        half_vector = glm.normalize(incoming + outgoing)
        ndoth = max(0.0, glm.dot(normal, half_vector))
        vdoth = max(0.0, glm.dot(incoming, half_vector))

        diffuse = (1.0 - self.metallic) * self.base_color / PI
        f0 = glm.mix(vec3(0.04, 0.04, 0.04), self.base_color, self.metallic)
        distribution = ggx_distribution(normal, half_vector, self.roughness)
        fresnel = schlick_fresnel(vdoth, f0)
        geometry = smith_ggx_masking(ndotl, self.roughness) * smith_ggx_masking(ndotv, self.roughness)
        specular = (distribution * geometry) / max(1e-6, 4.0 * ndotl * ndotv)
        return diffuse + fresnel * specular

    def pdf(self, incoming: glm.vec3, outgoing: glm.vec3, normal: glm.vec3) -> float:
        incoming = glm.normalize(incoming)
        outgoing = glm.normalize(outgoing)
        normal = glm.normalize(normal)
        ndotl = max(0.0, glm.dot(normal, outgoing))
        if ndotl <= 0.0:
            return 0.0
        diffuse_weight = max(0.15, 1.0 - self.metallic)
        specular_weight = min(0.85, 0.25 + 0.5 * self.metallic + 0.2 * (1.0 - self.roughness))
        total_weight = diffuse_weight + specular_weight
        specular_prob = specular_weight / max(1e-6, total_weight)
        diffuse_pdf = cosine_hemisphere_pdf(ndotl)
        return specular_prob * self._ggx_specular_pdf(incoming, outgoing, normal) + (1.0 - specular_prob) * diffuse_pdf
