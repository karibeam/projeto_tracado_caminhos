from __future__ import annotations

import math
import random

import glm


Color = glm.vec3

EPSILON = 1e-4
PI = math.pi
TAU = math.tau


def vec3(x: float, y: float, z: float) -> glm.vec3:
    return glm.vec3(float(x), float(y), float(z))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def clamp_color(color: glm.vec3, minimum: float = 0.0, maximum: float = 0.999) -> glm.vec3:
    return glm.vec3(
        clamp(color.x, minimum, maximum),
        clamp(color.y, minimum, maximum),
        clamp(color.z, minimum, maximum),
    )


def luminance(color: glm.vec3) -> float:
    return 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z


def near_zero(v: glm.vec3) -> bool:
    threshold = 1e-8
    return abs(v.x) < threshold and abs(v.y) < threshold and abs(v.z) < threshold


def random_vec3(rng: random.Random, minimum: float = 0.0, maximum: float = 1.0) -> glm.vec3:
    return glm.vec3(
        rng.uniform(minimum, maximum),
        rng.uniform(minimum, maximum),
        rng.uniform(minimum, maximum),
    )


def random_in_unit_sphere(rng: random.Random) -> glm.vec3:
    while True:
        p = random_vec3(rng, -1.0, 1.0)
        if glm.dot(p, p) < 1.0:
            return p


def random_unit_vector(rng: random.Random) -> glm.vec3:
    return glm.normalize(random_in_unit_sphere(rng))


def random_cosine_direction(rng: random.Random) -> glm.vec3:
    r1 = rng.random()
    r2 = rng.random()
    z = math.sqrt(max(0.0, 1.0 - r2))
    phi = TAU * r1
    r = math.sqrt(r2)
    x = math.cos(phi) * r
    y = math.sin(phi) * r
    return glm.vec3(x, y, z)


def make_orthonormal_basis(normal: glm.vec3) -> tuple[glm.vec3, glm.vec3, glm.vec3]:
    w = glm.normalize(normal)
    a = glm.vec3(1.0, 0.0, 0.0) if abs(w.x) < 0.9 else glm.vec3(0.0, 1.0, 0.0)
    v = glm.normalize(glm.cross(w, a))
    u = glm.cross(v, w)
    return u, v, w


def to_world(local: glm.vec3, normal: glm.vec3) -> glm.vec3:
    u, v, w = make_orthonormal_basis(normal)
    return local.x * u + local.y * v + local.z * w


def reflect(vector: glm.vec3, normal: glm.vec3) -> glm.vec3:
    return vector - 2.0 * glm.dot(vector, normal) * normal


def schlick_fresnel(cosine: float, f0: glm.vec3) -> glm.vec3:
    return f0 + (glm.vec3(1.0, 1.0, 1.0) - f0) * ((1.0 - cosine) ** 5)


def ggx_distribution(normal: glm.vec3, half_vector: glm.vec3, roughness: float) -> float:
    alpha = max(0.001, roughness * roughness)
    alpha2 = alpha * alpha
    nh = max(0.0, glm.dot(normal, half_vector))
    denom = nh * nh * (alpha2 - 1.0) + 1.0
    return alpha2 / (PI * denom * denom + 1e-12)


def smith_ggx_masking(cos_theta: float, roughness: float) -> float:
    if cos_theta <= 0.0:
        return 0.0
    alpha = max(0.001, roughness * roughness)
    k = ((alpha + 1.0) * (alpha + 1.0)) / 8.0
    return cos_theta / (cos_theta * (1.0 - k) + k)


def cosine_hemisphere_pdf(cosine: float) -> float:
    return cosine / PI if cosine > 0.0 else 0.0


def balance_heuristic(pdf_a: float, pdf_b: float) -> float:
    denom = pdf_a + pdf_b
    if denom <= 0.0:
        return 0.0
    return pdf_a / denom


def power_heuristic(pdf_a: float, pdf_b: float, beta: float = 2.0) -> float:
    a = pdf_a ** beta
    b = pdf_b ** beta
    denom = a + b
    if denom <= 0.0:
        return 0.0
    return a / denom


def ray_color_to_rgb(color: glm.vec3, spp: int) -> tuple[int, int, int]:
    scale = 1.0 / max(1, spp)
    mapped = glm.vec3(
        math.sqrt(max(0.0, color.x * scale)),
        math.sqrt(max(0.0, color.y * scale)),
        math.sqrt(max(0.0, color.z * scale)),
    )
    clamped = clamp_color(mapped)
    return (
        int(255.999 * clamped.x),
        int(255.999 * clamped.y),
        int(255.999 * clamped.z),
    )


def aces_tone_map(x: float) -> float:
    x = max(0.0, x)
    return min(1.0, (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14))


def ray_color_to_rgb_hq(color: glm.vec3, spp: int) -> tuple[int, int, int]:
    """ACES filmic tone mapping + gamma 2.2 para o render final de alta qualidade."""
    scale = 1.0 / max(1, spp)
    gamma = 1.0 / 2.2
    r = max(0.0, min(1.0, aces_tone_map(color.x * scale) ** gamma))
    g = max(0.0, min(1.0, aces_tone_map(color.y * scale) ** gamma))
    b = max(0.0, min(1.0, aces_tone_map(color.z * scale) ** gamma))
    return (int(255.999 * r), int(255.999 * g), int(255.999 * b))
