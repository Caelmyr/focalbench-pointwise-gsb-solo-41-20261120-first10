"""Deterministic synthetic targets that exercise the normal image pipeline."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .alignment import transform_layer
from .imaging import frame_input
from .models import CaptureScale, RunSettings

WIDTH, HEIGHT = 690, 462


@dataclass(frozen=True)
class Demo:
    key: str
    label: str
    purpose: str
    position_text: str
    positions: tuple[float, ...]
    seed: int
    pixel_um: float


DEMOS = {
    "circuit": Demo(
        "circuit", "Soldered controller", "Copper traces, resistor bridge and inspection vias at different heights.",
        "10 samples at unequal rail intervals", (-20, 0, 19, 40, 64, 89, 121, 152, 184, 222), 6028, 8.2,
    ),
    "shim": Demo(
        "shim", "Machined shim and gauge", "Brushed metal, a stepped shim and raised engraved studs.",
        "9 samples with measured breathing", (0, 22, 44, 67, 92, 118, 146, 175, 206), 98221, 6.5,
    ),
}


def demo_catalog() -> list[dict]:
    return [
        {"id": demo.key, "name": demo.label, "description": demo.purpose,
         "sampling": demo.position_text, "count": len(demo.positions), "synthetic": True}
        for demo in DEMOS.values()
    ]


def coords() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:HEIGHT, :WIDTH].astype(np.float32)
    return y, x


def grain(seed: int, *, directional: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (HEIGHT, WIDTH)).astype(np.float32)
    if directional:
        stretched = ndimage.gaussian_filter(base, sigma=(0.6, 4.2), mode="reflect") * 2.5
        pattern = stretched + base * 0.26
    else:
        medium = ndimage.gaussian_filter(base, 2.8, mode="reflect")
        coarse = ndimage.gaussian_filter(base, 14, mode="reflect")
        pattern = 0.69 * base + medium * 1.7 + coarse * 4
    return pattern


def font(size: int):
    return ImageFont.load_default(size=size)


class Target:
    def __init__(self, base_color: tuple[int, int, int], height: float):
        self.color = Image.new("RGB", (WIDTH, HEIGHT), base_color)
        self.height = Image.new("F", self.color.size, height)
        self.paint = ImageDraw.Draw(self.color)
        self.rail = ImageDraw.Draw(self.height)

    def rectangle(self, box, color, position, *, radius=0, outline=None, width=1):
        if radius:
            self.paint.rounded_rectangle(box, radius=radius, fill=color, outline=outline, width=width)
            self.rail.rounded_rectangle(box, radius=radius, fill=position)
        else:
            self.paint.rectangle(box, fill=color, outline=outline, width=width)
            self.rail.rectangle(box, fill=position)

    def ellipse(self, box, color, position, *, outline=None, width=1):
        self.paint.ellipse(box, fill=color, outline=outline, width=width)
        self.rail.ellipse(box, fill=position)

    def line(self, points, color, position, *, width=1):
        self.paint.line(points, fill=color, width=width, joint="curve")
        self.rail.line(points, fill=position, width=width, joint="curve")

    def text(self, point, text, color, position, *, size=15):
        face = font(size)
        self.paint.text(point, text, fill=color, font=face)
        self.rail.text(point, text, fill=position, font=face)

    def final(self, seed, directional=False):
        y, x = coords()
        scene = np.asarray(self.color, dtype=np.float32)
        depth = np.asarray(self.height, dtype=np.float32).copy()
        hot = np.exp(-((x - 184) / 470) ** 2 - ((y - 120) / 365) ** 2)
        illumination = (0.2 * hot + 0.79).astype(np.float32)
        texture = grain(seed, directional=directional) * (5.1 if directional else 4.4)
        color = scene * illumination[..., None] + texture[..., None]
        return np.clip(color, 0, 255), depth


def via(target: Target, px: int, py: int, size: int, rail: float):
    end = (px + size, py + size)
    target.ellipse((px - size, py - size, *end), (21, 41, 38), rail - 2)
    radius = size - 3
    target.ellipse((px - radius, py - radius, px + radius, py + radius), (191, 147, 91), rail)
    depth = max(2, radius - 3)
    target.ellipse((px - depth, py - depth, px + depth, py + depth), (16, 28, 31), rail - 15)


def resistor(target: Target, x: int, y: int, key: str, position: float):
    target.rectangle((x - 32, y - 10, x + 31, y + 10), (24, 39, 37), 25, radius=4)
    target.rectangle((x - 30, y - 9, x + 30, y + 9), (176, 169, 144), position - 10, radius=3)
    target.rectangle((x - 18, y - 10, x + 18, y + 10), (37, 44, 44), position, radius=2)
    target.line(((x - 28, y - 5), (x - 22, y - 3)), (237, 227, 192), position - 10, width=2)
    target.line(((x + 21, y - 4), (x + 27, y - 6)), (225, 218, 184), position - 10, width=2)
    target.text((x - 11, y - 6), key, (205, 197, 173), position, size=11)


def circuit_board(demo: Demo) -> tuple[np.ndarray, np.ndarray]:
    target = Target((29, 51, 51), 1)
    target.rectangle((29, 29, WIDTH - 31, HEIGHT - 28), (50, 94, 81), 26, radius=26,
                     outline=(104, 130, 107), width=4)
    target.rectangle((38, 39, WIDTH - 40, HEIGHT - 39), (35, 76, 67), 30, radius=22)
    for n in range(9):
        offset = n * 11
        route = [(84, 100 + offset), (170 + offset, 100 + offset),
                 (250 + offset, 180 + offset), (350, 180 + offset)]
        target.line(route, (53, 104, 88), 34, width=6)
        route = [(527 + offset, 185), (527 + offset, 267 + offset),
                 (479 + offset, 315 + offset), (351, 315 + offset)]
        target.line(route, (58, 110, 85), 30, width=5)
    for x, y in ((66, 68), (622, 70), (66, 390), (623, 390)):
        via(target, x, y, 21, 47)
    for y in (128, 156, 185, 213, 241, 269, 297, 326):
        via(target, 93, y, 9, 38)
        via(target, 598, y, 8, 42)
    for y in (331, 358, 389):
        for x in range(150, 245, 23):
            via(target, x, y, 6, 37)
    for index in range(15):
        left = 249 + index * 15
        target.rectangle((left, 209, left + 8, 239), (196, 185, 153), 97, radius=2,
                         outline=(232, 211, 170))
        target.rectangle((left, 296, left + 8, 324), (171, 162, 134), 99, radius=2,
                         outline=(213, 200, 157))
    target.rectangle((226, 225, 492, 309), (26, 41, 34), 44, radius=13)
    target.rectangle((234, 233, 484, 302), (51, 51, 49), 165, radius=11,
                     outline=(98, 102, 91), width=2)
    target.line(((246, 295), (469, 295)), (82, 86, 76), 164, width=2)
    target.ellipse((250, 246, 261, 257), (131, 130, 100), 173)
    target.text((281, 248), "FB-261  SENSE", (189, 188, 163), 175, size=21)
    target.text((282, 275), "U4     6 2 7", (129, 132, 112), 171, size=13)
    for x, y, value, z in (
        (148, 224, "472", 112), (149, 254, "100", 118), (147, 281, "221", 123),
        (528, 123, "220", 93), (561, 123, "101", 93), (553, 353, "103", 104),
        (539, 379, "333", 110),
    ):
        resistor(target, x, y, value, z)
    for y in range(91, 159, 13):
        target.rectangle((341, y, 413, y + 6), (196, 162, 112), 59, radius=3)
    target.rectangle((331, 79, 428, 167), (63, 55, 45), 116, radius=10,
                     outline=(127, 109, 78), width=3)
    for x in range(343, 420, 12):
        target.line(((x, 90), (x, 143)), (121, 102, 66), 118, width=3)
    target.text((334, 177), "J2   RAIL", (174, 183, 147), 35, size=15)
    target.text((118, 58), "CONTROL  /  REV D", (192, 197, 151), 29, size=19)
    target.text((394, 404), "FOCUS TARGET 01", (186, 197, 159), 27, size=14)
    target.text((118, 309), "TP-A", (176, 187, 157), 31, size=12)
    for x in range(300, 386, 12):
        top = 402 if x % 24 == 12 else 409
        target.line(((x, top), (x, 422)), (181, 190, 162), 28, width=2)
    return target.final(demo.seed)


def gauge_shim(demo: Demo) -> tuple[np.ndarray, np.ndarray]:
    target = Target((66, 71, 76), 8)
    target.rectangle((28, 29, 662, 434), (102, 110, 116), 31, radius=23,
                     outline=(159, 165, 169), width=5)
    target.rectangle((57, 59, 632, 404), (134, 138, 141), 36, radius=13,
                     outline=(84, 92, 98), width=3)
    rng = np.random.default_rng(demo.seed)
    for index in range(200):
        start_x = int(rng.integers(70, 622))
        y = int(rng.integers(66, 394))
        length = int(rng.integers(5, 52))
        tone = int(rng.integers(117, 158))
        target.line(((start_x, y), (min(625, start_x + length), y)), (tone, tone + 1, tone + 5), 37)
    target.rectangle((109, 96, 528, 361), (77, 82, 82), 68, radius=32,
                     outline=(207, 199, 178), width=3)
    target.rectangle((120, 101, 504, 339), (183, 168, 140), 91, radius=24,
                     outline=(244, 228, 196), width=4)
    target.rectangle((214, 120, 484, 298), (160, 143, 116), 145, radius=18,
                     outline=(220, 202, 166), width=4)
    target.rectangle((337, 139, 472, 260), (112, 100, 86), 181, radius=13,
                     outline=(200, 180, 148), width=3)
    for x in range(180, 461, 11):
        lower = 321 if x % 22 else 311
        target.line(((x, lower), (x, 332)), (80, 69, 57), 91, width=2)
    for index, x in enumerate(range(191, 432, 55)):
        target.text((x - 8, 293 if x < 210 else 304), str(index * 5), (81, 70, 59), 91, size=15)
    for x, y, z in ((143, 160, 96), (291, 169, 151), (410, 202, 196), (559, 300, 47)):
        via(target, x, y, 24, z)
        target.line(((x - 12, y), (x + 12, y)), (90, 91, 83), z - 14, width=4)
        target.line(((x, y - 12), (x, y + 12)), (94, 94, 81), z - 14, width=4)
    target.text((68, 70), "Z / GAUGE 06", (74, 80, 82), 37, size=18)
    target.text((130, 222), "0.40", (78, 65, 49), 95, size=22)
    target.text((228, 250), "1.20", (83, 70, 55), 148, size=20)
    target.text((538, 372), "µm", (74, 83, 92), 37, size=16)
    return target.final(demo.seed, directional=True)


def optically_blur(scene: np.ndarray, heights: np.ndarray, position: float) -> np.ndarray:
    distance = np.abs(heights - position)
    radii = (0.30, 0.82, 1.67, 2.80, 4.6, 7.0)
    band_step = 27
    output = np.empty_like(scene)
    band = np.clip(distance / band_step, 0, len(radii) - 1)
    weights = [np.maximum(1 - np.abs(index - band), 0) for index in range(len(radii))]
    for channel in range(3):
        source = scene[..., channel]
        output[..., channel] = 0
        for weight, sigma in zip(weights, radii):
            if not np.any(weight):
                continue
            blurred = ndimage.gaussian_filter(source, sigma, mode="reflect")
            output[..., channel] += blurred * weight
    return output


def encode_demo_capture(scene, heights, demo: Demo, index: int) -> bytes:
    position = demo.positions[index]
    optically_sampled = optically_blur(scene, heights, position)
    middle = (len(demo.positions) - 1) / 2
    phase = index - middle
    escala = 1 + phase * (0.0084 if demo.key == "shim" else 0.005)
    dx = float(np.sin(index * 1.4) * 3.7 + phase * 1.5)
    dy = float(np.cos(index * 1.21) * 2.9 - phase * 0.56)
    moving = np.stack([
        transform_layer(optically_sampled[..., band], dx, dy, escala, order=1, cval=24)
        for band in range(3)
    ], axis=2)
    rng = np.random.default_rng(demo.seed + 183 * index)
    lighting = 1 + 0.026 * np.sin(index * 0.91)
    moving = moving * lighting + rng.normal(0, 0.75, moving.shape)
    rgb = np.clip(moving, 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", compress_level=4)
    return buf.getvalue()


def create_demo(library, demo_id: str, title: str) -> dict:
    demo = DEMOS[demo_id]
    if demo.key == "circuit":
        texture, height = circuit_board(demo)
    else:
        texture, height = gauge_shim(demo)
    records = []
    for index, position in enumerate(demo.positions):
        data = encode_demo_capture(texture, height, demo, index)
        record = frame_input(data, f"{demo.key}-{index + 1:02d}.png", position, library)
        records.append(record)
    scale = CaptureScale(pixel_um=demo.pixel_um, rail_unit="µm", rail_offset=0.0)
    return library.new_project(title, RunSettings(scale=scale), demo=demo_id, captures=records)
