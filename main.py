from __future__ import annotations

import argparse
from pathlib import Path

from integrators import render_all_steps, render_step


OUTPUT_FILENAMES = {
    "1": "passo1_path_tracing_basico.png",
    "2": "passo2_roleta_russa.png",
    "3": "passo3_mis.png",
    "4": "passo4_microfacets.png",
    "5": "passo5_bdpt.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motor de Renderização Path Tracing (Cornell Box)")
    parser.add_argument("--step", type=str, choices=["1", "2", "3", "4", "5"], default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--spp", type=int, default=25)
    parser.add_argument("--d_max", type=int, default=4)
    parser.add_argument("--use_filter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    steps = [args.step] if args.step else ["1", "2", "3", "4", "5"]
    for step in steps:
        print(f"A executar o Passo {step}...")

        render_spp = args.spp
        render_d_max = args.d_max
        render_use_filter = args.use_filter
        render_step_id = int(step)

        lbl = f"d_max={render_d_max}, SPP={render_spp}"
        image = render_step(
            render_step_id,
            args.width,
            args.height,
            render_spp,
            render_d_max,
            render_use_filter,
            label=lbl,
        )
        target = output_dir / OUTPUT_FILENAMES[step]
        image.save(target)
        print(f"Imagem guardada em {target}")


if __name__ == "__main__":
    main()
