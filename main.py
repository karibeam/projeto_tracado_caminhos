from __future__ import annotations

import argparse
from pathlib import Path

from integrators import render_all_steps, render_step


OUTPUT_FILENAMES = {
    "1": "passo1_path_tracing_basico.png",
    "2": "passo2_roleta_russa.png",
    "3": "passo3_mis.png",
    "3.1": "passo3_1_teste_mis_microfacets.png",
    "4": "passo4_microfacets.png",
    "5": "passo5_bdpt.png",
    "5.1": "passo5_1_realista.png",
    "6": "passo6_luz_infinita_teste.png",
    "7": None,  # gera dois ficheiros: passo7_dmax4 e passo7_dmax8
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motor de Renderização Path Tracing (Cornell Box)")
    parser.add_argument("--step", type=str, choices=["1", "2", "3", "3.1", "4", "5", "5.1", "6", "7"], default=None)
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

    steps = [args.step] if args.step else ["1", "2", "3", "4", "5", "6"]
    for step in steps:
        print(f"A executar o Passo {step}...")

        if step == "7":
            spp_7 = args.spp if args.spp != 25 else 128
            dmax_7 = args.d_max
            lbl = f"d_max={dmax_7}, SPP={spp_7}"
            image = render_step(7, args.width, args.height, spp_7, dmax_7, args.use_filter, label=lbl)
            fname = f"passo7_dmax{dmax_7}_spp{spp_7}.png"
            target = output_dir / fname
            image.save(str(target))
            print(f"Imagem guardada em {target}")
            continue

        render_spp = 64 if step == "3.1" else (128 if step == "5.1" else args.spp)
        render_d_max = 4 if step == "3.1" else (6 if step == "5.1" else args.d_max)
        render_use_filter = True if step == "5.1" else args.use_filter
        if step == "6":
            render_spp = max(render_spp, 96)
            render_d_max = max(render_d_max, 6)
        render_step_id = 3 if step == "3.1" else (5 if step == "5.1" else int(step))
        image = render_step(render_step_id, args.width, args.height, render_spp, render_d_max, render_use_filter)
        target = output_dir / OUTPUT_FILENAMES[step]
        image.save(target)
        print(f"Imagem guardada em {target}")


if __name__ == "__main__":
    main()
