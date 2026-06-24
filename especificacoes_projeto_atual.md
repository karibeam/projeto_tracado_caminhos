# Especificações do Motor de Traçado de Caminhos — Estado Actual

Motor de renderização baseado em física (PBR) implementado em **Python**, usando **pyGLM** para álgebra vetorial e **Pillow** para gestão de imagem. Cena de teste: **Cornell Box** clássica.

---

## 1. Estrutura de Ficheiros

```
projeto2/
├── main.py          — CLI com argparse; orquestra os passos a executar
├── scene.py         — Primitivas geométricas, materiais e montagem da cena
├── camera.py        — Câmara perspectiva (look-at + geração de raios)
├── materials.py     — BSDFs: Lambertiano, Microfacetas, Emissivo, Espelho
├── integrators.py   — Algoritmos de renderização para cada passo (1–7)
├── utils.py         — Utilitários matemáticos, amostragem, tone mapping
└── output/          — Imagens geradas
```

---

## 2. Parâmetros de Linha de Comando (CLI)

| Argumento      | Tipo    | Padrão | Descrição |
|----------------|---------|--------|-----------|
| `--step`       | string  | None   | Passo a executar: `1`–`7` (ou `3.1`, `5.1`). Se omitido, executa 1–6. |
| `--width`      | int     | 512    | Largura da imagem em píxeis |
| `--height`     | int     | 512    | Altura da imagem em píxeis |
| `--spp`        | int     | 25     | Amostras por píxel (Samples Per Pixel) |
| `--d_max`      | int     | 4      | Profundidade máxima dos caminhos (bounces) |
| `--use_filter` | flag    | False  | Aplica filtro de pós-processamento na imagem final |

**Exemplos de uso:**
```bash
python main.py --step 7 --width 512 --height 512          # Passo 7 com defeitos (128 SPP)
python main.py --step 5 --spp 64 --d_max 4                # Passo 5 com 64 SPP
python main.py --step 3 --spp 100 --use_filter            # Passo 3 com filtro
python main.py --width 800 --height 800 --spp 50          # Todos os passos (1–6)
```

---

## 3. Cena: Cornell Box

A cena mantém-se constante em todos os passos para permitir comparação directa dos integradores.

### Geometria

| Elemento         | Tipo  | Posição / Dimensões        | Material        |
|------------------|-------|----------------------------|-----------------|
| Piso             | Quad  | y=0, de (0,0,0) a (1,0,1) | Branco Lambert. |
| Tecto            | Quad  | y=1, de (0,1,0) a (1,1,1) | Branco Lambert. |
| Parede de fundo  | Quad  | z=1                        | Branco Lambert. |
| Parede esquerda  | Quad  | x=0                        | Vermelho Lambert. (0.65, 0.05, 0.05) |
| Parede direita   | Quad  | x=1                        | Verde Lambert. (0.12, 0.45, 0.15)   |
| Luz de área      | Quad  | Tecto, centro (~0.34–0.66) | Emissivo: 15.0 W/sr (branco) |
| Esfera           | Sphere| Centro variável por passo  | Varia por passo |
| Caixa            | Box   | Canto direito, rotacionada 18° | Varia por passo |

### Câmara

- **Posição:** (0.5, 0.5, −2.2) — olha para (0.5, 0.5, 0.5)
- **FOV vertical:** 40°
- **Up vector:** (0, 1, 0)

### Iluminação

- **Luz de área rectangular:** emissão 15.0 W/sr, posicionada no tecto
- **Luz ambiente:** valor subtil constante `(0.015, 0.015, 0.018)` — ausente no Passo 7

---

## 4. Materiais Implementados

### LambertianMaterial
- **BRDF:** $f_r = \rho / \pi$
- **Amostragem:** direcção cosseno-ponderada no hemisfério (Cosine-Weighted Hemisphere Sampling)
- **PDF:** $p(\omega) = \cos\theta / \pi$
- Usado nas paredes da Cornell Box

### EmissiveMaterial
- Devolve a emissão configurada (`emission: vec3`)
- Não produz amostras de BRDF (especular puro, sem bounce)
- Usado na luz de área do tecto

### MicrofacetMaterial (Cook-Torrance)
- **BRDF:** $f_r(v, l) = \frac{\rho_d}{\pi} + \frac{D(h)\, F(v,h)\, G(l,v)}{4\langle n \cdot l \rangle \langle n \cdot v \rangle}$
- **D — Distribuição GGX:** $D(h) = \frac{\alpha^2}{\pi ((\alpha^2-1)\cos^2\theta + 1)^2}$
- **F — Fresnel de Schlick:** $F(v,h) = F_0 + (1-F_0)(1 - \langle v \cdot h \rangle)^5$
- **G — Smith/GGX Masking:** produto das visibilidades de entrada e saída
- **Amostragem mista:** escolhe entre lóbo especular GGX e difuso cosseno com probabilidade proporcional ao peso de cada componente
- **Parâmetros:** `base_color`, `metallic` (0–1), `roughness` (0–1)

### MirrorMaterial
- Reflexão especular perfeita: $\omega_r = \omega_i - 2(\omega_i \cdot n)n$
- Sem difusão, PDF = 1
- Usado no passo 5 para a esfera espelhada

---

## 5. Passos de Renderização

### Passo 1 — Path Tracing Básico
- **Integrador:** Monte Carlo simples sem técnicas de redução de variância
- **BRDF:** Lambertiana nas paredes; materiais básicos na esfera e caixa
- **Iluminação indirecta:** acumulada pelos bounces aleatórios
- **Encerramento forçado:** quando `bounce == d_max`, amostra directamente a luz para fechar o estimador
- **Saída:** `passo1_path_tracing_basico.png`
- **Ruído esperado:** elevado a 25 SPP; color bleeding (sangria de cor) visível

### Passo 2 — Roleta Russa
- **Acrescenta ao Passo 1:** terminação probabilística dos caminhos
- **Mecânica:** a partir do bounce 2, calcula $q = \max(0.05,\, 1 - \text{luminância}(\text{throughput}))$; se o número aleatório $<q$, termina; caso contrário divide o throughput por $(1-q)$ (compensação *unbiased*)
- **Benefício:** equilibra melhor o orçamento computacional — caminhos com baixo throughput terminam mais cedo, libertando amostras para outros
- **Saída:** `passo2_roleta_russa.png`

### Passo 3 — Múltiplas Amostragens por Importância (MIS)
- **Acrescenta ao Passo 2:** amostragem directa da fonte de luz + combinação via MIS
- **Estratégias combinadas:**
  1. Amostragem da BRDF (direcção aleatória conforme o lóbo)
  2. Amostragem directa da superfície da luz (*shadow ray*)
- **Heurística Balanceada:** $w_s(x) = \frac{n_s\, p_s(x)}{\sum_i n_i\, p_i(x)}$
- **Resultado:** eliminação de *fireflies* causados por caminhos que acidentalmente acertam na luz; redução significativa de ruído nas zonas de iluminação directa
- **Saída:** `passo3_mis.png`

### Passo 3.1 — Teste MIS + Microfacetas
- Variante interna: MIS com materiais microfacetas activados a 64 SPP

### Passo 4 — Microfacetas (Cook-Torrance com GGX)
- **Acrescenta ao Passo 3:** substituição da BRDF Lambertiana por Cook-Torrance
- **Esfera:** `MicrofacetMaterial(base_color=(0.86,0.84,0.80), metallic=0.10, roughness=0.34)` — aspecto metálico-plástico
- **Caixa:** `MicrofacetMaterial(base_color=(0.88,0.86,0.82), metallic=0.0, roughness=0.30)` — plástico brilhante
- **Saída:** `passo4_microfacets.png`

### Passo 5 — BDPT (Traçado de Caminhos Bidirecional)
- **Acrescenta ao Passo 4:** conexão de sub-caminhos de câmara e de luz
- **Mecânica:**
  1. Traça um sub-caminho a partir da **câmara** ($p_i$)
  2. Amostra um **vértice de luz** a partir da fonte de área ($q_j$): posição, normal, throughput inicial
  3. Liga ambos via **raio de visibilidade** e calcula o termo geométrico $G = \frac{\cos\theta_c \cdot \cos\theta_l}{d^2}$
  4. Acumula a contribuição bidirecional: $L = \beta_c \cdot f_c \cdot G \cdot f_l \cdot \beta_l$
- **Esfera no passo 5:** espelhada (`Material(reflectivity=0.95, shininess=100)`) para demonstrar caminhos especulares complexos
- **Mitigação de fireflies:** $G$ limitado a 25.0; distância mínima entre vértices de 1 cm; clamp por amostra (luminância > 20.0)
- **Saída:** `passo5_bdpt.png`

### Passo 5.1 — Variante Realista
- Passo 5 com 128 SPP, d_max=6, filtro Gaussiano activado

### Passo 6 — Power Heuristic + Amostras Directas Múltiplas
- **Acrescenta ao Passo 5:** substituição da Heurística Balanceada pela **Heurística de Potência** ($\beta=2$)
  - $w_s = \frac{p_s^2}{p_s^2 + p_t^2}$ — favorece mais a estratégia de maior PDF; reduz variância em fontes de luz pequenas
- **4 amostras de luz directa** por bounce (em vez de 1)
- **Clamp de throughput:** limitado a 12.0 por canal para evitar explosões de energia
- **Saída:** `passo6_luz_infinita_teste.png`

### Passo 7 — Render Final de Alta Qualidade
- **Todas as técnicas activas** (acumula passos 1–6)
- **Diferenças chave em relação ao Passo 6:**
  - **8 amostras de luz directa** por bounce (dobro do passo 6)
  - **Clamp de throughput:** reduzido a 8.0 (mais conservador)
  - **ACES Filmic Tone Mapping** + gamma 2.2 (em vez de simples $\sqrt{x}$)
  - **Clamp por amostra:** luminância > 20.0 escala a amostra antes de acumular
  - **G do BDPT:** limitado a 25.0 + distância mínima $10^{-4}$ entre vértices
  - **Renderização paralela** com `multiprocessing.Pool` (todos os núcleos da CPU)
  - **Sem luz ambiente** — a iluminação indirecta domina naturalmente
- **Materiais do Passo 7:**
  - Esfera: ouro metálico `(1.0, 0.78, 0.34)`, metallic=0.95, roughness=0.05 — reflecte vividamente as paredes coloridas
  - Caixa: plástico branco suave `(0.93, 0.93, 0.93)`, metallic=0.0, roughness=0.15 — mostra color bleeding por difusão
- **SPP padrão:** 128 | **d_max padrão:** 4
- **Saídas:** `passo7_dmax{N}_spp{M}.png`

---

## 6. Técnicas Transversais

### Amostragem de Monte Carlo
Estimador base: $F_N = \frac{1}{N} \sum_{i=1}^N \frac{f(X_i)}{p(X_i)}$

As direcções de bounce são amostradas de acordo com a PDF da BRDF para minimizar variância.

### Cosine-Weighted Hemisphere Sampling
Direcções amostradas com probabilidade proporcional a $\cos\theta$ usando a fórmula:
- $\phi = 2\pi r_1$, $r = \sqrt{r_2}$, $z = \sqrt{1 - r_2}$

### MIS — Multiple Importance Sampling
Combinação de $n$ estratégias de amostragem com pesos:
- **Balanceada (passos 3–5):** $w_s = p_s / \sum_i p_i$
- **Potência (passos 6–7):** $w_s = p_s^2 / \sum_i p_i^2$

### Roleta Russa
Terminação probabilística *unbiased* a partir do bounce 2:
$$q = \text{clamp}(1 - \text{lum}(\beta),\, 0.05,\, 0.80)$$
Se o caminho sobrevive, o throughput é compensado: $\beta \leftarrow \beta / (1 - q)$.

### Tone Mapping
- **Passos 1–6:** Gamma $\sqrt{x}$ (aproximação gamma 2.0)
- **Passo 7:** ACES Filmic + Gamma 2.2
  - $\text{ACES}(x) = \frac{x(2.51x + 0.03)}{x(2.43x + 0.59) + 0.14}$

### Controlo de Fireflies
1. **Clamp G no BDPT:** $G = \min(G,\, 25.0)$ + distância mínima $d > 10^{-2}$ cm
2. **Clamp por amostra (passos 5+):** se $\text{lum}(s) > 20.0$, escala $s \leftarrow s \cdot 20.0 / \text{lum}(s)$
3. **Clamp de throughput:** 8.0 (passo 7) / 12.0 (passo 6) por canal

---

## 7. Materiais por Passo (Resumo)

| Passo | Esfera | Caixa |
|-------|--------|-------|
| 1–4   | MicrofacetMaterial (metallic=0.10, rough=0.34) | MicrofacetMaterial (metallic=0.0, rough=0.30) |
| 5     | Material espelhado (reflectivity=0.95, shininess=100) | MicrofacetMaterial (plastic) |
| 6     | MicrofacetMaterial (metallic=0.10, rough=0.34) | MicrofacetMaterial (plastic) |
| 7     | Ouro metálico (metallic=0.95, rough=0.05) | Plástico branco suave (metallic=0.0, rough=0.15) |

---

## 8. SPP e d_max Padrão por Passo

| Passo | SPP padrão | d_max padrão | Filtro padrão |
|-------|-----------|-------------|---------------|
| 1     | 25 (args) | 4 (args)    | Não           |
| 2     | 25 (args) | 4 (args)    | Não           |
| 3     | 25 (args) | 4 (args)    | Não           |
| 3.1   | 64        | 4           | Não           |
| 4     | 25 (args) | 4 (args)    | Não           |
| 5     | 25 (args) | 4 (args)    | Não           |
| 5.1   | 128       | 6           | Sim (Gaussian)|
| 6     | 96 (mín.) | 6 (mín.)    | Não           |
| 7     | 128       | 4 (args)    | Não (MedianFilter com --use_filter) |
