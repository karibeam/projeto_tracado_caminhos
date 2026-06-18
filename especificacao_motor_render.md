# Especificação de Implementação: Motor de Traçado de Caminhos Incremental

Este documento detalha o planeamento para a implementação de um motor de renderização baseado em física (PBR) utilizando a técnica de **Traçado de Caminhos (Path Tracing)**, seguindo uma abordagem incremental com base nos slides fornecidos. O projeto será desenvolvido em **Python**, usando a biblioteca **pyGLM** para álgebra vetorial e transformações geométricas, e a biblioteca **Pillow** (PIL) para a gestão, pós-processamento e gravação do buffer de píxeis no diretório `output/`.

## 1. Parâmetros Globais e Configuração via Linha de Comando (CLI)
Todas as etapas e integradores do motor de traçado de caminhos devem ser acessíveis através da linha de comandos do terminal. O motor deve aceitar e respeitar obrigatoriamente as seguintes variáveis de entrada:

* **Passo a Executar (`step`):** Um número inteiro de 1 a 5 indicando qual algoritmo/passo rodar. Se este parâmetro não for especificado, o motor deve executar sequencialmente **todos** os passos (de 1 a 5), gerando as respetivas imagens.
* **Largura da imagem (`width`):** Resolução horizontal da imagem gerada em píxeis.
* **Altura da imagem (`height`):** Resolução vertical da imagem gerada em píxeis.
* **Quantidade de caminhos por píxel (`spp`):** Quantidade de amostras estatísticas (*Samples Per Pixel*) calculadas e acumuladas por píxel. *(Valor predefinido: `25`)*.
* **Profundidade máxima do caminho (`d_max`):** Quantidade limite de rebatimentos (segmentos) que um caminho pode sofrer. *(Valor predefinido: `4`)*.
  * **Regra de Fecho do Último Caminho:** Quando o ciclo de rebatimentos atingir o limite configurado (`i == d_max`), o motor deve forçar o encerramento do caminho amostrando explicitamente uma direção que aponte para a fonte de luz retangular do cenário para recolher a radiância final, fechando o estimador antes de descartar o caminho.
* **Filtro de Suavização (`use_filter`):** Uma flag booleana que, quando ativada, aplica um filtro de pós-processamento (ex: filtro Gaussiano da Pillow) na imagem final para mitigar o ruído visual excessivo gerado pela variância de Monte Carlo. *(Valor predefinido: `False`)*.

## 2. Sistema de Iluminação e Cenário (Cornell Box)
Para garantir a consistência física e o comportamento esperado em todos os passos, o motor irá utilizar como ambiente de testes a clássica **Cornell Box**. A cena deve conter rigidamente:
* **Estrutura Base:** Piso, teto, parede de fundo, parede esquerda (com albedo vermelho) e parede direita (com albedo verde).
* **Objetos Interiores:** Exatamente **uma bola (esfera) e uma caixa**, dispostas no interior do cenário para demonstrar a interação da luz com diferentes geometrias e materiais.
* **Iluminação - Fonte de Luz Retangular (Área):** Uma primitiva plana geométrica emissora posicionada no teto da Cornell Box. É responsável por gerar sombras suaves e servir de alvo para as amostragens diretas de luz.
* **Iluminação - Luz Ambiente:** Uma constante de radiação de fundo uniforme de intensidade subtil, presente em todo o cenário (computada caso o raio fuja por alguma fenda ou limite não coberto pela geometria).

---

## 3. Passos da Implementação Incremental

### Passo 1: Traçado de Caminhos Básico (Monte Carlo Path Tracing)
* **Objetivo:** Estabelecer a infraestrutura algorítmica base para o lançamento de raios de câmara, cálculo de interseções e a integração de Monte Carlo.
* **Cenário:** A cena da **Cornell Box** com a sua esfera e a sua caixa iluminadas pela fonte retangular no teto e luz ambiente.
* **Formulação Teórica:**
  * Estimador de Monte Carlo básico: $F_N = \frac{1}{N} \sum_{i=1}^N \frac{f(X_i)}{p(X_i)}$
  * BRDF Difusa (Lambertiana): $f(p, \omega_o, \omega_i) = \frac{\rho}{\pi}$, onde $\rho$ é o albedo.
* **Resultado Esperado:** Geração de `output/passo1_path_tracing_basico.png` evidenciando o ruído inicial a 25 SPP e a "sangria de cor" (color bleeding) das paredes.

### Passo 2: Profundidade Parametrizável e Roleta Russa
* **Objetivo:** Permitir a variação do parâmetro `d_max` e aplicar o algoritmo de Roleta Russa para encerrar caminhos de maneira probabilística e *unbiased*.
* **Mecânica:** * Se o caminho atingir a profundidade limite (`d_max`), executa o fecho forçado amostrando a luz.
  * Para rebatimentos intermediários, avalia-se a probabilidade de terminação $q$. Se o caminho sobreviver, a sua contribuição acumulada deve ser multiplicada pelo peso compensatório $\frac{1}{1-q}$.
* **Resultado Esperado:** Imagens guardadas em `output/` demonstrando o balanço de energia na iluminação global indireta graças à Roleta Russa.

### Passo 3: Múltiplas Amostragens por Importância (MIS)
* **Objetivo:** Reduzir drasticamente a variância do integrador distribuindo as amostras de forma inteligente no hemisfério.
* **Estratégia:** Combinar duas estratégias: amostrar os raios de acordo com o lobo da BRDF e amostrar diretamente a superfície da luz retangular geométrica no teto (*Shadow Rays*).
* **Heurística Balanceada:** Diferente da heurística da potência, as amostras serão combinadas utilizando a Heurística Balanceada, onde os pesos são dados por:
  $$w_s(x) = \frac{n_s p_s(x)}{\sum_i n_i p_i(x)}$$
* **Resultado Esperado:** Eliminação de *fireflies* e redução significativa do ruído de amostragem direta, com imagem guardada em `output/passo3_mis.png`.

### Passo 4: Modelo de Microfacetas (Cook-Torrance)
* **Objetivo:** Substituir a BRDF difusa por materiais realistas baseados em microgeometrias.
* **BRDF de Microfaceta:** $f_r(v,l) = \frac{\rho_d}{\pi} + \frac{D(h) F(v,h) G(l,v)}{4 \langle n\cdot l\rangle \langle n\cdot v\rangle}$
  * *Distribuição GGX ($D$)*, *Fresnel de Schlick ($F$)*, e *Termo Geométrico de Smith/Schlick ($G$)*.
* **Resultado Esperado:** A esfera e a caixa exibindo características físicas contrastantes (ex: metal vs plástico), guardada em `output/passo4_microfacets.png`.

### Passo 5: Métodos Bidirecionais (BDPT)
* **Objetivo:** Adicionar robustez para recolher caminhos de luz complexos (cantos obstruídos ou cáusticas).
* **Funcionamento:** Traçado de sub-caminhos partindo da câmara ($p_i$) e da luz ($q_j$), conetando os vértices no final por meio de raios de teste de visibilidade.
* **Resultado Esperado:** Efeitos complexos de iluminação indireta guardados em `output/passo5_bdpt.png`.

---

## 4. Sugestão de Estrutura do Código (Python / CLI)

O ponto de entrada (`main.py`) deve utilizar `argparse` para lidar com os parâmetros de execução no terminal:

```python
import argparse
import glm
from PIL import Image, ImageFilter

# Imports hipotéticos dos integradores
# from integrators import run_passo_1, run_passo_2, run_passo_3, run_passo_4, run_passo_5

def parse_args():
    parser = argparse.ArgumentParser(description="Motor de Renderização Path Tracing (Cornell Box)")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5], default=None,
                        help="Passo específico a ser executado (1 a 5). Se não for passado, executa todos.")
    parser.add_argument("--width", type=int, default=800, help="Largura da imagem gerada")
    parser.add_argument("--height", type=int, default=600, help="Altura da imagem gerada")
    parser.add_argument("--spp", type=int, default=25, help="Quantidade de amostras por píxel")
    parser.add_argument("--d_max", type=int, default=4, help="Profundidade máxima do caminho")
    parser.add_argument("--use_filter", action="store_true", help="Ativa a flag de pós-processamento/suavização")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # ... Inicialização da cena (Cornell Box, Esfera, Caixa, Luzes) ...
    
    steps_to_run = [args.step] if args.step else [1, 2, 3, 4, 5]
    
    for current_step in steps_to_run:
        print(f"A executar o Passo {current_step}...")
        
        # Exemplo de lógica de roteamento:
        # if current_step == 1:
        #     image = run_passo_1(args.width, args.height, args.spp, args.d_max, scene)
        # elif current_step == 2:
        #     ...
        
        # Pós-processamento genérico
        # if args.use_filter:
        #     image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        
        # image.save(f"output/passo{current_step}_resultado.png")

if __name__ == "__main__":
    main()
```

## 5. Organização do Repositório
```text
/motor_path_tracing/
  ├── main.py             # Parser do terminal via CLI e gestão dos passos a serem rodados
  ├── scene.py            # Definição das primitivas e montagem da Cornell Box
  ├── camera.py           # Geração de raios baseada na resolução
  ├── materials.py        # Cálculos de BRDF (Lambert e Cook-Torrance com GGX)
  ├── integrators.py      # Lógica separada para cada um dos 5 passos (Monte Carlo, Roleta, MIS, Microfacetas, BDPT)
  └── output/             # Diretório obrigatório para guardar as imagens geradas
```
