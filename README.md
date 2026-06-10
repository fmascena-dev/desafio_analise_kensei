# Kensei Challenge — Inteligência de Mercado com IA

Análise automatizada do mercado Airbnb do Rio de Janeiro usando dados públicos do [InsideAirbnb](https://insideairbnb.com/rio-de-janeiro/) + Claude AI (Anthropic).

## Estrutura do Projeto

```text
Cyber_AI/
├── kensei_airbnb.py       # Script principal (rodar direto)
├── kensei_airbnb.ipynb    # Notebook Jupyter (exploração interativa)
├── requirements.txt       # Dependências Python
├── .env.example           # Template de variáveis de ambiente
├── .env                   # Suas chaves (não commitar — no .gitignore)
├── .gitignore
├── data/                  # Dados baixados automaticamente
├── charts/                # Gráficos gerados (7 PNGs)
└── reports/               # Relatório em Markdown
```

## Pré-requisitos

- Python 3.9+
- Conta na [Anthropic Console](https://console.anthropic.com/) para a API key

## Setup

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Configurar API key
copy .env.example .env
# Edite .env e coloque sua ANTHROPIC_API_KEY

# 3. Rodar
python kensei_airbnb.py
```

Ou, para exploração interativa:

```bash
jupyter notebook kensei_airbnb.ipynb
```

## O que o script faz

| Etapa | Detalhe |
| --- | --- |
| Coleta | Descobre a data mais recente e baixa `listings.csv.gz` do InsideAirbnb |
| Limpeza | Remove outliers de preço (P1–P99), normaliza tipos, cria métricas derivadas |
| Análise | Preço por bairro/tipo, ocupação estimada, receita, reviews, anfitriões, saturação |
| Visualizações | 7 gráficos PNG salvos em `charts/` |
| IA | Chama Claude para gerar insights narrativos para 3 públicos |
| Relatório | Exporta `reports/relatorio_airbnb_rio_<data>.md` |

## Métricas Derivadas

- **Ocupação estimada** = `1 − disponibilidade_365 / 365`
- **Receita mensal estimada** = `preço × ocupação × 30`
- **Score de saturação** = `0.6 × rank(listings) + 0.4 × rank(1 − ocupação)`
- **Score host (bairro)** = `0.5 × rank(ocupação) + 0.5 × rank(receita)`
- **Score guest (bairro)** = `0.5 × rank(nota) + 0.5 × rank(1 − preço)`

## Outputs

- `reports/relatorio_airbnb_rio_<data>.md` — relatório completo com tabelas + insights de IA
- `charts/01_price_distribution.png` — histograma e boxplot de preços
- `charts/02_price_by_neighborhood.png` — top 20 bairros por preço
- `charts/03_room_types.png` — distribuição por tipo de imóvel
- `charts/04_occupancy_by_neighborhood.png` — top 15 por ocupação
- `charts/05_revenue_by_neighborhood.png` — top 15 por receita estimada
- `charts/06_price_vs_reviews.png` — scatter preço × reviews
- `charts/07_host_profile.png` — perfil dos anfitriões

## Stack

- **Python** — pandas, numpy, matplotlib, seaborn, requests, beautifulsoup4
- **Claude AI (Anthropic)** — `claude-opus-4-8` para síntese narrativa
- **InsideAirbnb** — fonte de dados públicos abertos

> **Aviso:** Ocupação e receita são estimativas baseadas em disponibilidade declarada,
> não em reservas confirmadas. Use para comparação relativa entre bairros.
