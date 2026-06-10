"""
Kensei Challenge — Inteligência de Mercado com IA
Análise automatizada do mercado Airbnb do Rio de Janeiro
usando dados públicos do InsideAirbnb + Claude AI (Anthropic)

Uso:
    pip install -r requirements.txt
    cp .env.example .env  # preencha ANTHROPIC_API_KEY
    python kensei_airbnb.py
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv
import anthropic

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
load_dotenv()

# ─── Configuração ─────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
CHARTS_DIR = Path("charts")
REPORTS_DIR = Path("reports")

for d in [DATA_DIR, CHARTS_DIR, REPORTS_DIR]:
    d.mkdir(exist_ok=True)

BASE_URL = "http://data.insideairbnb.com"
CITY_PATH = "brazil/rj/rio-de-janeiro"
CLAUDE_MODEL = "claude-opus-4-8"

AIRBNB_COLORS = ["#FF5A5F", "#FC642D", "#00A699", "#484848", "#767676"]


# ─── 1. Coleta de Dados ───────────────────────────────────────────────────────

def discover_latest_date() -> str:
    """Descobre a data mais recente dos dados do Rio de Janeiro no InsideAirbnb."""
    print("🔍 Descobrindo data mais recente dos dados...")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research/academic)"}

    try:
        r = requests.get("https://insideairbnb.com/get-the-data/", headers=headers, timeout=30)
        soup = BeautifulSoup(r.text, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True) if "rio-de-janeiro" in a["href"]]

        dates = []
        for link in links:
            for part in link.split("/"):
                try:
                    datetime.strptime(part, "%Y-%m-%d")
                    dates.append(part)
                except ValueError:
                    continue

        if dates:
            latest = sorted(set(dates))[-1]
            print(f"   ✅ Data encontrada: {latest}")
            return latest
    except Exception as e:
        print(f"   ⚠️  Scraping falhou ({e}), usando fallback.")

    # Fallback: datas conhecidas em ordem decrescente
    for fallback in ["2025-03-17", "2024-12-22", "2024-09-29", "2024-06-28"]:
        url = f"{BASE_URL}/{CITY_PATH}/{fallback}/data/listings.csv.gz"
        try:
            r = requests.head(url, headers=headers, timeout=10)
            if r.status_code == 200:
                print(f"   ✅ Fallback disponível: {fallback}")
                return fallback
        except Exception:
            continue

    print("   ⚠️  Usando data padrão: 2024-09-29")
    return "2024-09-29"


def _download(url: str, dest: Path, label: str) -> bool:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research/academic)"}
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=120)
        if r.status_code != 200:
            return False
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=f"   📥 {label}", total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as pb:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                pb.update(len(chunk))
        return True
    except Exception as e:
        print(f"   ❌ {label}: {e}")
        return False


def load_listings(date: str) -> pd.DataFrame:
    cache = DATA_DIR / f"listings_{date}.csv.gz"
    if not cache.exists():
        url = f"{BASE_URL}/{CITY_PATH}/{date}/data/listings.csv.gz"
        if not _download(url, cache, "listings.csv.gz"):
            # tenta versão sumária
            cache = DATA_DIR / f"listings_summary_{date}.csv"
            url = f"{BASE_URL}/{CITY_PATH}/{date}/visualisations/listings.csv"
            _download(url, cache, "listings.csv (summary)")

    print("   📂 Carregando listings...", end="")
    try:
        df = pd.read_csv(cache, low_memory=False)
        print(f" {len(df):,} registros.")
        return df
    except Exception as e:
        print(f"\n   ❌ {e}")
        return pd.DataFrame()


def load_reviews(date: str) -> pd.DataFrame:
    cache = DATA_DIR / f"reviews_{date}.csv"
    if not cache.exists():
        url = f"{BASE_URL}/{CITY_PATH}/{date}/visualisations/reviews.csv"
        _download(url, cache, "reviews.csv")
    if cache.exists():
        print("   📂 Carregando reviews...", end="")
        df = pd.read_csv(cache)
        print(f" {len(df):,} registros.")
        return df
    return pd.DataFrame()


# ─── 2. Limpeza e Enriquecimento ──────────────────────────────────────────────

COLS_KEEP = [
    "id", "name", "neighbourhood_cleansed", "neighbourhood_group_cleansed",
    "latitude", "longitude", "room_type", "accommodates", "bedrooms",
    "bathrooms_text", "price", "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "review_scores_rating",
    "review_scores_cleanliness", "review_scores_location", "review_scores_value",
    "calculated_host_listings_count", "host_is_superhost", "instant_bookable",
    "reviews_per_month",
]


def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🧹 Limpando dados...")
    cols = [c for c in COLS_KEEP if c in df.columns]
    df = df[cols].copy()

    # Preço
    if "price" in df.columns:
        df["price"] = (
            df["price"].astype(str)
            .str.replace(r"[\$,\s]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
        )
        q01, q99 = df["price"].quantile(0.01), df["price"].quantile(0.99)
        before = len(df)
        df = df.query("@q01 <= price <= @q99 and price > 0")
        print(f"   Outliers removidos: {before - len(df):,} ({(before-len(df))/before*100:.1f}%)")

    # Banheiros
    if "bathrooms_text" in df.columns:
        df["bathrooms"] = df["bathrooms_text"].str.extract(r"(\d+\.?\d*)")[0].astype(float)

    # Booleanos
    for col in ["host_is_superhost", "instant_bookable"]:
        if col in df.columns:
            df[col] = df[col].map({"t": True, "f": False})

    # Métricas derivadas
    if "availability_365" in df.columns:
        df["est_occupancy"] = (1 - df["availability_365"] / 365).clip(0, 1)

    if "price" in df.columns and "est_occupancy" in df.columns:
        df["est_monthly_revenue"] = df["price"] * df["est_occupancy"] * 30

    print(f"   ✅ Dataset limpo: {len(df):,} listings | {df.shape[1]} colunas")
    return df


# ─── 3. Análises ─────────────────────────────────────────────────────────────

class Analyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results: dict = {}

    def run(self) -> dict:
        print("\n📊 Executando análises...")
        self.results["overview"] = self._overview()
        self.results["price_neighborhood"] = self._price_by_neighborhood()
        self.results["price_room"] = self._price_by_room_type()
        self.results["availability"] = self._availability()
        self.results["reviews"] = self._reviews()
        self.results["hosts"] = self._hosts()
        self.results["saturation"] = self._saturation()
        self.results["neighborhood_scores"] = self._neighborhood_scores()
        return self.results

    # ── sub-análises ──────────────────────────────────────────────────────────

    def _overview(self) -> dict:
        df = self.df
        d: dict = {
            "total_listings": int(len(df)),
            "median_price": float(df["price"].median()),
            "mean_price": float(df["price"].mean()),
            "p25_price": float(df["price"].quantile(0.25)),
            "p75_price": float(df["price"].quantile(0.75)),
        }
        if "neighbourhood_cleansed" in df.columns:
            d["total_bairros"] = int(df["neighbourhood_cleansed"].nunique())
        if "room_type" in df.columns:
            d["room_types"] = df["room_type"].value_counts().to_dict()
        if "host_is_superhost" in df.columns:
            d["superhost_pct"] = float(df["host_is_superhost"].mean() * 100)
        if "review_scores_rating" in df.columns:
            d["avg_rating"] = float(df["review_scores_rating"].mean())
        if "est_occupancy" in df.columns:
            d["avg_occupancy_pct"] = float(df["est_occupancy"].mean() * 100)
        if "est_monthly_revenue" in df.columns:
            d["median_monthly_revenue"] = float(df["est_monthly_revenue"].median())
        print(
            f"   📍 {d['total_listings']:,} listings | "
            f"R${d['median_price']:.0f}/noite | "
            f"{d.get('total_bairros','?')} bairros | "
            f"{d.get('avg_occupancy_pct', 0):.1f}% ocupação"
        )
        return d

    def _price_by_neighborhood(self, n: int = 25) -> pd.DataFrame:
        if "neighbourhood_cleansed" not in self.df.columns:
            return pd.DataFrame()
        return (
            self.df.groupby("neighbourhood_cleansed")["price"]
            .agg(mediana="median", media="mean", total="count", dp="std")
            .query("total >= 10")
            .sort_values("mediana", ascending=False)
            .head(n)
        )

    def _price_by_room_type(self) -> pd.DataFrame:
        if "room_type" not in self.df.columns:
            return pd.DataFrame()
        return (
            self.df.groupby("room_type")["price"]
            .agg(mediana="median", media="mean", total="count")
            .sort_values("mediana", ascending=False)
        )

    def _availability(self) -> dict:
        df = self.df
        d: dict = {}
        if "est_occupancy" in df.columns:
            d["avg_occupancy_pct"] = float(df["est_occupancy"].mean() * 100)

        if "neighbourhood_cleansed" in df.columns:
            if "est_occupancy" in df.columns:
                d["occ_by_bairro"] = (
                    df.groupby("neighbourhood_cleansed")["est_occupancy"]
                    .mean().sort_values(ascending=False).head(20) * 100
                ).to_dict()
            if "est_monthly_revenue" in df.columns:
                d["rev_by_bairro"] = (
                    df.groupby("neighbourhood_cleansed")["est_monthly_revenue"]
                    .median().sort_values(ascending=False).head(20)
                ).to_dict()
        return d

    def _reviews(self) -> dict:
        df = self.df
        d: dict = {}
        if "review_scores_rating" in df.columns:
            d["avg_rating"] = float(df["review_scores_rating"].mean())
            d["pct_gt_4_5"] = float((df["review_scores_rating"] >= 4.5).mean() * 100)
        if "number_of_reviews" in df.columns:
            d["avg_reviews"] = float(df["number_of_reviews"].mean())
            d["median_reviews"] = float(df["number_of_reviews"].median())
            if "price" in df.columns:
                corr = df[["price", "number_of_reviews"]].corr().iloc[0, 1]
                d["price_reviews_corr"] = float(corr)
        return d

    def _hosts(self) -> dict:
        df = self.df
        d: dict = {}
        if "calculated_host_listings_count" in df.columns:
            d["avg_listings_per_host"] = float(df["calculated_host_listings_count"].mean())
            d["pct_multi_listing"] = float((df["calculated_host_listings_count"] > 1).mean() * 100)
            d["pct_commercial"] = float((df["calculated_host_listings_count"] >= 5).mean() * 100)
        if "host_is_superhost" in df.columns:
            d["superhost_pct"] = float(df["host_is_superhost"].mean() * 100)
            if "price" in df.columns:
                sh = df[df["host_is_superhost"] == True]["price"].median()
                nsh = df[df["host_is_superhost"] == False]["price"].median()
                d["superhost_premium_pct"] = float((sh - nsh) / nsh * 100) if nsh else 0
        return d

    def _saturation(self) -> pd.DataFrame:
        if "neighbourhood_cleansed" not in self.df.columns:
            return pd.DataFrame()
        df = self.df
        agg = df.groupby("neighbourhood_cleansed").agg(
            total_listings=("id", "count"),
            median_price=("price", "median"),
            avg_occupancy=("est_occupancy", "mean"),
            superhost_pct=("host_is_superhost", "mean"),
        ).query("total_listings >= 20")

        if "avg_occupancy" in agg.columns:
            agg["saturation_score"] = (
                agg["total_listings"].rank(pct=True) * 0.6
                + (1 - agg["avg_occupancy"].rank(pct=True)) * 0.4
            )
            agg = agg.sort_values("saturation_score", ascending=False)
        return agg

    def _neighborhood_scores(self) -> pd.DataFrame:
        if "neighbourhood_cleansed" not in self.df.columns:
            return pd.DataFrame()
        df = self.df
        nbhd = df.groupby("neighbourhood_cleansed").agg(
            total=("id", "count"),
            median_price=("price", "median"),
            avg_rating=("review_scores_rating", "mean"),
            avg_occupancy=("est_occupancy", "mean"),
            avg_revenue=("est_monthly_revenue", "median"),
        ).query("total >= 20")

        nbhd["guest_score"] = (
            nbhd["avg_rating"].rank(pct=True) * 0.5
            + (1 - nbhd["median_price"].rank(pct=True)) * 0.5
        )
        nbhd["host_score"] = (
            nbhd["avg_occupancy"].rank(pct=True) * 0.5
            + nbhd["avg_revenue"].rank(pct=True) * 0.5
        )
        return nbhd.sort_values("host_score", ascending=False)


# ─── 4. Visualizações ─────────────────────────────────────────────────────────

class Visualizer:
    def __init__(self, df: pd.DataFrame, results: dict):
        self.df = df
        self.r = results
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette(AIRBNB_COLORS)

    def run(self) -> list[str]:
        print("\n📈 Gerando visualizações...")
        paths = []
        paths.append(self._price_distribution())
        paths.append(self._price_by_neighborhood())
        paths.append(self._room_types())
        paths.append(self._occupancy_by_neighborhood())
        paths.append(self._revenue_by_neighborhood())
        paths.append(self._price_vs_reviews())
        paths.append(self._host_profile())
        paths = [p for p in paths if p]
        print(f"   ✅ {len(paths)} gráficos em {CHARTS_DIR}/")
        return paths

    def _save(self, name: str) -> str:
        p = CHARTS_DIR / f"{name}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        return str(p)

    def _price_distribution(self) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Distribuição de Preços — Airbnb Rio de Janeiro", fontsize=14, fontweight="bold")

        prices = self.df["price"].dropna()

        ax = axes[0]
        ax.hist(prices, bins=60, color="#FF5A5F", alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(prices.median(), color="#484848", ls="--", lw=2, label=f"Mediana R${prices.median():.0f}")
        ax.axvline(prices.mean(), color="#00A699", ls="--", lw=2, label=f"Média R${prices.mean():.0f}")
        ax.set_xlabel("Preço (R$/noite)")
        ax.set_ylabel("Frequência")
        ax.set_title("Histograma de Preços")
        ax.legend()

        ax = axes[1]
        if "room_type" in self.df.columns:
            room_prices = [
                self.df.loc[self.df["room_type"] == rt, "price"].dropna()
                for rt in self.df["room_type"].unique()
            ]
            bp = ax.boxplot(
                room_prices, labels=self.df["room_type"].unique(),
                patch_artist=True, medianprops=dict(color="white", linewidth=2)
            )
            for patch, color in zip(bp["boxes"], AIRBNB_COLORS):
                patch.set_facecolor(color)
            ax.set_xlabel("Tipo de Imóvel")
            ax.set_ylabel("Preço (R$/noite)")
            ax.set_title("Boxplot por Tipo")
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

        return self._save("01_price_distribution")

    def _price_by_neighborhood(self) -> str:
        pn = self.r.get("price_neighborhood")
        if pn is None or pn.empty:
            return ""
        fig, ax = plt.subplots(figsize=(12, 9))
        data = pn.head(20)
        bars = ax.barh(range(len(data)), data["mediana"], color="#FF5A5F", alpha=0.85)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Preço Mediano (R$/noite)")
        ax.set_title("Top 20 Bairros por Preço Mediano", fontsize=13, fontweight="bold")
        for bar, (_, row) in zip(bars, data.iterrows()):
            ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
                    f"R${bar.get_width():.0f}  ({int(row['total'])})", va="center", fontsize=8)
        return self._save("02_price_by_neighborhood")

    def _room_types(self) -> str:
        if "room_type" not in self.df.columns:
            return ""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Distribuição por Tipo de Imóvel", fontsize=13, fontweight="bold")

        counts = self.df["room_type"].value_counts()
        axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                    colors=AIRBNB_COLORS[:len(counts)], startangle=90)
        axes[0].set_title("% de Listings")

        pr = self.r.get("price_room")
        if pr is not None and not pr.empty:
            axes[1].bar(pr.index, pr["mediana"], color=AIRBNB_COLORS[:len(pr)])
            axes[1].set_xlabel("Tipo")
            axes[1].set_ylabel("Preço Mediano (R$/noite)")
            axes[1].set_title("Preço por Tipo")
            plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right")

        return self._save("03_room_types")

    def _occupancy_by_neighborhood(self) -> str:
        occ = self.r.get("availability", {}).get("occ_by_bairro", {})
        if not occ:
            return ""
        fig, ax = plt.subplots(figsize=(12, 8))
        nbhds = list(occ.keys())[:15]
        vals = [occ[n] for n in nbhds]
        colors = ["#FF5A5F" if v > 70 else "#FC642D" if v > 50 else "#00A699" for v in vals]
        bars = ax.barh(range(len(nbhds)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(nbhds)))
        ax.set_yticklabels(nbhds, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(50, color="#484848", ls="--", lw=1.5, label="50% referência")
        ax.set_xlabel("Taxa de Ocupação Estimada (%)")
        ax.set_title("Top 15 Bairros por Ocupação", fontsize=13, fontweight="bold")
        ax.legend()
        for bar in bars:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.1f}%", va="center", fontsize=8)
        return self._save("04_occupancy_by_neighborhood")

    def _revenue_by_neighborhood(self) -> str:
        rev = self.r.get("availability", {}).get("rev_by_bairro", {})
        if not rev:
            return ""
        fig, ax = plt.subplots(figsize=(12, 8))
        nbhds = list(rev.keys())[:15]
        vals = [rev[n] for n in nbhds]
        bars = ax.barh(range(len(nbhds)), vals, color="#FC642D", alpha=0.85)
        ax.set_yticks(range(len(nbhds)))
        ax.set_yticklabels(nbhds, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Receita Mensal Estimada Mediana (R$)")
        ax.set_title("Top 15 Bairros por Receita Estimada", fontsize=13, fontweight="bold")
        for bar in bars:
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                    f"R${bar.get_width():,.0f}", va="center", fontsize=8)
        return self._save("05_revenue_by_neighborhood")

    def _price_vs_reviews(self) -> str:
        needed = {"price", "number_of_reviews"}
        if not needed.issubset(self.df.columns):
            return ""
        fig, ax = plt.subplots(figsize=(10, 6))
        sample = self.df[list(needed | {"review_scores_rating"})].dropna().sample(
            min(2500, len(self.df)), random_state=42
        )
        sc = ax.scatter(
            sample["number_of_reviews"], sample["price"],
            c=sample.get("review_scores_rating", pd.Series([3.0] * len(sample))),
            cmap="RdYlGn", alpha=0.45, s=18, vmin=3, vmax=5
        )
        plt.colorbar(sc, ax=ax, label="Nota Média")
        ax.set_xlabel("Número de Reviews")
        ax.set_ylabel("Preço (R$/noite)")
        ax.set_title("Preço × Número de Reviews", fontsize=13, fontweight="bold")
        z = np.polyfit(sample["number_of_reviews"], sample["price"], 1)
        x_line = np.linspace(0, sample["number_of_reviews"].quantile(0.95), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), "r--", alpha=0.7, lw=1.5, label="Tendência")
        ax.legend()
        return self._save("06_price_vs_reviews")

    def _host_profile(self) -> str:
        if "calculated_host_listings_count" not in self.df.columns:
            return ""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Perfil dos Anfitriões", fontsize=13, fontweight="bold")

        # Distribuição de listings por host
        counts = self.df["calculated_host_listings_count"].clip(upper=20)
        axes[0].hist(counts, bins=20, color="#00A699", alpha=0.85, edgecolor="white")
        axes[0].set_xlabel("Listings por Anfitrião (cap 20)")
        axes[0].set_ylabel("Frequência")
        axes[0].set_title("Distribuição de Listings por Host")

        # Superhost x non-superhost price
        if "host_is_superhost" in self.df.columns and "price" in self.df.columns:
            sh_data = {
                "Superhost": self.df[self.df["host_is_superhost"] == True]["price"].dropna(),
                "Regular": self.df[self.df["host_is_superhost"] == False]["price"].dropna(),
            }
            axes[1].boxplot(sh_data.values(), labels=sh_data.keys(), patch_artist=True,
                            medianprops=dict(color="white", linewidth=2))
            axes[1].set_ylabel("Preço (R$/noite)")
            axes[1].set_title("Preço: Superhost vs Regular")

        return self._save("07_host_profile")


# ─── 5. Insights com Claude AI ────────────────────────────────────────────────

class AIInsights:
    def __init__(self, results: dict):
        self.r = results
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _summarize(self) -> str:
        ov = self.r.get("overview", {})
        av = self.r.get("availability", {})
        rv = self.r.get("reviews", {})
        ht = self.r.get("hosts", {})
        pn = self.r.get("price_neighborhood")
        pr = self.r.get("price_room")
        sat = self.r.get("saturation")

        top_caros = pn.head(5).to_string() if pn is not None and not pn.empty else "N/A"
        mais_baratos = pn.tail(5).to_string() if pn is not None and not pn.empty else "N/A"
        room_str = pr.to_string() if pr is not None and not pr.empty else "N/A"
        sat_str = sat.head(5).to_string() if sat is not None and not sat.empty else "N/A"

        occ_str = "\n".join(f"  {k}: {v:.1f}%" for k, v in list(av.get("occ_by_bairro", {}).items())[:10])
        rev_str = "\n".join(f"  {k}: R${v:,.0f}/mês" for k, v in list(av.get("rev_by_bairro", {}).items())[:10])

        return f"""
=== DATASET AIRBNB RIO DE JANEIRO ===

VISÃO GERAL
- Total listings: {ov.get('total_listings', '?'):,}
- Preço mediano: R${ov.get('median_price', 0):,.0f}/noite
- Preço médio: R${ov.get('mean_price', 0):,.0f}/noite
- Faixa P25-P75: R${ov.get('p25_price', 0):,.0f} – R${ov.get('p75_price', 0):,.0f}
- Bairros únicos: {ov.get('total_bairros', '?')}
- % Superhosts: {ov.get('superhost_pct', 0):.1f}%
- Nota média geral: {ov.get('avg_rating', 0):.2f}/5.0
- Taxa de ocupação estimada média: {ov.get('avg_occupancy_pct', 0):.1f}%
- Receita mensal mediana estimada: R${ov.get('median_monthly_revenue', 0):,.0f}

TIPOS DE IMÓVEL (preço mediano)
{room_str}

TOP 5 BAIRROS MAIS CAROS
{top_caros}

5 BAIRROS MAIS ACESSÍVEIS (min. 10 listings)
{mais_baratos}

OCUPAÇÃO ESTIMADA POR BAIRRO (top 10)
{occ_str}

RECEITA MENSAL ESTIMADA POR BAIRRO (top 10)
{rev_str}

REVIEWS
- Nota média: {rv.get('avg_rating', 0):.2f}/5.0
- % listings nota ≥ 4.5: {rv.get('pct_gt_4_5', 0):.1f}%
- Média de reviews por listing: {rv.get('avg_reviews', 0):.1f}
- Correlação preço × reviews: {rv.get('price_reviews_corr', 0):.3f}

ANFITRIÕES
- % Superhosts: {ht.get('superhost_pct', 0):.1f}%
- Prêmio Superhost no preço: +{ht.get('superhost_premium_pct', 0):.1f}%
- % anfitriões multi-listing: {ht.get('pct_multi_listing', 0):.1f}%
- % anfitriões "comerciais" (5+ listings): {ht.get('pct_commercial', 0):.1f}%

BAIRROS MAIS SATURADOS (top 5)
{sat_str}
"""

    def _call(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1800,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def generate(self) -> dict:
        print("\n🤖 Gerando insights com Claude AI...")
        summary = self._summarize()

        specs = {
            "hospedes": {
                "title": "Insights para Hóspedes",
                "prompt": f"""Você é analista especialista em aluguel por temporada.
Com base nos dados reais do Airbnb Rio de Janeiro abaixo, gere insights CONCRETOS para HÓSPEDES.

{summary}

Responda em português brasileiro com estas seções:
## 1. Melhores Bairros por Custo-Benefício
## 2. Estratégia de Reserva para Economizar
## 3. O Que os Dados Revelam sobre Qualidade
## 4. Alertas: O Que Evitar

Use números reais dos dados. Seja direto e acionável.""",
            },
            "anfitrioes": {
                "title": "Insights para Anfitriões",
                "prompt": f"""Você é consultor especialista em rentabilidade de aluguel por temporada.
Com base nos dados reais do Airbnb Rio de Janeiro abaixo, gere insights CONCRETOS para ANFITRIÕES.

{summary}

Responda em português brasileiro com estas seções:
## 1. Precificação Estratégica (por bairro e tipo)
## 2. Como Maximizar Ocupação
## 3. Vale a Pena Ser Superhost? (dados mostram o prêmio)
## 4. Onde Investir: Melhores Bairros por ROI
## 5. Onde Evitar: Mercados Saturados

Use números reais dos dados. Seja direto e acionável.""",
            },
            "mercado": {
                "title": "Análise de Mercado",
                "prompt": f"""Você é analista de inteligência de mercado imobiliário.
Com base nos dados reais do Airbnb Rio de Janeiro abaixo, produza uma análise estratégica.

{summary}

Responda em português brasileiro com estas seções:
## 1. Estado Atual e Concentração do Mercado
## 2. Padrões e Tendências Identificados nos Dados
## 3. Oportunidades Sub-exploradas
## 4. Riscos: Saturação e Competição
## 5. Recomendações Estratégicas

Base-se exclusivamente nos dados fornecidos. Seja analítico.""",
            },
        }

        insights = {}
        for key, spec in specs.items():
            print(f"   🔮 {spec['title']}...", end="", flush=True)
            try:
                text = self._call(spec["prompt"])
                insights[key] = {"title": spec["title"], "content": text}
                print(" ✅")
            except Exception as e:
                print(f" ❌ ({e})")
                insights[key] = {
                    "title": spec["title"],
                    "content": f"_Erro ao gerar insights: {e}_",
                }
        return insights


# ─── 6. Relatório Markdown ────────────────────────────────────────────────────

def generate_report(results: dict, insights: dict, date: str) -> str:
    ov = results.get("overview", {})
    av = results.get("availability", {})
    rv = results.get("reviews", {})
    ht = results.get("hosts", {})
    pn = results.get("price_neighborhood")
    pr = results.get("price_room")
    sat = results.get("saturation")

    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    md = f"""# Relatório de Inteligência de Mercado — Airbnb Rio de Janeiro

**Fonte:** InsideAirbnb.com | **Data dos dados:** {date} | **Gerado em:** {now}
**Metodologia:** Coleta automatizada (Python) + Análise (pandas/numpy) + Insights (Claude AI)

---

## Visão Geral do Mercado

| Métrica | Valor |
|---|---|
| Total de Listings Ativos | {ov.get('total_listings', '?'):,} |
| Preço Mediano | R$ {ov.get('median_price', 0):,.0f} / noite |
| Preço Médio | R$ {ov.get('mean_price', 0):,.0f} / noite |
| Faixa Budget (P25) | R$ {ov.get('p25_price', 0):,.0f} / noite |
| Faixa Premium (P75) | R$ {ov.get('p75_price', 0):,.0f} / noite |
| Bairros Mapeados | {ov.get('total_bairros', '?')} |
| % Superhosts | {ov.get('superhost_pct', 0):.1f}% |
| Nota Média Geral | {ov.get('avg_rating', 0):.2f} / 5.0 |
| Taxa de Ocupação Estimada | {ov.get('avg_occupancy_pct', 0):.1f}% |
| Receita Mensal Mediana Estimada | R$ {ov.get('median_monthly_revenue', 0):,.0f} |

"""

    if pr is not None and not pr.empty:
        md += "### Distribuição por Tipo de Imóvel\n\n"
        md += "| Tipo | Listings | Preço Mediano | Preço Médio |\n|---|---|---|---|\n"
        for t, row in pr.iterrows():
            md += f"| {t} | {int(row['total']):,} | R$ {row['mediana']:,.0f} | R$ {row['media']:,.0f} |\n"
        md += "\n"

    md += "---\n\n## Análise por Bairros\n\n"

    if pn is not None and not pn.empty:
        md += "### Top 20 por Preço Mediano\n\n"
        md += "| # | Bairro | Listings | Preço Mediano | Preço Médio |\n|---|---|---|---|---|\n"
        for i, (bairro, row) in enumerate(pn.head(20).iterrows(), 1):
            md += f"| {i} | {bairro} | {int(row['total']):,} | R$ {row['mediana']:,.0f} | R$ {row['media']:,.0f} |\n"
        md += "\n"

    occ_bairro = av.get("occ_by_bairro", {})
    if occ_bairro:
        md += "### Top 15 por Taxa de Ocupação Estimada\n\n"
        md += "| # | Bairro | Ocupação |\n|---|---|---|\n"
        for i, (bairro, occ) in enumerate(list(occ_bairro.items())[:15], 1):
            bar = "█" * int(occ / 10)
            md += f"| {i} | {bairro} | {bar} {occ:.1f}% |\n"
        md += "\n"

    rev_bairro = av.get("rev_by_bairro", {})
    if rev_bairro:
        md += "### Top 15 por Receita Mensal Estimada\n\n"
        md += "| # | Bairro | Receita Estimada/Mês |\n|---|---|---|\n"
        for i, (bairro, rev) in enumerate(list(rev_bairro.items())[:15], 1):
            md += f"| {i} | {bairro} | R$ {rev:,.0f} |\n"
        md += "\n"

    md += "---\n\n## Reviews e Qualidade\n\n"
    md += f"""| Métrica | Valor |
|---|---|
| Nota Média | {rv.get('avg_rating', 0):.2f} / 5.0 |
| % Listings com Nota ≥ 4.5 | {rv.get('pct_gt_4_5', 0):.1f}% |
| Média de Reviews por Listing | {rv.get('avg_reviews', 0):.1f} |
| Correlação Preço × Reviews | {rv.get('price_reviews_corr', 0):+.3f} |

"""

    md += "---\n\n## Perfil dos Anfitriões\n\n"
    md += f"""| Métrica | Valor |
|---|---|
| % Superhosts | {ht.get('superhost_pct', 0):.1f}% |
| Prêmio de Preço Superhost | +{ht.get('superhost_premium_pct', 0):.1f}% |
| % Anfitriões Multi-listing | {ht.get('pct_multi_listing', 0):.1f}% |
| % Anfitriões Comerciais (5+) | {ht.get('pct_commercial', 0):.1f}% |

"""

    if sat is not None and not sat.empty:
        md += "---\n\n## Saturação de Mercado\n\n"
        md += "| Bairro | Listings | Preço Med. | Ocupação | Score |\n|---|---|---|---|---|\n"
        for bairro, row in sat.head(10).iterrows():
            sc = row.get("saturation_score", 0)
            label = "🔴 Alto" if sc > 0.7 else "🟡 Médio" if sc > 0.4 else "🟢 Baixo"
            md += (f"| {bairro} | {int(row['total_listings']):,} | "
                   f"R$ {row['median_price']:,.0f} | {row['avg_occupancy']*100:.1f}% | {label} |\n")
        md += "\n"

    md += "---\n\n## Insights Gerados por IA (Claude)\n\n"
    icons = {"hospedes": "🧳", "anfitrioes": "🏠", "mercado": "📈"}
    for key in ["hospedes", "anfitrioes", "mercado"]:
        ins = insights.get(key, {})
        icon = icons.get(key, "💡")
        md += f"### {icon} {ins.get('title', key)}\n\n{ins.get('content', '')}\n\n---\n\n"

    md += """## Metodologia

1. **Coleta:** Scraping de data + download automático do InsideAirbnb.com
2. **Limpeza:** Remoção de outliers P1–P99 em preço, conversão de tipos, normalização
3. **Métricas derivadas:**
   - Ocupação estimada = `1 − disponibilidade_365 / 365`
   - Receita estimada = `preço × ocupação × 30 dias`
   - Score de saturação = `0.6 × rank(listings) + 0.4 × rank(1 − ocupação)`
4. **Insights:** Claude AI (Anthropic) para síntese narrativa
5. **Visualizações:** matplotlib + seaborn (7 gráficos gerados)

> **Aviso:** Ocupação e receita são *estimativas* baseadas na disponibilidade declarada
> pelo anfitrião, não em reservas confirmadas. Use para comparação relativa entre bairros.

---
*Kensei Challenge — Inteligência de Mercado com IA | `kensei_airbnb.py`*
"""

    path = REPORTS_DIR / f"relatorio_airbnb_rio_{date}.md"
    path.write_text(md, encoding="utf-8")
    print(f"\n   ✅ Relatório salvo: {path}")
    return str(path)


# ─── 7. Orquestrador Principal ────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  KENSEI CHALLENGE — Inteligência de Mercado Airbnb RJ")
    print("=" * 62)

    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    if not has_api_key:
        print("\n⚠️  ANTHROPIC_API_KEY ausente — insights de IA serão pulados.")
        print("   Configure no arquivo .env para habilitar os insights.\n")

    # 1 — descobrir data
    date = discover_latest_date()

    # 2 — baixar/carregar dados
    print(f"\n📡 Baixando dados de {date}...")
    df_raw = load_listings(date)
    if df_raw.empty:
        print("❌ Falha ao carregar listings. Verifique a conexão.")
        sys.exit(1)

    # 3 — limpar
    df = clean_listings(df_raw)

    # 4 — analisar
    analyzer = Analyzer(df)
    results = analyzer.run()

    # 5 — visualizar
    viz = Visualizer(df, results)
    viz.run()

    # 6 — insights IA
    if has_api_key:
        ai = AIInsights(results)
        insights = ai.generate()
    else:
        placeholder = "_Configure `ANTHROPIC_API_KEY` no `.env` para gerar esta seção._"
        insights = {
            k: {"title": t, "content": placeholder}
            for k, t in [
                ("hospedes", "Insights para Hóspedes"),
                ("anfitrioes", "Insights para Anfitriões"),
                ("mercado", "Análise de Mercado"),
            ]
        }

    # 7 — relatório
    report_path = generate_report(results, insights, date)

    print("\n" + "=" * 62)
    print("  CONCLUÍDO!")
    print(f"  Relatório : {report_path}")
    print(f"  Gráficos  : {CHARTS_DIR}/")
    print(f"  Dados     : {DATA_DIR}/")
    print("=" * 62)


if __name__ == "__main__":
    main()
