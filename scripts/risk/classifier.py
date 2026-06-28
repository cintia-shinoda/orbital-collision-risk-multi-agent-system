"""Classificador de risco de colisão treinado no dataset CDM da ESA.

Treina um LightGBM para classificar o tier de risco (LOW/MED/HIGH) a partir
de features que a NOSSA camada orbital também produz, garantindo que o modelo
seja aplicável às conjunções reais do Cosmos 2251.

Rótulo: risk = log10(prob. de colisão) do último CDM de cada evento.
Limiar acionável da ESA: risk >= -6 (HIGH).

Reuso do TCC: mesma metodologia de gradient boosting, novo alvo.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_ESA_CSV = _DATA_DIR / "esa/dataset/kelvins_competition_data/train_data.csv"
_MODEL_PATH = _DATA_DIR / "risk_model.txt"

# Features deployáveis: as únicas que a nossa pipeline também calcula
# (em km e km/s, após converter as unidades da ESA, que vêm em m e m/s).
_FEATURES = ["miss_distance_km", "relative_speed_km_s", "time_to_tca"]


def load_training_data() -> pl.DataFrame:
    """Carrega o CSV da ESA e extrai o ÚLTIMO CDM de cada evento.

    O CDM final é o de menor time_to_tca (mais próximo da aproximação).
    Converte miss_distance/relative_speed de m e m/s para km e km/s.
    """
    df = pl.read_csv(_ESA_CSV, infer_schema_length=10000)

    # Último CDM de cada evento = menor time_to_tca
    final = (
        df.sort("time_to_tca")
        .group_by("event_id", maintain_order=True)
        .first()
    )

    return final.select(
        (pl.col("miss_distance") / 1000.0).alias("miss_distance_km"),
        (pl.col("relative_speed") / 1000.0).alias("relative_speed_km_s"),
        pl.col("time_to_tca"),
        pl.col("risk"),
    ).with_columns(
        # Acionável (1) vs não-acionável (0). Limiar operacional: risk >= -15.
        pl.when(pl.col("risk") >= -15.0).then(1).otherwise(0).alias("tier")
    )


def train() -> None:
    """Treina o classificador e reporta métricas honestas."""
    data = load_training_data()

    print("Distribuição dos tiers (0=LOW, 1=MED, 2=HIGH):")
    print(data.group_by("tier").len().sort("tier"))

    X = data.select(_FEATURES).to_numpy()
    y = data["tier"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nRelatório (test set):")
    print(classification_report(y_test, y_pred, target_names=["NÃO-ACIONÁVEL", "ACIONÁVEL"]))
    print("Matriz de confusão (linha=real, coluna=previsto):")
    print(confusion_matrix(y_test, y_pred))

    model.booster_.save_model(_MODEL_PATH)
    print(f"\nModelo salvo em: {_MODEL_PATH}")


def classify_conjunctions(conjunctions: pl.DataFrame) -> pl.DataFrame:
    """Aplica o modelo treinado às nossas conjunções (saída de conjunctions.py).

    Espera as colunas min_distance_km e rel_speed_km_s. Devolve risk_tier
    e a probabilidade da classe HIGH (p_high), útil para ranquear.
    """
    booster = lgb.Booster(model_file=str(_MODEL_PATH))
    X = conjunctions.select(
        pl.col("min_distance_km").alias("miss_distance_km"),
        pl.col("rel_speed_km_s").alias("relative_speed_km_s"),
        (pl.col("t_ca_min") / 1440.0).alias("time_to_tca"),  # min -> dias
    ).to_numpy()

    proba = booster.predict(X)  # vetor (N,) com prob. da classe ACIONÁVEL
    return conjunctions.with_columns(
        pl.Series("actionable_proba", np.round(proba, 4)),
        pl.Series("risk_flag", np.where(proba >= 0.5, "ACIONÁVEL", "monitorar")),
    )


if __name__ == "__main__":
    train()