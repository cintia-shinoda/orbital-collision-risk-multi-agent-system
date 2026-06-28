"""Inspeção do dataset CDM da ESA — checkpoint de colunas e distribuição."""
import polars as pl

_PATH = "data/esa/dataset/kelvins_competition_data/train_data.csv"

df = pl.read_csv(_PATH)
print("Linhas, colunas:", df.shape)
print("Eventos únicos:", df["event_id"].n_unique())

alvos = ["event_id", "time_to_tca", "risk", "relative_speed", "miss_distance"]
print("Presentes:", [c for c in alvos if c in df.columns])
print(df.select([c for c in alvos if c in df.columns]).head(8))

# Distribuição do risk na ÚLTIMA linha de cada evento (o alvo real de treino)
final = df.group_by("event_id").last()
print("\nDistribuição do risk final:")
print(final.select("risk").describe())