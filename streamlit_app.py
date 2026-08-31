import streamlit as st
import nflreadpy as nfl
import pandas as pd

from sklearn.linear_model import LinearRegression



# Page configuration
st.set_page_config(
    page_title="Fantasy Football Rankings",
    page_icon="🏈",
    layout="wide"
)



# Load data
@st.cache_data
def load_data():
    return nfl.load_ff_opportunity(
        seasons=range(2018, 2026)
    ).to_pandas()


ff_opp = load_data()


# Model
FEATURES = [
    "total_fantasy_points_exp",
    "total_yards_gained_exp",
    "total_touchdown_exp",
    "total_first_down_exp"
]

TARGET = "total_fantasy_points"


@st.cache_resource
def train_model(df):

    data = df[
        FEATURES + [TARGET]
    ].dropna()

    X = data[FEATURES]
    y = data[TARGET]

    model = LinearRegression()

    model.fit(X, y)

    return model


model = train_model(ff_opp)


# Get Top Players
def get_top_players(df, position, model, n=15):

    players = (
        df[df["position"] == position]
        .groupby(
            ["player_id", "full_name", "position"]
        )[FEATURES]
        .mean()
        .reset_index()
    )

    if players.empty:
        return players

    # Model prediction
    players["Predicted PPR"] = model.predict(
        players[FEATURES]
    )

    # Expected PPR
    players["Expected PPR"] = (
        players["total_fantasy_points_exp"]
    )

    # Sort by model prediction
    players = players.sort_values(
        "Predicted PPR",
        ascending=False
    ).head(n)

    players["Rank"] = range(
        1,
        len(players) + 1
    )

    players = players.rename(
        columns={
            "full_name": "Player",
            "position": "Position"
        }
    )

    return players[
        [
            "Rank",
            "Player",
            "Position",
            "Expected PPR",
            "Predicted PPR"
        ]
    ]


# Title
st.title("🏈 Fantasy Football Rankings")
st.write(
    "Comparing expected fantasy points with model predictions."
)

# Sidebar
st.sidebar.header("Filters")

season = st.sidebar.selectbox(
    "Season",
    sorted(
        ff_opp["season"].unique(),
        reverse=True
    )
)

position = st.sidebar.selectbox(
    "Position",
    [
        "QB",
        "RB",
        "WR",
        "TE",
        "K",
        "LB",
        "DL",
        "DB",
        "P",
        "SPEC"
    ]
)


# Filter Season
current = ff_opp[
    ff_opp["season"] == season
].copy()


# Rankings
top_players = get_top_players(
    current,
    position,
    model
)


# Display
st.subheader(
    f"Top 15 {position}s — {season}"
)


if top_players.empty:

    st.warning(
        f"No players found for {position} in {season}."
    )

else:

    st.dataframe(
        top_players,
        hide_index=True,
        use_container_width=True
    )