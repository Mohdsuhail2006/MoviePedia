import os
from datetime import date, timedelta

import requests
import streamlit as st
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# ---------------------------------------------------
# STREAMLIT PAGE SETTINGS
# ---------------------------------------------------
st.set_page_config(
    page_title="MoviePedia",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 MoviePedia")
st.caption("AI-powered movie recommendation system using Groq, LangGraph and TMDB")


# ---------------------------------------------------
# LOAD API KEYS FROM STREAMLIT SECRETS
# ---------------------------------------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    tmdb_api_key = st.secrets["TMDB_API_KEY"]
except KeyError as exc:
    st.error(
        "API keys are missing. Create .streamlit/secrets.toml and add "
        "GROQ_API_KEY and TMDB_API_KEY."
    )
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["TMDB_API_KEY"] = tmdb_api_key


# ---------------------------------------------------
# TMDB HELPERS
# ---------------------------------------------------
TMDB_BASE_URL = "https://api.themoviedb.org/3"


def tmdb_request(endpoint, params=None):
    """Send a request to TMDB API."""
    try:
        api_key = os.environ["TMDB_API_KEY"]

        params = dict(params or {})
        params["api_key"] = api_key

        response = requests.get(
            f"{TMDB_BASE_URL}/{endpoint}",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"TMDB request failed: {str(e)}"}


def format_movies(results, limit=10):
    """Format TMDB movie results into readable text."""
    if not results:
        return "No movies found."

    output = []

    for movie in results[:limit]:
        title = movie.get("title", "Unknown")
        rating = movie.get("vote_average", 0)
        release_date = movie.get("release_date", "Unknown")
        overview = movie.get("overview", "No description available.")
        movie_id = movie.get("id", "Unknown")

        output.append(
            f"""Title: {title}
TMDB ID: {movie_id}
Rating: {rating}
Release Date: {release_date}
Overview: {overview}
"""
        )

    return "\n".join(output)


# ---------------------------------------------------
# TOOL 1: SEARCH MOVIES BY TITLE
# ---------------------------------------------------
@tool
def search_movies(movie_name: str) -> str:
    """Search for movies by title. Use this when the user gives a specific movie name."""
    data = tmdb_request(
        "search/movie",
        {
            "query": movie_name,
            "language": "en-US",
            "include_adult": False,
        },
    )

    if "error" in data:
        return data["error"]

    results = data.get("results", [])

    if not results:
        return f"No movies found for '{movie_name}'."

    return format_movies(results, limit=10)


# ---------------------------------------------------
# TOOL 2: DISCOVER MOVIES
# ---------------------------------------------------
@tool
def discover_movies(
    genre: str = "",
    min_rating: float = 0,
    min_year: int = 0,
    max_year: int = 0,
) -> str:
    """Discover movies using filters such as genre, rating and release year."""
    params = {
        "language": "en-US",
        "sort_by": "popularity.desc",
        "include_adult": False,
        "include_video": False,
        "vote_count.gte": 50,
    }

    genre_map = {
        "action": 28,
        "adventure": 12,
        "animation": 16,
        "comedy": 35,
        "crime": 80,
        "documentary": 99,
        "drama": 18,
        "family": 10751,
        "fantasy": 14,
        "history": 36,
        "horror": 27,
        "music": 10402,
        "mystery": 9648,
        "romance": 10749,
        "science fiction": 878,
        "sci-fi": 878,
        "thriller": 53,
        "war": 10752,
        "western": 37,
    }

    if genre:
        genre_lower = genre.lower().strip()
        if genre_lower in genre_map:
            params["with_genres"] = genre_map[genre_lower]

    if min_rating > 0:
        params["vote_average.gte"] = min_rating

    if min_year > 0:
        params["primary_release_date.gte"] = f"{min_year}-01-01"

    if max_year > 0:
        params["primary_release_date.lte"] = f"{max_year}-12-31"

    data = tmdb_request("discover/movie", params)

    if "error" in data:
        return data["error"]

    results = data.get("results", [])

    if not results:
        return "No movies matched the requested filters."

    return format_movies(results, limit=10)


# ---------------------------------------------------
# TOOL 3: LATEST MOVIES
# ---------------------------------------------------
@tool
def get_latest_movies() -> str:
    """Get recently released movies."""
    today = date.today()
    start_date = today - timedelta(days=180)

    params = {
        "language": "en-US",
        "sort_by": "primary_release_date.desc",
        "include_adult": False,
        "include_video": False,
        "primary_release_date.gte": start_date.strftime("%Y-%m-%d"),
        "primary_release_date.lte": today.strftime("%Y-%m-%d"),
        "vote_count.gte": 5,
    }

    data = tmdb_request("discover/movie", params)

    if "error" in data:
        return data["error"]

    results = data.get("results", [])

    if not results:
        return "No recent movies found."

    return format_movies(results, limit=10)


# ---------------------------------------------------
# TOOL 4: POPULAR MOVIES
# ---------------------------------------------------
@tool
def get_popular_movies() -> str:
    """Get currently popular movies."""
    data = tmdb_request(
        "movie/popular",
        {
            "language": "en-US",
            "page": 1,
        },
    )

    if "error" in data:
        return data["error"]

    results = data.get("results", [])

    if not results:
        return "No popular movies found."

    return format_movies(results, limit=10)


# ---------------------------------------------------
# TOOL 5: SIMILAR MOVIES
# ---------------------------------------------------
@tool
def get_similar_movies(movie_name: str) -> str:
    """Find movies similar to a specific movie."""
    search_data = tmdb_request(
        "search/movie",
        {
            "query": movie_name,
            "language": "en-US",
            "include_adult": False,
        },
    )

    if "error" in search_data:
        return search_data["error"]

    search_results = search_data.get("results", [])

    if not search_results:
        return f"Could not find '{movie_name}'."

    movie_id = search_results[0].get("id")

    if not movie_id:
        return "Could not identify the movie."

    similar_data = tmdb_request(
        f"movie/{movie_id}/similar",
        {
            "language": "en-US",
            "page": 1,
        },
    )

    if "error" in similar_data:
        return similar_data["error"]

    results = similar_data.get("results", [])

    if not results:
        return f"No similar movies found for '{movie_name}'."

    return format_movies(results, limit=10)


# ---------------------------------------------------
# LANGGRAPH AGENT
# ---------------------------------------------------
tools = [
    search_movies,
    discover_movies,
    get_latest_movies,
    get_popular_movies,
    get_similar_movies,
]

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)


def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    },
)
builder.add_edge("tools", "agent")

graph = builder.compile()


# ---------------------------------------------------
# STREAMLIT USER INTERFACE
# ---------------------------------------------------
st.markdown("### Ask MoviePedia")
st.write(
    "Try: **Tell me about Interstellar**, "
    "**Suggest me 10 highly rated thriller movies**, "
    "**Show me the latest movies**, or "
    "**Give me movies similar to Interstellar**."
)

query = st.chat_input("Ask for movie recommendations...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("MoviePedia is searching TMDB and generating a response..."):
            try:
                result = graph.invoke(
                    {
                        "messages": [
                            HumanMessage(content=query)
                        ]
                    }
                )

                answer = result["messages"][-1].content
                st.markdown(answer)

            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.divider()
st.caption("Movie data provided by TMDB. AI reasoning powered by Groq.")
