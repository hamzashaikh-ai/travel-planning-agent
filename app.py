import streamlit as st
import json
import requests
import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

# ── Load data ──────────────────────────────────────────────
with open("data/flights.json") as f:
    flights_data = json.load(f)
with open("data/hotels.json") as f:
    hotels_data = json.load(f)
with open("data/places.json") as f:
    places_data = json.load(f)

# ── Tools ──────────────────────────────────────────────────
@tool
def search_flights(source: str, destination: str) -> str:
    """Search for available flights from source city to destination city."""
    results = [
        f for f in flights_data
        if f["from"].lower() == source.lower()
        and f["to"].lower() == destination.lower()
    ]
    if not results:
        return f"No flights found from {source} to {destination}."
    results.sort(key=lambda x: x["price"])
    output = f"Available flights from {source} to {destination}:\n"
    for f in results[:3]:
        output += (f"- {f['airline']} | ₹{f['price']} | "
                   f"Departs: {f['departure_time'][11:16]} | "
                   f"Arrives: {f['arrival_time'][11:16]}\n")
    return output

@tool
def recommend_hotels(city: str, max_price: int = 10000) -> str:
    """Recommend top-rated hotels in a given city within a price range."""
    results = [
        h for h in hotels_data
        if h["city"].lower() == city.lower()
        and h["price_per_night"] <= max_price
    ]
    if not results:
        return f"No hotels found in {city} under ₹{max_price}/night."
    results.sort(key=lambda x: x["stars"], reverse=True)
    output = f"Top hotels in {city}:\n"
    for h in results[:3]:
        output += (f"- {h['name']} | {h['stars']}⭐ | "
                   f"₹{h['price_per_night']}/night | "
                   f"Amenities: {', '.join(h['amenities'])}\n")
    return output

@tool
def find_places(city: str) -> str:
    """Find top tourist attractions and places of interest in a city."""
    results = [p for p in places_data if p["city"].lower() == city.lower()]
    if not results:
        return f"No places found in {city}."
    results.sort(key=lambda x: x["rating"], reverse=True)
    output = f"Top places to visit in {city}:\n"
    for p in results[:6]:
        output += f"- {p['name']} | Type: {p['type']} | Rating: {p['rating']}⭐\n"
    return output

@tool
def get_weather(city: str) -> str:
    """Get real-time weather forecast for a city for the next 5 days."""
    city_coords = {
        "delhi": (28.6139, 77.2090), "mumbai": (19.0760, 72.8777),
        "goa": (15.2993, 74.1240), "bangalore": (12.9716, 77.5946),
        "hyderabad": (17.3850, 78.4867), "chennai": (13.0827, 80.2707),
        "kolkata": (22.5726, 88.3639), "jaipur": (26.9124, 75.7873),
        "pune": (18.5204, 73.8567), "ahmedabad": (23.0225, 72.5714),
    }
    coords = city_coords.get(city.lower())
    if not coords:
        return f"Weather data not available for {city}."
    lat, lon = coords
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
           f"&timezone=auto&forecast_days=5")
    try:
        data = requests.get(url, timeout=10).json()
        dates = data["daily"]["time"]
        max_temps = data["daily"]["temperature_2m_max"]
        min_temps = data["daily"]["temperature_2m_min"]
        rain = data["daily"]["precipitation_sum"]
        output = f"Weather forecast for {city}:\n"
        for i in range(5):
            condition = "🌧️ Rainy" if rain[i] > 2 else "⛅ Cloudy" if rain[i] > 0 else "☀️ Sunny"
            output += (f"- {dates[i]}: {condition} | "
                      f"Max: {max_temps[i]}°C | Min: {min_temps[i]}°C\n")
        return output
    except Exception as e:
        return f"Could not fetch weather: {str(e)}"

@tool
def estimate_budget(flight_price: int, hotel_price_per_night: int, num_days: int) -> str:
    """Estimate total trip budget based on flight, hotel, and number of days."""
    hotel_total = hotel_price_per_night * num_days
    food_transport = 800 * num_days
    total = flight_price + hotel_total + food_transport
    return (f"\nBudget Breakdown:\n"
            f"- ✈️  Flight:            ₹{flight_price}\n"
            f"- 🏨  Hotel ({num_days} nights): ₹{hotel_total}\n"
            f"- 🍽️  Food & Transport:  ₹{food_transport}\n"
            f"------------------------------\n"
            f"- 💰  Total Estimate:    ₹{total}\n")

# ── LLM & Agent ────────────────────────────────────────────
# Works both locally (.env) and on Streamlit Cloud (secrets)
api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0
)

tools = [search_flights, recommend_hotels, find_places, get_weather, estimate_budget]

system_prompt = """You are an expert AI travel planning assistant.

When a user asks you to plan a trip, use the available tools in this order:
1. search_flights — find flight from source to destination
2. recommend_hotels — find hotels at destination
3. find_places — find tourist attractions at destination
4. get_weather — get weather forecast for destination
5. estimate_budget — calculate total trip cost

After collecting all information, generate a complete structured itinerary with:
- Trip Summary
- Flight Selected
- Hotel Recommendation
- Day-wise Itinerary (distribute places across days)
- Weather for each day
- Budget Breakdown
- Total Estimated Cost

Always use ₹ for currency."""

agent_executor = create_react_agent(llm, tools, prompt=system_prompt)

# ── Streamlit UI ───────────────────────────────────────────
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="centered")

st.title("✈️ AI Travel Planning Assistant")
st.markdown("Powered by LangChain + Groq LLaMA | Plan your perfect trip instantly")
st.divider()

col1, col2 = st.columns(2)
with col1:
    source = st.text_input("🏠 From (Source City)", placeholder="e.g. Hyderabad")
with col2:
    destination = st.text_input("📍 To (Destination City)", placeholder="e.g. Delhi")

col3, col4 = st.columns(2)
with col3:
    num_days = st.slider("📅 Number of Days", min_value=1, max_value=7, value=3)
with col4:
    budget = st.number_input("💰 Budget (₹)", min_value=5000, max_value=100000,
                              value=20000, step=1000)

if st.button("🗺️ Plan My Trip", use_container_width=True, type="primary"):
    if not source or not destination:
        st.warning("Please enter both source and destination cities.")
    else:
        with st.spinner("🤖 AI Agent is planning your trip..."):
            query = (f"Plan a {num_days}-day trip to {destination} from {source}. "
                     f"My total budget is around ₹{budget}.")
            try:
                response = agent_executor.invoke({
                    "messages": [{"role": "user", "content": query}]
                })
                result = response["messages"][-1].content

                st.success("✅ Your itinerary is ready!")
                st.divider()
                st.markdown(result)

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

st.divider()
st.caption("Data: JSON datasets + Open-Meteo API | LLM: Groq LLaMA 3.3 70B")