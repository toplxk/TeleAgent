import requests
from langchain.tools import tool


@tool()
def multipy(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    url = "https://eolink.o.apispace.com/456456/weather/v001/now"
    payload = {"areacode": "101010100", "city": city}
    headers = {
        "X-APISpace-Token": "h72uhbhva2t9s4k5ozrx23f8vmwqccs0",
    }
    response = requests.get(url, params=payload, headers=headers, timeout=10)
    return response.text
