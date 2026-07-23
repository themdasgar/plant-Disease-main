import requests

API_KEY = "dba400c61986e185aad5de3ab1a47984"

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url).json()

        return {
            "temp": response["main"]["temp"],
            "humidity": response["main"]["humidity"],
            "weather": response["weather"][0]["main"]
        }

    except:
        return None