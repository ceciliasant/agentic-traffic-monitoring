import asyncio, sys, httpx, pandas as pd
import mcp_server as s

async def main(start, end):
    h = {"Authorization": f"Bearer {s.auth_service.get_token()}"}
    base = s._base_url(True)
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(f"{base}events/footballGames:EstadioDragao",
                        headers=h, params={"since": start, "until": end})
        games = sorted({str(i.get("value", {}).get("properties", {}).get("timestamp"))
                        for i in r.json() if i.get("value", {}).get("properties", {}).get("timestamp")})
        print("football:", games)
        for d in pd.date_range(start, end, freq="D"):
            w = await s._get_weather(c, f"{base}events/historicWeatherData:dailyAverage", h, d)
            print(d.date(), d.day_name()[:3], w, "HOLIDAY" if d.date() in s.HOLIDAYS else "")

asyncio.run(main(sys.argv[1], sys.argv[2]))