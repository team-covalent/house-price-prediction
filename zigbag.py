from typing import Any, Literal
import aiohttp
import contextlib

ZIGBAG_API = "https://apis.zigbang.com"


def content_type(response: Any) -> Any:
    if response.content_type == "text/html":
        return response.text()
    with contextlib.suppress(Exception):
        return response.json()
    return response.text()


class ZigBagClient:
    def __init__(self) -> None:
        self.__session: aiohttp.ClientSession | None = None

    def clear(self) -> None:
        if self.__session and self.__session.closed:
            self.__session = None

    @staticmethod
    def set_browser_header(header: dict[str, str]) -> dict[str, str]:
        header["Accept"] = "application/json, text/plain, */*"
        header["Accept-Encoding"] = "gzip, deflate, br"
        header["Accept-Language"] = "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
        header["Origin"] = "https://zigbang.com"
        header["Referer"] = "https://zigbang.com"
        header["Sec-Ch-Ua"] = (
            '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"'
        )
        header["Sec-Ch-Ua-Mobile"] = "?0"
        header["Sec-Ch-Ua-Platform"] = '"Windows"'
        header["Sec-Fetch-Dest"] = "empty"
        header["Sec-Fetch-Mode"] = "cors"
        header["Sec-Fetch-Site"] = "same-site"
        header["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
        return header

    async def request(self, *args, **kwargs) -> Any:
        headers: dict[str, Any] = kwargs.get("headers", {})
        headers = self.set_browser_header(headers)

        kwargs["headers"] = headers

        if not self.__session:
            self.__session = aiohttp.ClientSession()

        async with self.__session.request(*args, **kwargs) as response:
            data = await content_type(response)

            if 300 > response.status >= 200:
                return data
            else:
                raise Exception(f"HTTP Status: {response.status}, {data}")

    async def close(self) -> None:
        if self.__session:
            await self.__session.close()

    async def search(
        self,
        query: str,
        service_type: Literal["아파트", "빌라", "원룸", "오피스텔", "상가"],
    ) -> list[dict] | None:
        """직방 API를 통해 검색 결과를 가져옵니다."""

        response = await self.request(
            "GET",
            ZIGBAG_API + "/v2/search",
            params={
                "leaseYn": "N",
                "q": query,
                "serviceType": service_type,
            },
        )
        if len(response["items"]) == 0:
            return

        return response["items"]

    async def get_apart_info(
        self, room_id: int, period: str = "3y"
    ) -> list[dict] | None:
        """
        직방 API를 통해 아파트 가격 정보 (3.3m^2 당 평균 가격)을 가져옵니다.
        """

        response = await self.request(
            "GET",
            ZIGBAG_API + "/apt/danjis/" + str(room_id) + "/price-chart",
            params={"period": period},
        )
        if not response.get("list"):
            raise Exception(f"No room({room_id}) found")

        apt_prices: list[dict] = response["list"][0]["data"]
        return apt_prices
