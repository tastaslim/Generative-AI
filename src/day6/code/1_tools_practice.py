import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# from pydantic import BaseModel, Field


# class AddInput(BaseModel):
#     a: int = Field(description="first number")
#     b: int = Field(description="second number")
#
#
# @tool("add_number", args_schema=AddInput)
# def add_number(a: int, b: int) -> int:
#     """Add two integers and return their sum."""
#     return a + b
#
#
# print(add_number.invoke({"a": 1, "b": 2}))


class CurrencyExchangeInput(BaseModel):
    url: str = Field(description="Public API URL to fetch exchange rates from")


class CurrencyExchangeRate(BaseModel):
    """The currency exchange rate for USD."""

    rates: dict = Field(description="Exchange rates")
    documentation: str = Field(description="Exchange rate documentation link")
    provider: str = Field(description="Exchange rate provider")
    result: str = Field(description="Exchange rate result whether success or failure")
    base_code: str = Field(description="Exchange rate base code")
    time_last_update_utc: str = Field(description="Exchange rate time last update")


@tool("get_currency_exchange_rate", args_schema=CurrencyExchangeInput)
def get_currency_exchange_rate(url: str) -> CurrencyExchangeRate:
    """Fetch data from public api to get exchange rate based on US Dollars"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return CurrencyExchangeRate(
            rates=data["rates"],
            documentation=data["documentation"],
            provider=data["provider"],
            base_code=data["base_code"],
            time_last_update_utc=data["time_last_update_utc"],
            result=data["result"],
        )
    except Exception as e:
        raise e


if __name__ == "__main__":
    current_rates: CurrencyExchangeRate = get_currency_exchange_rate.invoke(
        {"url": "https://open.er-api.com/v6/latest/USD"}
    )
    print(current_rates.rates)
