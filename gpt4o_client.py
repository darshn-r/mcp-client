import os
import asyncio
import httpx
from typing import List, Optional, Dict, Any


class GPTClient:
    def __init__(
        self,
        endpoint_url: str,
        deployment_name: str,
        api_version: str = "2024-08-01-preview",
        timeout: float = 10.0,
    ):
        subscription_key = os.getenv("BOSCH_API_KEY")
        if not subscription_key:
            raise ValueError("Environment variable 'BOSCH_API_KEY' is not set.")

        self.endpoint_url = f"{endpoint_url}/deployments/{deployment_name}/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "genaiplatform-farm-subscription-key": subscription_key,
        }
        self.params = {
            "api-version": api_version
        }
        self.timeout = timeout

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        payload.update(kwargs)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.endpoint_url,
                headers=self.headers,
                params=self.params,
                json=payload
            )
            response.raise_for_status()
            return response.json()


async def interactive_chat():
    client = GPTClient(
        endpoint_url="https://aoai-farm.bosch-temp.com/api/openai",  # Replace with actual base URL
        deployment_name="askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18"
    )

    system_prompt = input("System prompt [default: You are a helpful assistant]: ").strip()
    if not system_prompt:
        system_prompt = "You are a helpful assistant."

    messages = [{"role": "system", "content": system_prompt}]

    print("\nStart chatting (type 'exit' to quit):\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        try:
            response = await client.chat(messages)
            reply = response['choices'][0]['message']['content']
            print(f"Assistant: {reply}\n")
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(interactive_chat())
