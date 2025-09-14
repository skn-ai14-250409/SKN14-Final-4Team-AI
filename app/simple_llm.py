from openai import OpenAI


class SimpleChatLLM:

    client: OpenAI  = None

    def __init__(self, model:str = "gpt-4.1-nano", client: OpenAI = None):
        if SimpleChatLLM.client is None:
            SimpleChatLLM.client = OpenAI()

        self._client = client if client else SimpleChatLLM.client
        self._model  = model


    def __call__(self, message:str, history:list[dict] = None, model=None, *args, **kwargs):
        history = history or []
        history.append({"role": "user", "content": message})

        resp = self.client.chat.completions.create(
            model=self._model if model is None else model,
            messages=history,
        )
        return resp.choices[0].message.content