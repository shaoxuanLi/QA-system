"""硅基流动 API 调用：Qwen/Qwen2.5-7B-Instruct。

提供 chat() 单条调用 + chat_batch() 多线程并发调用。
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from tqdm import tqdm


class SiliconFlowLLM:
    def __init__(
        self,
        api_url: str = "https://api.siliconflow.cn/v1/chat/completions",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        api_key: str = "",
        api_key_env: str = "SILICONFLOW_API_KEY",
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.api_url = api_url
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(
                f"未提供 API key。请在 config.yaml 设置 generator.api_key，"
                f"或导出环境变量 {api_key_env}。"
            )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    # ---------- 单条 ----------
    def chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code == 429:
                    # rate limit, 指数退避
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.5 ** attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    # ---------- 批量 ----------
    def chat_batch(
        self,
        message_batches: list[list[dict]],
        num_workers: int = 8,
        desc: str = "LLM",
    ) -> list[str]:
        results: list[str] = [""] * len(message_batches)
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            future_to_idx = {
                ex.submit(self.chat, msgs): idx
                for idx, msgs in enumerate(message_batches)
            }
            for fut in tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc=desc,
            ):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:  # noqa: BLE001
                    results[idx] = f"[ERROR] {e}"
        return results
