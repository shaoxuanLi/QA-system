模型名称：Qwen/Qwen2.5-7B-Instruct
api_key: sk-xxxx


调用方式
```
curl --request POST \
  --url https://api.siliconflow.cn/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "Pro/zai-org/GLM-4.7",
    "messages": [
      {"role": "system", "content": "你是一个有用的助手"},
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ]
  }'
```
