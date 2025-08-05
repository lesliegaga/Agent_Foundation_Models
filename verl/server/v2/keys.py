serper_key_pool = [
    {
        "url": "https://google.serper.dev/search",
        "key": "..."
    }, 
    {
        "url": "https://google.serper.dev/search",
        "key": "..."
    }, 
    {
        "url": "https://google.serper.dev/search",
        "key": "..."
    }
]

qwen_api_pool = [
    {
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key": "sk-...",
        "model": "qwen2.5-72b-instruct",
    },
    {
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key": "sk-...",
        "model": "qwen2.5-72b-instruct",
    },
    {
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key": "sk-...",
        "model": "qwen2.5-72b-instruct",
    },
    {
        "url": "https://api.uniapi.vip/v1",
        "key": "sk-...",
        "model": "qwen2.5-72b-instruct",
    },
    {
        "url": "https://api.turboai.vip/v1",
        "key": "sk-...",
        "model": "qwen2.5-72b-instruct",
    },
        {
        "url": "https://api.uniapi.vip/v1",
        "key": "sk-...",
        "model": "qwen2.5-72b-instruct",
    },
    {
        "url": "https://api.turboai.vip/v1",
        "key": "sk-...",
        "model": "qwen2.5-72b-instruct",
    }
]


jina_api_pool = [
    {
        "key": "jina_..."
    },
    {
        "key": "jina_..."
    }
]


def get_serper_api() -> dict:
    """
    Returns a random Serper API from the pool.
    """
    import random
    return random.choice(serper_key_pool)

def get_qwen_api() -> dict:
    """
    Returns a random Qwen API from the pool.
    """
    import random
    return random.choice(qwen_api_pool)


def get_jina_api() -> dict:
    """
    Returns a random Jina API from the pool.
    """
    import random
    return random.choice(jina_api_pool)