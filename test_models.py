import os
from openai import OpenAI

try:
    api_key = os.environ.get('NVIDIA_API_KEY', 'dummy')
    client = OpenAI(base_url='https://integrate.api.nvidia.com/v1', api_key=api_key)
    models = client.models.list()
    for m in models.data:
        print(m.id)
except Exception as e:
    print(e)
