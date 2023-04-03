# LLM API

This application can be used to run LLMs (Large Language Models) in docker containers, it's built in a generic way so that it can be reused for multiple types of models

tested with the following models: 

- Llama 7b
- Llama 13b
- Llama 30b
- Llama 65b
- Alpaca 7b
- Alpaca 13b 
- Alpaca 30b

Alpaca is based on the Llama Language model, both Alpaca and Llama models are intended only for academic research and any commercial use is prohibited. We don't provide any links to download these models.

Contribution for supporting more models is welcomed.

### roadmap

- [x] Write an implementation for Alpaca
- [x] Write an implementation for Llama
- [ ] Write an implementation for RWKV-LM
- [ ] Write an implementation for OpenAI
- [ ] Write an implementation for [Vicuna](https://github.com/lm-sys/FastChat)

# Usage

In order to run this API on a local machine, a running docker engine is needed.

run using docker-compose:

```
docker compose up
```

When running for the first time, the app will download the model from huggingface based on the configurations in `setup_params` and name the local model file accordingly, on later runs it looks up the same local file and loads it into memory

to configure the application, edit `config.yaml` which is mounted into the docker container, the config file looks like this:

```
models_dir: /models     # dir inside the container
model_family: alpaca
model_name: 7b
setup_params:
  key: value
model_params:
  key: value
```

`setup_params` and `model_params` are model specific.

You can override any of the above mentioned configs using environment vars prefixed with `LLM_API_` for example: `LLM_API_MODELS_DIR=/models`

## Endpoints

In general all LLMs will have a generalized set of endpoints

```
POST /generate
{
    "prompt": "What is the capital of France?",
    "params": {
        ...
    }
}
```
```
POST /agenerate
{
    "prompt": "What is the capital of France?",
    "params": {
        ...
    }
}
```
```
POST /embeddings
{
    "text": "What is the capital of France?"
}
```


## Llama / Alpaca

You can configure the model usage in a local `config.yaml` file, the configs, here is an example:

```
models_dir: /models     # dir inside the container
model_family: alpaca
model_name: 7b
setup_params:
  repo_id: user/repo_id
  filename: ggml-model-q4_0.bin
  convert: true
  migrate: true
model_params:
  ctx_size: 2000
  seed: -1
  n_threads: 8
  n_batch: 2048
  n_parts: -1
  last_n_tokens_size: 16
```

Fill `repo_id` and `filename` to a huggingface repo where a model is hosted, and let the application download it for you.

- `convert` refers to https://github.com/ggerganov/llama.cpp/blob/master/convert-unversioned-ggml-to-ggml.py set this to true when you need to apply this script
- `migrate` refers to https://github.com/ggerganov/llama.cpp/blob/master/migrate-ggml-2023-03-30-pr613.py set this to true when you need to apply this script

the following example shows the different params you can sent to Alpaca generate and agenerate endpoints:

```
POST /generate

curl --location 'localhost:8000/generate' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "What is the capital of paris",
    "params": {
        "n_predict": 300,
        "temp": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "stop": ["\Q"],
        "repeat_penalty": 1.3
    }
}'
```

# Credits

credits goes to 
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for making it possible to run Llama and Alpaca models on CPU. 
- [serge](https://github.com/nsarrazin/serge) for providing an example on how to build an API using FastApi
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for the python bindings lib for `llama.cpp`
