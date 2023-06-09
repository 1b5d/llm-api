# LLM API

This application can be used to run LLMs (Large Language Models) in docker containers, it's built in a generic way so that it can be reused for multiple types of models.

The main motivation to start this project, was to be able to use different LLMs running on a local machine or a remote server with [langchain](https://github.com/hwchase17/langchain) using [langchain-llm-api](https://github.com/1b5d/langchain-llm-api)

Tested with the following models : 

- Llama 7b - ggml
- Llama 13b - ggml
- Llama 30b - ggml
- Alpaca 7b - ggml
- Alpaca 13b - ggml
- Alpaca 30b - ggml
- Vicuna 13b - ggml
- Koala 7b - ggml
- Vicuna GPTQ 7B-4bit-128g
- Vicuna GPTQ 13B-4bit-128g
- Koala GPTQ 7B-4bit-128g
- wizardLM GPTQ 7B-4bit-128g

Contribution for supporting more models is welcomed.

### roadmap

- [x] Write an implementation for Alpaca
- [x] Write an implementation for Llama
- [x] Write an implementation for [Vicuna](https://github.com/lm-sys/FastChat)
- [x] Support GPTQ-for-LLaMa
- [ ] Lora support
- [ ] huggingface pipeline
- [ ] Support OpenAI
- [ ] Support RWKV-LM

# Usage

In order to run this API on a local machine, a running docker engine is needed.

run using docker:

create a `config.yaml` file with the configs described below and then run:

```
docker run -v $PWD/models/:/models:rw -v $PWD/config.yaml:/llm-api/config.yaml:ro -p 8000:8000 --ulimit memlock=16000000000 1b5d/llm-api
```

or use the `docker-compose.yaml` in this repo and run using compose:

```
docker compose up
```

When running for the first time, the app will download the model from huggingface based on the configurations in `setup_params` and name the local model file accordingly, on later runs it looks up the same local file and loads it into memory

## Config

to configure the application, edit `config.yaml` which is mounted into the docker container, the config file looks like this:

```
models_dir: /models     # dir inside the container
model_family: alpaca
setup_params:
  key: value
model_params:
  key: value
```

`setup_params` and `model_params` are model specific, see below for model specific configs.

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


## Llama on CPU - using llama.cpp

Llama and models based on it such as Alpaca and Vicuna are intended only for academic research and any commercial use is prohibited. This project doesn't provide any links to download these models.

You can configure the model usage in a local `config.yaml` file, the configs, here is an example:

```
models_dir: /models     # dir inside the container
model_family: alpaca
setup_params:
  repo_id: user/repo_id
  filename: ggml-model-q4_0.bin
  convert: false
  migrate: false
model_params:
  ctx_size: 2000
  seed: -1
  n_threads: 8
  n_batch: 2048
  n_parts: -1
  last_n_tokens_size: 16
```

Fill `repo_id` and `filename` to a huggingface repo where a model is hosted, and let the application download it for you.

- `convert` refers to https://github.com/ggerganov/llama.cpp/blob/master/convert-unversioned-ggml-to-ggml.py set this to true when you need to use an older model which needs to be converted
- `migrate` refers to https://github.com/ggerganov/llama.cpp/blob/master/migrate-ggml-2023-03-30-pr613.py set this to true when you need to apply this script to an older model which needs to be migrated

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

## Llama / Alpaca on GPU - using GPTQ-for-LLaMa (beta)

**Note**: According to [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), you might want to install the [NVIDIA Driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) on your host machine. Verify that your nvidia environment is properly by running this:

```
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu20.04 nvidia-smi
```

You should see a table showing you the current nvidia driver version and some other info:
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 11.7     |
|-----------------------------------------+----------------------+----------------------+
...
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

You can also run the Llama model using GPTQ-for-LLaMa 4 bit quantization, you can use a docker image specially built for that purpose `1b5d/llm-api:0.0.4-gptq-llama-triton` instead of the default image.

a separate docker-compose file is also available to run this mode:

```
docker compose -f docker-compose.gptq-llama-triton.yaml up
```

or by directly running the container:

```
docker run --gpus all -v $PWD/models/:/models:rw -v $PWD/config.yaml:/llm-api/config.yaml:ro -p 8000:8000 1b5d/llm-api:0.0.4-gptq-llama-triton
```

**Note**: `llm-api:0.0.x-gptq-llama-cuda` image has been deprecated, please switch to the triton image as it seems more reliable

Example config file:

```
models_dir: /models
model_family: gptq_llama
setup_params:
  repo_id: user/repo_id
  filename: <model.safetensors or model.pt>
model_params:
  group_size: 128
  wbits: 4
  cuda_visible_devices: "0"
  device: "cuda:0"
  st_device: 0
```

**Note**: `st_device` is only needed in the case of safetensors model, otherwise you can either remove it or set it to `-1`

Example request:

```
POST /generate

curl --location 'localhost:8000/generate' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "What is the capital of paris",
    "params": {
        "temp": 0.8,
        "top_p": 0.95,
        "min_length": 10,
        "max_length": 50
    }
}'
```

# Credits

credits goes to 
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for making it possible to run Llama and Alpaca models on CPU. 
- [serge](https://github.com/nsarrazin/serge) for providing an example on how to build an API using FastApi
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for the python bindings lib for `llama.cpp`
- [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) for providing a GPTQ implementation for Llama based models.
