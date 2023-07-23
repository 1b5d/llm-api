# LLM API

This application can be used to run LLMs (Large Language Models) in docker containers, it's built in a generic way so that it can be reused for multiple types of models.

The main motivation to start this project, was to be able to use different LLMs running on a local machine or a remote server with [langchain](https://github.com/hwchase17/langchain) using [langchain-llm-api](https://github.com/1b5d/langchain-llm-api)

Contribution for supporting more models is welcomed.

### roadmap

- [x] Write an implementation for Alpaca
- [x] Write an implementation for Llama
- [x] Write an implementation for [Vicuna](https://github.com/lm-sys/FastChat)
- [x] Support GPTQ-for-LLaMa
- [x] huggingface pipeline
- [x] Llama 2
- [ ] Lora support
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
model_family: llama
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

## Huggingface transformers

Generally models for which can be inferenced using transformer's `AutoConfig`, `AutoModelForCausalLM` and `AutoTokenizer` can run using the `model_family: huggingface` config, the following is an example (runs one of the MPT models):

```
models_dir: /models
model_family: huggingface
setup_params:
  repo_id: <repo_id>
  tokenizer_repo_id: <repo_id>
  trust_remote_code: True
  config_params:
    init_device: cuda:0
    attn_config:
      attn_impl: triton
model_params:
  device_map: "cuda:0"
  trust_remote_code: True
  torch_dtype: torch.bfloat16
```

Note that you can pass configuration attributes in `config_params` in order to configure `AutoConfig` with additional attributes.

Configurations in `model_params` are directly passed into the `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained` initialization calls.

The following is an example with some parameters passed to the `generate` (or `agenerate`) endpoints, but you can pass any argments which is accepted by [transformer's GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig):

```
POST /generate

curl --location 'localhost:8000/generate' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "What is the capital of paris",
    "params": {
        "max_length": 25,
        "max_new_tokens": 25,
        "do_sample": true,
        "top_k": 40,
        "top_p": 0.95
    }
}'
```

To be able to accelerate inference using GPU, the `1b5d/llm-api:x.x.x-gpu` image can be used for this purpose. When running the docker image using compose, a dedicate compose file can be used:

```
docker compose -f docker-compose.gpu.yaml up
```

Note: currenty only `linux/amd64` architecture is supported for gpu images

## Llama on CPU - using llama.cpp

You can configure the model usage in a local `config.yaml` file, the configs, here is an example:

```
models_dir: /models
model_family: llama
setup_params:
  repo_id: user/repo_id
  filename: ggml-model-q4_0.bin
model_params:
  n_ctx: 512
  n_parts: -1
  n_gpu_layers: 0
  seed: -1
  use_mmap: True
  n_threads: 8
  n_batch: 2048
  last_n_tokens_size: 64
  lora_base: null
  lora_path: null
  low_vram: False
  tensor_split: null
  rope_freq_base: 10000.0
  rope_freq_scale: 1.0
  verbose: True
```

Fill `repo_id` and `filename` to a huggingface repo where a model is hosted, and let the application download it for you.

- `convert` refers to https://github.com/ggerganov/llama.cpp/blob/master/convert-unversioned-ggml-to-ggml.py set this to true when you need to use an older model which needs to be converted
- `migrate` refers to https://github.com/ggerganov/llama.cpp/blob/master/migrate-ggml-2023-03-30-pr613.py set this to true when you need to apply this script to an older model which needs to be migrated

the following example shows the different params you can sent to Llama generate and agenerate endpoints:

```
POST /generate

curl --location 'localhost:8000/generate' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "What is the capital of paris",
    "params": {
        "suffix": null or string,
        "max_tokens": 128,
        "temperature": 0.8,
        "top_p": 0.95,
        "logprobs": null or integer,
        "echo": False,
        "stop": ["\Q"],
        "frequency_penalty: 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.1
        "top_k": 40,
    }
}'
```

## Llama / Alpaca on GPU - using GPTQ-for-LLaMa

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

You can run the Llama model using GPTQ-for-LLaMa 4 bit quantization, you can use a docker image specially built for that purpose `1b5d/llm-api:x.x.x-gpu` instead of the default image.

a separate docker-compose file is also available to run this mode:

```
docker compose -f docker-compose.gpu.yaml up
```

or by directly running the container:

```
docker run --gpus all -v $PWD/models/:/models:rw -v $PWD/config.yaml:/llm-api/config.yaml:ro -p 8000:8000 1b5d/llm-api:x.x.x-gpu
```

**Note**: `llm-api:x.x.x-gptq-llama-cuda` and `llm-api:x.x.x-gptq-llama-triton` images have been deprecated, please switch to the `1b5d/llm-api:x.x.x-gpu` image when gpu support is required

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
```

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


This app was tested with the following models : 

- Llama and models based on it (ALpaca, Vicuna, Koala ..etc.) using the ggml format
- Llama and models based on it (ALpaca, Vicuna, Koala ..etc.) using the GPTQ format (4bit-128g)
- Popular models on huggingface (MPT, GPT2, Falcon) using PT format 
- Llama 2 using ggml and gptq formats

# Credits

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for making it possible to run Llama models on CPU. 
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for the python bindings lib for `llama.cpp`
- [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) for providing a GPTQ implementation for Llama based models.
