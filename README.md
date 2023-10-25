# LLM API

Welcome to the LLM-API project! This endeavor opens the door to the exciting world of Large Language Models (LLMs) by offering a versatile API that allows you to effortlessly run a variety of LLMs on different consumer hardware configurations. Whether you prefer to operate these powerful models within Docker containers or directly on your local machine, this solution is designed to adapt to your preferences.

With LLM-API, all you need to get started is a simple YAML configuration file. the app streamlines the process by automatically downloading the model of your choice and executing it seamlessly. Once initiated, the model becomes accessible through a unified and intuitive API.

There is also a client that's reminiscent of OpenAI's approach, making it easy to harness the capabilities of your chosen LLM. You can find the Python at [llm-api-python](https://github.com/1b5d/llm-api-python)

In addition to this, a LangChain integration exists, further expanding the possibilities and potential applications of LLM-API. You can explore this integration at [langchain-llm-api](https://github.com/1b5d/langchain-llm-api)

Whether you're a developer, researcher, or enthusiast, the LLM-API project simplifies the use of Large Language Models, making their power and potential accessible to all.

LLM enthusiasts, developers, researchers, and creators are invited to join this growing community. Your contributions, ideas, and feedback are invaluable in shaping the future of LLM-API. Whether you want to collaborate on improving the core functionality, develop new integrations, or suggest enhancements, your expertise is highly appreciated

### Tested with

- [x] Different Llama based-models in different versions such as (Llama, Alpaca, Vicuna, Llama 2 ) on CPU using llama.cpp
- [x] Llama & Llama 2 based models using GPTQ-for-LLaMa
- [x] Generic huggingface pipeline e.g. gpt-2, MPT
- [x] Mistral 7b
- [x] OpenAI-like interface using [llm-api-python](https://github.com/1b5d/llm-api-python)
- [ ] Support RWKV-LM

# Usage

To run LLM-API on a local machine, you must have a functioning Docker engine. The following steps outline the process for running LLM-API:

1. **Create a Configuration File**: Begin by creating a `config.yaml` file with the configurations as described below.

```
models_dir: /models     # dir inside the container
model_family: llama     # also `gptq_llama` or `huggingface`
setup_params:
  key: value
model_params:
  key: value
```

`setup_params` and `model_params` are model specific, see below for model specific configs.

You can override any of the above mentioned configs using environment vars prefixed with `LLM_API_` for example: `LLM_API_MODELS_DIR=/models`

2. **Run LLM-API Using Docker**: Execute the following command in your terminal:

```
docker run -v $PWD/models/:/models:rw -v $PWD/config.yaml:/llm-api/config.yaml:ro -p 8000:8000 --ulimit memlock=16000000000 1b5d/llm-api
```

This command launches a Docker container and mounts your local directory for models, the configuration file, and maps port 8000 for API access.

**Alternatively**, you can use the provided `docker-compose.yaml` file within this repository and run the application using Docker Compose. To do so, execute the following command:

```
docker compose up
```

Upon the first run, LLM-API will download the model from Hugging Face, based on the configurations defined in the `setup_params` of your `config.yaml` file. It will then name the local model file accordingly. Subsequent runs will reference the same local model file and load it into memory for seamless operation


## Endpoints

The LLM-API provides a standardized set of endpoints that are applicable across all Large Language Models (LLMs). These endpoints enable you to interact with the models effectively. Here are the primary endpoints:

### 1. Generate Text

- **POST /generate**
  - Request Example:
    ```json
    {
        "prompt": "What is the capital of France?",
        "params": {
            // Additional parameters...
        }
    }
    ```
  - Description: Use this endpoint to generate text based on a given prompt. You can include additional parameters for fine-tuning and customization.

### 2. Async Text Generation

- **POST /agenerate**
  - Request Example:
    ```json
    {
        "prompt": "What is the capital of France?",
        "params": {
            // Additional parameters...
        }
    }
    ```
  - Description: This endpoint is designed for asynchronous text generation. It allows you to initiate text generation tasks that can run in the background while your application continues to operate.

### 3. Text Embeddings

- **POST /embeddings**
  - Request Example:
    ```json
    {
        "text": "What is the capital of France?"
    }
    ```
  - Description: Use this endpoint to obtain embeddings for a given text. This is valuable for various natural language processing tasks such as semantic similarity and text analysis.


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

Leverage the flexibility of LLM-API by configuring various attributes using the following methods:

1. Pass specific configuration attributes within the `config_params` to fine-tune `AutoConfig`. These attributes allow you to tailor the behavior of your language model further.

2. Modify the model's parameters directly by specifying them within the `model_params`. These parameters are passed into the `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained` initialization calls.

Here's an example of how you can use parameters in the `generate` (or `agenerate`) endpoints, but remember, you can pass any arguments accepted by [transformer's GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig):

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

If you're looking to accelerate inference using a GPU, the `1b5d/llm-api:x.x.x-gpu` image is designed for this purpose. When running the Docker image using Compose, consider utilizing a dedicated Compose file for GPU support:

```
docker compose -f docker-compose.gpu.yaml up
```

**Note**: currenty only `linux/amd64` architecture is supported for gpu images

## Llama on CPU - using llama.cpp

Utilizing Llama on a CPU is made simple by configuring the model usage in a local `config.yaml` file. Below are the possible configurations:

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

Ensure to specify the repo_id and filename parameters to point to a Hugging Face repository where the desired model is hosted. The application will then handle the download for you.

The following example demonstrates the various parameters that can be sent to the Llama generate and agenerate endpoints:

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

## Llama on GPU - using GPTQ-for-LLaMa

**Important Note**: Before running Llama or Llama 2 on GPU, make sure to install the [NVIDIA Driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) on your host machine. You can verify the NVIDIA environment by executing the following command:

```
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu20.04 nvidia-smi
```

You should see a table displaying the current NVIDIA driver version and related information, confirming the proper setup.

When running the Llama model with GPTQ-for-LLaMa 4-bit quantization, you can use a specialized Docker image designed for this purpose, `1b5d/llm-api:x.x.x-gpu`, as an alternative to the default image. You can run this mode using a separate Docker Compose file:

```
docker compose -f docker-compose.gpu.yaml up
```

Or by directly running the container:

```
docker run --gpus all -v $PWD/models/:/models:rw -v $PWD/config.yaml:/llm-api/config.yaml:ro -p 8000:8000 1b5d/llm-api:x.x.x-gpu
```

**Important Note**: The `llm-api:x.x.x-gptq-llama-cuda` and `llm-api:x.x.x-gptq-llama-triton` images have been deprecated. Please switch to the `1b5d/llm-api:x.x.x-gpu` image when GPU support is required

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

# Credits

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for making it possible to run Llama models on CPU. 
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for the python bindings lib for `llama.cpp`.
- [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) for providing a GPTQ implementation for Llama based models.
- Huggingface for the great ecosystem of tooling they provide.
