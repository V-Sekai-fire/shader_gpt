# ShaderGPT

Run popular language models in Unity with pixel shaders.

Comparison to [Unity Sentis](https://unity.com/products/sentis):

* ShaderGPT
  - a tensor library with PyTorch/Transformers-like API;
  - pixel shader backend only, optimized for transformers;
* Sentis
  - a ONNX inference library;
  - multiple backends including compute shader, pixel shader and CPU;

## Supported model

The following architectures are supported:

[GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2),
[GPT-Neo](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neo),
[GPT-NeoX](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox),
[Phi](https://huggingface.co/docs/transformers/main/en/model_doc/phi),
[Llama2](https://huggingface.co/docs/transformers/main/en/model_doc/llama2),
[Mistral](https://huggingface.co/docs/transformers/main/en/model_doc/mistral),
[Qwen2](https://huggingface.co/docs/transformers/main/en/model_doc/qwen2),
[Gemma](https://huggingface.co/docs/transformers/main/en/model_doc/gemma),
[Phi3](https://huggingface.co/docs/transformers/main/en/model_doc/phi3),
OpenELM (experimental)

The following quantizations are supported:

[GPTQ](https://github.com/IST-DASLab/gptq/) (4bit, 8bit, act-order)

The following models are tested:

* [TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M) (GPT-Neo)
* [pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped) (GPT-NeoX)
* [Minueza-32M-UltraChat](https://huggingface.co/Felladrin/Minueza-32M-UltraChat) (Mistral)
* [LaMini-GPT-124M](https://huggingface.co/MBZUAI/LaMini-GPT-124M) (GPT2)
* [Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat) (Qwen2)
* [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) (Llama2)
* [phi-1_5](https://huggingface.co/microsoft/phi-1_5) (Phi)
* [gemma-2b-it](https://huggingface.co/google/gemma-2b-it) (Gemma)
* [OpenELM-270M-Instruct](https://huggingface.co/apple/OpenELM-270M-Instruct) (OpenELM)
* [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (Phi3)

## Import a model

You need Python to convert models from Hugging Face. [Install Python](https://www.python.org/downloads/) and run `pip install -r requirements.txt` in `Python` folder to install dependencies, including PyTorch and Transformers. 

The conversion script is located in `Python` folder. For example, if you run `convert.py roneneldan/TinyStories-33M ../Model/TinyStories-33M`, the script will download [TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M) from Hugging Face and generate a folder `Model/TinyStories-33M` which contains JSON configurations and EXR images for model parameters.

In Unity editor, select the generated folder, and click `Assets/ShaderGPT/ImportModel` in the menu. The editor script will reimport the textures and create a MonoBehaviour for running and testing the model. Please refer to the example scene to learn how to set it up.

## Example scene

The demo scene `Example/Main.unity` contains an example of [TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M) inference. A MonoBehaviour creates a list of materials and render textures at runtime to represent the neural network, and execute the network by calling `Graphics.Blit`. The MonoBehaviour can also "bake" the network in editor by saving the materials and render textures as assets.

The other scene `Example/Udon.unity` contains an example of executing the baked network. This is the preferred way of inference because there is no material or texture allocation at runtime. It also runs in Udon if VRCSDK is installed.