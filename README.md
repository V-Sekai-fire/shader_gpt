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

The following language models are supported (\* means experimental):

| Supported architectures | Tested models |
|-------------------------|---------------|
| [GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2)         | [LaMini-GPT-124M](https://huggingface.co/MBZUAI/LaMini-GPT-124M)
| [GPT-Neo](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neo)   | [TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M)
| [GPT-NeoX](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox) | [pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped)
| [Phi](https://huggingface.co/docs/transformers/main/en/model_doc/phi)           | [phi-1_5](https://huggingface.co/microsoft/phi-1_5)
| [Phi3](https://huggingface.co/docs/transformers/main/en/model_doc/phi3)         | [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
| [Llama2](https://huggingface.co/docs/transformers/main/en/model_doc/llama2)     | [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
| [Llama3](https://huggingface.co/docs/transformers/main/en/model_doc/llama3)     |
| [Mistral](https://huggingface.co/docs/transformers/main/en/model_doc/mistral)   |
| [StableLM](https://huggingface.co/docs/transformers/main/en/model_doc/stablelm) |
| [Qwen2](https://huggingface.co/docs/transformers/main/en/model_doc/qwen2)       | [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
| [Gemma](https://huggingface.co/docs/transformers/main/en/model_doc/gemma)       | [gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
| OpenELM\*                                                                       | [OpenELM-270M-Instruct](https://huggingface.co/apple/OpenELM-270M-Instruct)
| [T5](https://huggingface.co/docs/transformers/main/en/model_doc/t5)\*           | [t5-small](https://huggingface.co/google-t5/t5-small)
| [VITS](https://huggingface.co/docs/transformers/main/en/model_doc/vits)\*       | [mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng)

The following quantizations are supported:

[GPTQ](https://github.com/IST-DASLab/gptq/) (4bit, 8bit, act-order)

## Import a model

You need Python to convert models from Hugging Face. [Install Python](https://www.python.org/downloads/) and run `pip install -r requirements.txt` in `Python` folder to install dependencies, including PyTorch and Transformers. 

The conversion script is located in `Python` folder. For example, if you run `convert.py roneneldan/TinyStories-33M ../Model/TinyStories-33M`, the script will download [TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M) from Hugging Face and generate a folder `Model/TinyStories-33M` which contains JSON configurations and EXR images for model parameters.

In Unity editor, select the generated folder, and click `Assets/ShaderGPT/ImportModel` in the menu. The editor script will reimport the textures and create a MonoBehaviour for running and testing the model. Please refer to the example scene to learn how to set it up.

## Example scene

The demo scene `Example/Main.unity` contains an example of [TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M) inference. A MonoBehaviour creates a list of materials and render textures at runtime to represent the neural network, and execute the network by calling `Graphics.Blit`. The MonoBehaviour can also "bake" the network in editor by saving the materials and render textures as assets.

The other scene `Example/Udon.unity` contains an example of executing the baked network. This is the preferred way of inference because there is no material or texture allocation at runtime. It also runs in Udon if VRCSDK is installed.