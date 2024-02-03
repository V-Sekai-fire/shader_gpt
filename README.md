# ShaderGPT

GPT inference with HLSL pixel shader and C# script in Unity.

## Supported model

The following architectures are implemented:
* [GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2)
* [GPT-Neo](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neo)
* [GPT-NeoX](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox)
* [Phi](https://huggingface.co/docs/transformers/main/en/model_doc/phi)
* [LLaMA](https://huggingface.co/docs/transformers/main/en/model_doc/llama)

The following models are tested:
* [TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M) (GPT-Neo)
* [gpt-neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125M) (GPT-Neo)
* [pythia-160m](https://huggingface.co/EleutherAI/pythia-160m) (GPT-NeoX)
* [LaMini-GPT-124M](https://huggingface.co/MBZUAI/LaMini-GPT-124M) (GPT2)
* [phi-1_5](https://huggingface.co/microsoft/phi-1_5) (Phi)
* [TinyLlama-1.1B-orca-v1.0](https://huggingface.co/sreeramajay/TinyLlama-1.1B-orca-v1.0) (LLaMA)
* [zyte-1B](https://huggingface.co/aihub-app/zyte-1B) (LLaMA)

## Import a model

You need Python to convert models from Huggingface. [Install Python](https://www.python.org/downloads/) and run `pip install -r requirements.txt` in `Python` folder to install dependencies.

The conversion script is `convert.py` in `Python` folder. For example, if you want to use TinyStories-33M, run `convert.py roneneldan/TinyStories-33M ../Model/`. The script will generate a folder `Model/TinyStories-33M` which contains JSON configurations and EXR images for model parameters.

In Unity editor, select the generated folder, and click "Assets/ShaderGPT/ImportModel" in the menu. The editor script will reimport the textures and create a MonoBehaviour for running and testing the model. Please refer to the example scene to learn how to set it up.

## Example scene

The demo scene `Example/Main.unity` contains an example of [TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M) inference. A MonoBehaviour creates a list of materials and render textures at runtime to represent the neural network, and execute the network by calling `Graphics.Blit`. The MonoBehaviour can also "bake" the network in editor by saving the materials and render textures as assets.

The other scene `Example/Udon.unity` contains an example of executing the baked network. This is the preferred way of inference because there is no material or texture allocation at runtime. It also runs in Udon if VRCSDK is installed.