# ShaderGPT

GPT inference with HLSL pixel shader and C# script in Unity.

## Supported model

The following architectures are implemented:
* [GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2)
* [GPT-Neo](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neo)
* [GPT-NeoX](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox)

The following models are tested:
* [TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M) (GPT-Neo)
* [gpt-neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125M) (GPT-Neo)
* [pythia-160m](https://huggingface.co/EleutherAI/pythia-160m) (GPT-NeoX)
* [LaMini-GPT-124M](https://huggingface.co/MBZUAI/LaMini-GPT-124M) (GPT2)

## Import a model

You need Python to import a model. The importer script is `Python/import.py`.

For example, if you want to import TinyStories-33M, run `import.py roneneldan/TinyStories-33M ../Model/`. It will generates a folder `Model/TinyStories-33M` with EXR images which encode the model weights.

The images are not correctly imported in Unity by default. Select them in editor and click "ImportAsTensor" in the context menu of texture importer inspector.

Please refer to the example scene to set up a MonoBehaviour for testing.

## Example scene

The demo scene `Example/Main.unity` contains an example of [TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M) inference. A MonoBehaviour creates a list of materials and render textures at runtime to represent the neural network, and execute the network by calling `Graphics.Blit`. The MonoBehaviour can also "bake" the network in editor by saving the materials and render textures as assets.

The other scene `Example/Udon.unity` contains an example of executing the baked network. This is the preferred way of inference because there is no material or texture allocation at runtime. It also runs in Udon if VRCSDK is installed.