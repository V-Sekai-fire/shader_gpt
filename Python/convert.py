import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import math
import re
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, GPTNeoXForCausalLM

def imwrite(path, data):
	data = data[..., [2,1,0,3]] # RGBA to BGRA
	if data.dtype == np.float16 or data.dtype == np.float32:
		cv2.imwrite(str(path), data.astype(np.float32), (
			cv2.IMWRITE_EXR_TYPE,
			cv2.IMWRITE_EXR_TYPE_HALF if data.dtype == np.float16 else cv2.IMWRITE_EXR_TYPE_FLOAT,
			cv2.IMWRITE_EXR_COMPRESSION,
			cv2.IMWRITE_EXR_COMPRESSION_PIZ)) # https://aras-p.info/blog/2021/08/04/EXR-Lossless-Compression/
	elif data.dtype == np.ubyte:
		cv2.imwrite(str(path), data)
	else:
		raise NotImplementedError

def pad_align(data, align):
	return np.pad(data, [(0,(-n)%m) for n, m in zip(data.shape, align)])

def export_custom_int8(data, asym=True, group_size=4, *, estep=2, exact=False):
	presets = [(-127/256, +127/256, 0)] + ([(-63/256, +191/256, +85), (-191/256, +63/256, -85)] if asym else [])
	bmin, bmax, eoff = np.array(presets, dtype=data.dtype).T[..., None, None, None]

	shape1 = (data.shape[1]+3)&~3
	data = pad_align(data, [1, group_size]).reshape(data.shape[0], -1, group_size)

	expo = np.clip(np.ceil(estep*np.log2(np.maximum(
		np.minimum(0, np.amin(data, axis=-1, keepdims=True))/bmin,
		np.maximum(0, np.amax(data, axis=-1, keepdims=True))/bmax))), -42, 42)

	scale = np.exp2(expo/estep, dtype=(np.float32 if exact else data.dtype))
	mant = np.clip(data/scale, bmin, bmax)
	qerr = np.amax(np.abs(np.round(mant*256)/256 * scale - data), axis=-1, keepdims=True)
	best = np.argmin(qerr, axis=0, keepdims=True)
	expo = np.take_along_axis(expo+eoff, best, axis=0)[0]
	mant = np.take_along_axis(mant, best, axis=0)[0]
	mant /= 255/256
	mant += np.where(mant < -1/510, 1, 0)

	mant = mant.reshape(mant.shape[0], -1)[:, :shape1]
	expo = pad_align(expo, [4, 1, 1]).reshape(-1, 4, expo.shape[1]).transpose(0,2,1)
	expo = expo.reshape(expo.shape[0], -1)[:, :shape1]
	return mant, expo.astype(np.int8).astype(np.uint8)

def export_gptq(layer):
	assert layer.group_size % 4 == 0, f"group_size {layer.group_size} should be a multiple of 4"
	assert 8 % layer.bits == 0, f"bits {layer.bits} should be a divisor of 8"
	assert str(type(layer)).find("exllama") < 0, "exllama backend is not supported. please set disable_exllama=True in quantization config"
	wf = torch.tensor(list(range(0, 32, layer.bits)), dtype=torch.int32).unsqueeze(0).to(layer.qweight.device)

	scales = layer.scales
	zeros = torch.bitwise_right_shift(
		torch.unsqueeze(layer.qzeros, 2).expand(-1, -1, 32//layer.bits),
		wf.unsqueeze(0)
	).to(torch.uint8).add(1).bitwise_and(2**layer.bits - 1).reshape(scales.shape) # not sure on wrapping add 1
	weight = torch.bitwise_right_shift(
		torch.unsqueeze(layer.qweight, 1).expand(-1, 32//layer.bits, -1),
		wf.unsqueeze(-1)
	).to(torch.uint8).bitwise_and(2**layer.bits - 1)
	weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

	default_g_idx = torch.arange(layer.g_idx.shape[0], device=layer.g_idx.device) // layer.group_size
	if torch.all(layer.g_idx == default_g_idx):
		indices = None
	else:
		g_idx = layer.g_idx.long()
		g_idx, indices = torch.sort(g_idx)
		assert torch.all(g_idx == default_g_idx), "g_idx is not a permutation"

	mult = 255 // (2**layer.bits - 1)
	weight = weight * mult
	zeros  = zeros * mult
	scales = scales.float() / mult * 256
	scales = scales.view(dtype=torch.int32).bitwise_and(0xFFFFFF00).bitwise_or(zeros).view(dtype=torch.float32)

	if indices is not None:
		weight = weight[indices,:]
		indices = indices.float().cpu().numpy() # use float for int indices
	weight = weight.transpose(0,1).cpu().numpy()
	scales = scales.transpose(0,1).cpu().numpy()
	scales = pad_align(scales, [4, 1]).reshape(-1, 4, scales.shape[1]).transpose(0,2,1)
	scales = scales.reshape(scales.shape[0], -1)
	return weight, scales, indices

def export_lm(model, folder, force_write=False, quantize=None, max_positions=16384):
	os.makedirs(folder, exist_ok=True)
	print(folder/"config.json")
	with open(folder/"config.json", "w") as f:
		json.dump(model.config.to_dict(), f, indent=2, sort_keys=True)

	gptq_layers = {}
	state_dict = dict(model.state_dict())
	for name, layer in model.named_modules():
		# RotaryEmbedding => Linear
		if hasattr(layer, "cos_cached"):
			half_dim = layer.cos_cached.shape[-1]//2
			weight = torch.cat(( # pad rotary weights as complex numbers
				torch.nn.functional.pad(layer.cos_cached[:max_positions, :half_dim], (0, -half_dim%4), value=1),
				torch.nn.functional.pad(layer.sin_cached[:max_positions, :half_dim], (0, -half_dim%4), value=0)), dim=-1)
			name0 = re.sub(r"[.]\d+[.]", ".0.", name, count=1)
			if f"{name0}.weight" in state_dict and torch.allclose(weight, state_dict[f"{name0}.weight"]):
				pass # skip duplicate weights to save space
			else:
				state_dict[f"{name}.weight"] = weight
		# QuantLinear => Linear
		elif hasattr(layer, "qweight"):
			layer.bias, bias = None, layer.bias
			weight = layer(torch.eye(layer.infeatures, dtype=torch.float16, device=model.device)).transpose(1,0)
			layer.bias = bias
			gptq_layers[id(weight)] = layer
			state_dict[f"{name}.weight"] = weight
			for key in ("qweight", "qzeros", "scales", "g_idx"):
				del state_dict[f"{name}.{key}"]

	# model-specific transform
	model_type = model.config.model_type
	if model_type in ["gpt2", "gpt_neo"]:
		for name, data in list(state_dict.items()):
			if name in ["transformer.wte.weight", "transformer.wpe.weight", "lm_head.weight"]:
				if name == "lm_head.weight" and torch.allclose(data, model.state_dict()["transformer.wte.weight"]):
					pass # skip duplicate weights to save space
				else:
					state_dict[f"{name}.T"] = data.T
			elif m := re.fullmatch(r"(.*[.]\d+[.]attn)[.]c_attn[.](weight|bias)", name):
				view = (data.transpose(0,1) if m[2] == "weight" else data).chunk(3, dim=0)
				state_dict[f"{m[1]}.c_query.{m[2]}"] = view[0]
				state_dict[f"{m[1]}.c_key.{  m[2]}"] = view[1]
				state_dict[f"{m[1]}.c_value.{m[2]}"] = view[2]
			else:
				if model_type == "gpt2" and re.search(r"(c_fc|c_proj)[.]weight$", name):
					# Conv1D => Linear
					state_dict[name] = data.transpose(0,1)
				continue
			del state_dict[name]
	elif model_type == "gpt_neox":
		for name, data in list(state_dict.items()):
			if name in ["gpt_neox.embed_in.weight", "embed_out.weight"]:
				state_dict[f"{name}.T"] = data.T
			elif m := re.fullmatch(r"(.*[.]\d+[.]attention)[.]query_key_value[.](weight|bias)", name):
				view = data.view(model.config.num_attention_heads, 3, -1)
				state_dict[f"{m[1]}.query.{m[2]}"] = view[:, 0].reshape(model.config.hidden_size, *data.shape[1:])
				state_dict[f"{m[1]}.key.{  m[2]}"] = view[:, 1].reshape(model.config.hidden_size, *data.shape[1:])
				state_dict[f"{m[1]}.value.{m[2]}"] = view[:, 2].reshape(model.config.hidden_size, *data.shape[1:])
			else:
				continue
			del state_dict[name]
	elif model_type in ["phi", "llama", "mistral", "qwen2"]:
		for name, data in list(state_dict.items()):
			if name in ["model.embed_tokens.weight", "lm_head.weight"]:
				state_dict[f"{name}.T"] = data.T
			else:
				if m := re.fullmatch(r"(.*[.]\d+[.]self_attn)[.]([qkv]_proj[.](weight|bias)|o_proj[.]weight)", name):
					half_dim = getattr(model.get_submodule(m[1]), "head_dim", 0)//2
					if half_dim % 4 != 0: # pad half_dim
						assert model_type != "phi"
						if m[2].startswith("k_proj"): # fix softmax_scale and bake into k_proj
							data = data * (math.sqrt((-half_dim%4+half_dim)*2) / math.sqrt(half_dim*2))
						if m[2].startswith("o_proj"):
							view = data.view(data.shape[0], -1, half_dim)
							view = torch.nn.functional.pad(view, (0, -half_dim%4))
							state_dict[name] = view.view(data.shape[0], -1)
						else:
							view = data.view(-1, half_dim, *data.shape[1:])
							view = torch.nn.functional.pad(view, (*(0,0)*len(data.shape[1:]), 0, -half_dim%4))
							state_dict[name] = view.view(-1, *data.shape[1:])
				continue
			del state_dict[name]
	else:
		raise NotImplementedError(f"{model_type=}")

	for name in list(state_dict.keys()):
		# reduce memory use
		data = state_dict.pop(name)
		print(f"\t{name}\t{tuple(data.shape)}")
		assert (re.search(r"\.weight(\.T)?$", name) and len(data.shape) == 2)\
			or (re.search(r"\.(weight|bias)$", name) and len(data.shape) == 1)

		quantizable = bool(re.search(r"(?<!rotary_emb)\.weight(\.T)?$", name))
		quantizable = quantizable and len(data.shape) == 2
		quantizable = quantizable and quantize is not None and quantize(name, data.shape)

		filenames = []
		if id(data) in gptq_layers:
			filenames = (f"{name}.png", f"{name}.q8.exr", f"{name}.q8.idx.exr")
		elif quantizable:
			filenames = (f"{name}.exr", f"{name}.q8.png")
		else:
			filenames = (f"{name}.exr",)

		possible_filenames = [f"{name}.exr", f"{name}.png", f"{name}.q8.exr", f"{name}.q8.png"]
		if not force_write and all((folder/x).exists() == (x in filenames) for x in possible_filenames):
			continue
		possible_filenames.append(f"{name}.q8.idx.exr")
		for x in possible_filenames:
			(folder/x).unlink(missing_ok=True)

		if id(data) in gptq_layers:
			arrays = export_gptq(gptq_layers[id(data)])
		elif quantizable:
			arrays = export_custom_int8(data.cpu().numpy())
		else:
			arrays = (data.cpu().numpy(),)
		for filename, array in zip(filenames, arrays):
			if array is None:
				continue
			print(f"\t\t{filename}\t{tuple(array.shape)}")
			if len(array.shape) == 2:
				array = pad_align(array, [1, 4]).reshape(array.shape[0], -1, 4)
			elif len(array.shape) == 1:
				array = array.reshape(1, -1, 4)
			else:
				raise KeyError(f"unexpected {array.shape}")

			# tile wide texture
			MAX_SIZE = 16384
			if array.shape[1] > MAX_SIZE:
				lvl = 0
				while ((array.shape[1]-1)>>lvl)+1 > MAX_SIZE:
					lvl += 1
				array = pad_align(array, [1, 1<<lvl, 1]).reshape(array.shape[0], -1, 1<<lvl, array.shape[2])\
					.transpose(0, 2, 1, 3).reshape(array.shape[0]<<lvl, -1, array.shape[2])

			array = array[::-1] # flip Y for d3d
			imwrite(folder/filename, array)

def export_tokenizer(tokenizer, folder):
	if tokenizer.is_fast:
		fast_tokenizer = tokenizer.backend_tokenizer
	else:
		from transformers.convert_slow_tokenizer import convert_slow_tokenizer
		fast_tokenizer = convert_slow_tokenizer(tokenizer)
	model = fast_tokenizer.model
	pre_tokenizer = fast_tokenizer.pre_tokenizer

	if type(model).__name__ == "BPE":
		data = json.loads(fast_tokenizer.to_str())
		merges = [x.split(" ", 1) for x in data["model"]["merges"]]
		if pre_tokenizer is None:
			prefix_len = len(tokenizer.decode([0]))
			decode_token = lambda token: bytes((int(token[1:-1], 16),)) \
				if model.byte_fallback and re.fullmatch(r"<0x[0-9A-F]{2}>", token) \
				else tokenizer.decode([0, tokenizer.convert_tokens_to_ids(token)])[prefix_len:].encode()
		elif type(pre_tokenizer).__name__ == "ByteLevel" or type(fast_tokenizer.decoder).__name__ == "ByteLevel":
			from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
			byte_decoder = {c: b for b, c in bytes_to_unicode().items()}
			decode_token = lambda token: bytes(byte_decoder[c] for c in token)
		else:
			raise NotImplementedError(f"pre_tokenizer={type(pre_tokenizer)}, decoder={type(fast_tokenizer.decoder)} is not supported")
	else:
		raise NotImplementedError(f"model {type(model)} is not supported")

	vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer.get_vocab()))]
	merges = [tokenizer.convert_tokens_to_ids(x) for x in merges]
	added_tokens = [k for k, v in sorted(tokenizer.added_tokens_encoder.items(), key=lambda item: item[1])]

	added_tokens_set = set(added_tokens)
	vocab = [token if token in added_tokens_set else "".join(chr(b) for b in decode_token(token)) for token in vocab]
	merges = [f"{vocab[i]} {vocab[j]}" for i, j in merges]

	chat_templates = {}
	messages = [
		{"role": "system", "content": "{0}"},
		{"role": "user", "content": "{0}"},
		{"role": "assistant", "content": "{0}"},
	]
	last_text = ""
	for i in range(3):
		text = tokenizer.apply_chat_template(messages[:i+1], tokenize=False)
		chat_templates[messages[i]["role"]] = text[len(last_text):]
		last_text = text

	# workaround for broken json parser
	escape_brackets = lambda lst: [re.sub(r'[\[\]\{\}]', lambda m: f"\\u{ord(m.group()) :04X}", x) for x in lst]
	unescape_brackets = lambda x: x.replace(r"\\u00", r"\u00")
	output = unescape_brackets(json.dumps(dict(
		vocab  = escape_brackets(vocab),
		merges = escape_brackets(merges),
		bos_token_id = tokenizer.bos_token_id,
		eos_token_id = tokenizer.eos_token_id,
		added_tokens = added_tokens,
		chat_templates = chat_templates,
	), ensure_ascii=True, indent=2))

	os.makedirs(folder, exist_ok=True)
	print(folder/"tokenizer.json")
	with open(folder/"tokenizer.json", "w", encoding="utf-8") as f:
		f.write(output)

@torch.no_grad()
def export_testcase(model, tokenizer, folder, force_write=False):
	if not force_write and (folder/"testcase.json").exists():
		return

	prompt = (
		"In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
		"previously unexplored valley, in the Andes Mountains. Even more surprising to the "
		"researchers was the fact that the unicorns spoke perfect English."
	)
	input_ids = tokenizer(prompt, return_tensors="pt", padding=False, add_special_tokens=False).input_ids.to(model.device)
	position_ids = torch.ones_like(input_ids, device=input_ids.device).long().cumsum(-1) - 1
	outputs = model(input_ids=input_ids, position_ids=position_ids, output_hidden_states=True, return_dict=True)

	os.makedirs(folder, exist_ok=True)
	print(folder/"testcase.json")
	with open(folder/"testcase.json", "w") as f:
		json.dump(dict(
			input_ids = input_ids[0].tolist(),
			hidden_states = outputs.hidden_states[-1][0,-1].tolist(),
			logits = outputs.logits[0,-1].tolist(),
		), f)

def main():
	from transformers import AutoModelForCausalLM, AutoTokenizer
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('model', help='model id or path. for example: roneneldan/TinyStories-1M')
	parser.add_argument('folder', help='save path. for example: ../Model/')
	parser.add_argument('--force', action='store_true')
	parser.add_argument('--quantize', type=float)
	parser.add_argument('--device', type=str)
	args = parser.parse_args()

	folder = Path(args.folder)
	if re.search(r"[/\\]$", args.folder):
		folder /= Path(args.model).name
	print(f"convert: {args.model} => {folder}")
	model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device or None)
	tokenizer = AutoTokenizer.from_pretrained(args.model)
	quantize = (lambda name, shape: np.prod(shape) >= args.quantize*1024*1024) if args.quantize else None
	print(f"model: {type(model)}")
	export_lm(model, folder, force_write=bool(args.force), quantize=quantize)
	export_tokenizer(tokenizer, folder)
	export_testcase(model, tokenizer, folder, force_write=bool(args.force))

if __name__ == '__main__':
	main()