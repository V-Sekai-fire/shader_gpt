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

def pad_align(data, align):
	return np.pad(data, [(0,(-n)%m) for n, m in zip(data.shape, align)])

def quantize_unorm8(data, asym=True, bits=8, group=4, *, dstep=256, estep=2):
	presets = [(-127, +127, 0)] + ([(-63, +191, +85), (-191, +63, -85)] if asym else [])
	bmin, bmax, eoff = np.array(presets, dtype=data.dtype).T[..., None, None, None]
	gdata = pad_align(data.reshape(data.shape[0], -1), [1, group]).reshape(data.shape[0], -1, group)

	dmin = np.minimum(0, np.amin(gdata, axis=-1, keepdims=True))
	dmax = np.maximum(0, np.amax(gdata, axis=-1, keepdims=True))
	expo = np.clip(np.ceil(np.log2(np.maximum(dmin/(bmin/dstep), dmax/(bmax/dstep)))*estep), -42, 42)
	scale = 255/dstep * np.exp2(expo/estep)
	mant = np.clip(gdata/scale, bmin/255, bmax/255).astype(gdata.dtype)
	qerr = np.amax(np.abs(np.round(mant*(2**bits-1))/(2**bits-1) * scale - gdata), axis=-1, keepdims=True)
	best = np.argmin(qerr, axis=0, keepdims=True)
	expo = np.take_along_axis(expo+eoff, best, axis=0)[0]
	mant = np.take_along_axis(mant, best, axis=0)[0]

	mant = np.where(mant < -1/512, 1+mant, mant) # keep tiny float
	mant = mant.reshape(mant.shape[0], -1, data.shape[2])[:, :data.shape[1]]
	expo = pad_align(expo, [4, 1, 1]).reshape(-1, 4, expo.shape[1]).transpose(0,2,1)
	return mant, expo.astype(np.int8).astype(np.uint8)

def export_lm(model, folder, force_write=False, quantize=None, max_positions=4096):
	os.makedirs(folder, exist_ok=True)
	print(folder/"config.json")
	with open(folder/"config.json", "w") as f:
		json.dump(model.config.to_dict(), f, indent=2, sort_keys=True)

	state_dict = dict(model.state_dict())
	for name, layer in model.named_modules():
		# RotaryEmbedding => Linear
		if hasattr(layer, "cos_cached"):
			weight = torch.cat((
				layer.cos_cached[..., :layer.cos_cached.shape[-1]//2],
				layer.sin_cached[..., :layer.sin_cached.shape[-1]//2]), dim=-1)
			state_dict[f"{name}.weight"] = weight[:max_positions]
		# QuantLinear => Linear
		elif hasattr(layer, "qweight"):
			layer.bias, bias = None, layer.bias
			weight = layer(torch.eye(layer.infeatures, dtype=torch.float16, device=model.device)).transpose(1,0)
			layer.bias = bias
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
	elif model_type in ["phi", "llama", "qwen2"]:
		for name, data in list(state_dict.items()):
			if name in ["model.embed_tokens.weight", "lm_head.weight"]:
				state_dict[f"{name}.T"] = data.T
			else:
				continue
			del state_dict[name]
	else:
		raise NotImplementedError(f"{model_type=}")

	for name in list(state_dict.keys()):
		# reduce memory use
		data = state_dict.pop(name).cpu().numpy()
		print("\t", name, data.shape, data.dtype)
		if re.search(r"\.weight(\.T)?$", name) and len(data.shape) == 2:
			data = pad_align(data, [1, 4]).reshape(data.shape[0], -1, 4)
		elif re.search(r"\.(weight|bias)$", name) and len(data.shape) == 1:
			data = data.reshape(1, -1, 4)
		else:
			raise KeyError(f"unexpected {name} : {data.shape}")

		# wrap wide texture
		MAX_SIZE = 16384
		wrap1 = 1+(data.shape[1]-1) // MAX_SIZE
		if wrap1 > 1:
			data = pad_align(data, (1, MAX_SIZE, 1))
			data = data.reshape(-1, MAX_SIZE, data.shape[2])
		assert data.shape[0] <= MAX_SIZE and data.shape[1] <= MAX_SIZE

		quantizable = data.shape[0] % 4 == 0 and bool(re.search(r"(?<!rotary_emb)\.weight(\.T)?$", name))
		if data.dtype == np.uint8:
			if not force_write and (folder/f"{name}.png").exists():
				continue

			imwrite(folder/f"{name}.png", data[::-1])
		elif quantizable and quantize is not None and quantize(name, data.shape):
			if not force_write and (folder/f"{name}.exr").exists() and (folder/f"{name}.q8.png").exists():
				continue
			(folder/f"{name}.exr").unlink(missing_ok=True)
			(folder/f"{name}.q8.png").unlink(missing_ok=True)

			# NOTE: quantize unwrapped
			if wrap1 > 1:
				data = data.reshape(-1, wrap1*MAX_SIZE, data.shape[2])
			data, expo = quantize_unorm8(data)
			if wrap1 > 1:
				data = data.reshape(-1, MAX_SIZE, data.shape[2])
				expo = expo.reshape(-1, MAX_SIZE, expo.shape[2])
			imwrite(folder/f"{name}.exr", data[::-1])
			imwrite(folder/f"{name}.q8.png", expo[::-1])
		else:
			if not force_write and (folder/f"{name}.exr").exists() and not (folder/f"{name}.q8.png").exists():
				continue
			(folder/f"{name}.exr").unlink(missing_ok=True)
			(folder/f"{name}.q8.png").unlink(missing_ok=True)

			imwrite(folder/f"{name}.exr", data[::-1])

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

	# workaround for broken json parser
	escape_brackets = lambda lst: [re.sub(r'[\[\]\{\}]', lambda m: f"\\u{ord(m.group()) :04X}", x) for x in lst]
	unescape_brackets = lambda x: x.replace(r"\\u00", r"\u00")
	output = unescape_brackets(json.dumps(dict(
		vocab  = escape_brackets(vocab),
		merges = escape_brackets(merges),
		eos_token_id = tokenizer.eos_token_id,
		added_tokens = added_tokens,
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