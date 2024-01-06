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

def export_lm(model, folder, force_write=False, quantize=None):
	os.makedirs(folder, exist_ok=True)
	print(folder/"config.json")
	with open(folder/"config.json", "w") as f:
		json.dump(model.config.to_dict(), f, indent=2, sort_keys=True)

	model_type = model.config.model_type
	state_dict = dict(model.state_dict())
	if model_type == "gpt_neo":
		for name, data in list(state_dict.items()):
			if name in ["transformer.wte.weight", "transformer.wpe.weight"]:
				state_dict[f"{name}.T"] = data.T
			elif name == "lm_head.weight":
				pass
			else:
				continue
			del state_dict[name]
	elif model_type == "gpt2":
		for name, data in list(state_dict.items()):
			if name in ["transformer.wte.weight", "transformer.wpe.weight"]:
				state_dict[f"{name}.T"] = data.T
			elif name == "lm_head.weight":
				pass
			elif m := re.fullmatch(r"transformer[.]h[.](\d+)[.]attn[.]c_(attn[.](weight|bias))", name):
				prefix = name[:-len(m[2])]
				norm_factor = 1 / (model.transformer.h[int(m[1])].attn.head_dim ** 0.5)
				data = (data.transpose(0,1) if m[3] == "weight" else data).chunk(3, dim=0)
				if m[3] == "weight":
					state_dict[f"{prefix}query.weight"] = data[0] * norm_factor
					state_dict[f"{prefix  }key.weight"] = data[1]
					state_dict[f"{prefix}value.weight"] = data[2]
				else:
					state_dict[f"{prefix}query.bias"] = data[0] * norm_factor
					state_dict[f"{prefix  }key.bias"] = data[1]
					state_dict[f"{prefix}value.bias"] = data[2]
			else:
				if m := re.search(r"(c_fc|c_proj)[.]weight$", name):
					state_dict[name] = data.transpose(0,1)
				continue
			del state_dict[name]
	elif model_type == "gpt_neox":
		for name, data in list(state_dict.items()):
			if name in ["gpt_neox.embed_in.weight", "embed_out.weight"]:
				state_dict[f"{name}.T"] = data.T
			elif m := re.fullmatch(r"gpt_neox[.]layers[.](\d+)[.]attention([.]query_key_value[.](weight|bias))", name):
				prefix = name[:-len(m[2])]
				norm_factor = model.gpt_neox.layers[int(m[1])].attention.norm_factor
				data = data.view(model.config.num_attention_heads, 3, -1)
				if m[3] == "weight":
					state_dict[f"{prefix}.query.weight"] = data[:, 0].reshape(model.config.hidden_size, -1) * norm_factor
					state_dict[f"{prefix  }.key.weight"] = data[:, 1].reshape(model.config.hidden_size, -1)
					state_dict[f"{prefix}.value.weight"] = data[:, 2].reshape(model.config.hidden_size, -1)
				else:
					state_dict[f"{prefix}.query.bias"] = data[:, 0].reshape(model.config.hidden_size) * norm_factor
					state_dict[f"{prefix  }.key.bias"] = data[:, 1].reshape(model.config.hidden_size)
					state_dict[f"{prefix}.value.bias"] = data[:, 2].reshape(model.config.hidden_size)
			elif m := re.fullmatch(r"gpt_neox[.]layers[.](\d+)[.]attention[.]rotary_emb([.]inv_freq)", name):
				prefix = name[:-len(m[2])]
				rotary_emb = model.gpt_neox.layers[int(m[1])].attention.rotary_emb
				weight = torch.cat((
					rotary_emb.cos_cached[..., :rotary_emb.cos_cached.shape[-1]//2],
					rotary_emb.sin_cached[..., :rotary_emb.sin_cached.shape[-1]//2]), dim=-1)
				state_dict[f"{prefix}.weight"] = weight[0,0]
			else:
				continue
			del state_dict[name]
	elif model_type == "phi-msft":
		for name, data in list(state_dict.items()):
			if name in ["transformer.embd.wte.weight", "lm_head.linear.weight"]:
				state_dict[f"{name}.T"] = data.T
			elif m := re.fullmatch(r"transformer[.]h[.](\d+)[.]mixer([.]Wqkv[.](weight|bias))", name):
				prefix = name[:-len(m[2])]
				mixer = model.transformer.h[int(m[1])].mixer
				norm_factor = mixer.inner_cross_attn.softmax_scale or 1.0 / math.sqrt(mixer.head_dim)
				data = data.view(3, -1)
				if m[3] == "weight":
					state_dict[f"{prefix}.Wq.weight"] = data[0].reshape(model.config.n_embd, -1)
					state_dict[f"{prefix}.Wk.weight"] = data[1].reshape(model.config.n_embd, -1) * norm_factor
					state_dict[f"{prefix}.Wv.weight"] = data[2].reshape(model.config.n_embd, -1)
				else:
					state_dict[f"{prefix}.Wq.bias"] = data[0].reshape(model.config.n_embd)
					state_dict[f"{prefix}.Wk.bias"] = data[1].reshape(model.config.n_embd) * norm_factor
					state_dict[f"{prefix}.Wv.bias"] = data[2].reshape(model.config.n_embd)

				rotary_emb = mixer.rotary_emb
				weight = torch.cat((
					rotary_emb._cos_cached[..., :mixer.rotary_dim//2],
					rotary_emb._sin_cached[..., :mixer.rotary_dim//2]), dim=-1)
				state_dict[f"{prefix}.rotary_emb.weight"] = weight[..., :, :]
			else:
				continue
			del state_dict[name]
	else:
		raise NotImplementedError(f"{model_type=}")

	for name, data in state_dict.items():
		data = data.cpu().numpy()
		print("\t", name, data.shape)
		if re.search(r"\.weight(\.T)?$", name) and len(data.shape) == 2:
			data = pad_align(data, [1, 4]).reshape(data.shape[0], -1, 4)
		elif re.search(r"\.(weight|bias)$", name) and len(data.shape) == 1:
			data = data.reshape(1, -1, 4)
		else:
			raise KeyError(f"unexpected {name} : {data.shape}")

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

			data, expo = quantize_unorm8(data)
			imwrite(folder/f"{name}.exr", data[::-1])
			imwrite(folder/f"{name}.q8.png", expo[::-1])
		else:
			if not force_write and (folder/f"{name}.exr").exists() and not (folder/f"{name}.q8.png").exists():
				continue
			(folder/f"{name}.exr").unlink(missing_ok=True)
			(folder/f"{name}.q8.png").unlink(missing_ok=True)

			imwrite(folder/f"{name}.exr", data[::-1])

def export_tokenizer(tokenizer, folder):
	byte_decoder = None
	token_to_chrs = lambda token: "".join(chr(byte_decoder.get(ch) or ord(ch)) for ch in token)
	if tokenizer.is_fast:
		from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
		byte_decoder = {ch: b for b, ch in bytes_to_unicode().items()}
		data = json.loads(tokenizer.backend_tokenizer.to_str())
		merges = [" ".join(token_to_chrs(token) for token in merge.split()) for merge in data["model"]["merges"]]
	else:
		byte_decoder = tokenizer.byte_decoder
		merges = [" ".join(token_to_chrs(token) for token in pair) for pair in tokenizer.bpe_ranks.keys()]
	
	vocab = [None] * len(tokenizer.get_vocab())
	for token, id in tokenizer.get_vocab().items():
		vocab[id] = token_to_chrs(token)

	# workaround for broken json parser
	escape_brackets = lambda lst: [re.sub(r'[\[\]\{\}]', lambda m: f"\\u{ord(m.group()) :04X}", x) for x in lst]
	unescape_brackets = lambda x: x.replace(r"\\u00", r"\u00")
	output = unescape_brackets(json.dumps(dict(
		vocab  = escape_brackets(vocab),
		merges = escape_brackets(merges),
		eos_token_id = tokenizer.eos_token_id,
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
	input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
	if model.config.model_type == "phi-msft":
		attention_mask = torch.ones_like(input_ids, device=input_ids.device).long()
		hidden_states = model.transformer(**model.prepare_inputs_for_generation(input_ids, None, attention_mask))
		lm_logits = model.lm_head(hidden_states)
		from transformers.modeling_outputs import CausalLMOutputWithPast
		outputs = CausalLMOutputWithPast(logits=lm_logits, hidden_states=[hidden_states])
	else:
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
	args = parser.parse_args()

	folder = Path(args.folder)
	if re.search(r"[/\\]$", args.folder):
		folder /= Path(args.model).name
	print(f"convert: {args.model} => {folder}")
	model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
	tokenizer = AutoTokenizer.from_pretrained(args.model)
	quantize = (lambda name, shape: np.prod(shape) >= args.quantize*1024*1024) if args.quantize else None
	export_lm(model, folder, force_write=bool(args.force), quantize=quantize)
	export_tokenizer(tokenizer, folder)
	export_testcase(model, tokenizer, folder, force_write=bool(args.force))

if __name__ == '__main__':
	main()