import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import re
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, GPTNeoXForCausalLM

def export_lm(model, folder, force_write=False):
	os.makedirs(folder, exist_ok=True)
	print(folder/"config.json")
	with open(folder/"config.json", "w") as f:
		json.dump(model.config.to_dict(), f, indent=2, sort_keys=True)

	state_dict = dict(model.state_dict())
	if isinstance(model, GPTNeoForCausalLM):
		for name, data in list(state_dict.items()):
			if name in ["transformer.wte.weight", "transformer.wpe.weight"]:
				state_dict[f"{name}.T"] = data.T
			elif name == "lm_head.weight":
				pass
			else:
				continue
			del state_dict[name]
	elif isinstance(model, GPT2LMHeadModel):
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
	elif isinstance(model, GPTNeoXForCausalLM):
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
	else:
		raise NotImplementedError(f"{type(model)=}")

	for name, data in state_dict.items():
		data = data.numpy()
		print("\t", name, data.shape)
		if re.search(r"\.weight(\.T)?$", name) and len(data.shape) == 2:
			data = np.pad(data, [(0,0), (0,(-data.shape[1])&3)])
			data = data.reshape(data.shape[0], data.shape[1]//4, 4)
		elif re.search(r"\.(weight|bias)$", name) and len(data.shape) == 1:
			data = data.reshape(1, data.shape[0]//4, 4)
		else:
			raise KeyError(f"unexpected {name} : {data.shape}")

		data = data[::-1]
		data = data[..., [2,1,0,3]] # RGBA to BGRA
		if data.dtype == np.float16 or data.dtype == np.float32:
			if force_write or not (folder/f"{name}.exr").exists():
				cv2.imwrite(str(folder/f"{name}.exr"), data.astype(np.float32),
					(cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF if data.dtype == np.float16 else cv2.IMWRITE_EXR_TYPE_FLOAT))
		elif data.dtype == np.ubyte:
			if force_write or not (folder/f"{name}.png").exists():
				cv2.imwrite(str(folder/f"{name}.png"), data)
		else:
			raise NotImplementedError(f"{data.dtype=}")

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

def export_testcase(model, tokenizer, folder, force_write=False):
	if not force_write and (folder/"testcase.json").exists():
		return

	prompt = (
		"In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
		"previously unexplored valley, in the Andes Mountains. Even more surprising to the "
		"researchers was the fact that the unicorns spoke perfect English."
	)
	input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cpu()
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
	args = parser.parse_args()

	folder = Path(args.folder)
	if re.search(r"[/\\]$", args.folder):
		folder /= Path(args.model).name
	print(f"import model from {args.model} to {folder}")
	model = AutoModelForCausalLM.from_pretrained(args.model)
	tokenizer = AutoTokenizer.from_pretrained(args.model)
	export_lm(model, folder, force_write=bool(args.force))
	export_tokenizer(tokenizer, folder)
	export_testcase(model, tokenizer, folder, force_write=bool(args.force))

if __name__ == '__main__':
	main()