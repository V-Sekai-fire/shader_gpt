import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import math
import re
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path

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

def export_custom_int8(data, asym=True, group_size=4, *, estep=2, exact=False, act_order=True):
	data = F.pad(data, (0,-data.shape[-1] % group_size)) # must pad before sorting
	indices = None
	if act_order:
		indices = torch.argsort(torch.linalg.norm(data, ord=1, dim=0), stable=True, descending=True)
		data = data[:,indices]
		indices = torch.stack((indices, torch.argsort(indices))).float().cpu().numpy() # use float for int indices
	data = data.reshape(data.shape[0], -1, group_size)

	presets = [(-127/256, +127/256, 0)] + ([(-63/256, +191/256, +85), (-191/256, +63/256, -85)] if asym else [])
	bmin, bmax, eoff = torch.tensor(presets, dtype=data.dtype, device=data.device).T[..., None, None, None]
	expo = torch.clip(torch.ceil(estep*torch.log2(torch.maximum(
		torch.clip(torch.amin(data, dim=-1, keepdims=True), max=0)/bmin,
		torch.clip(torch.amax(data, dim=-1, keepdims=True), min=0)/bmax))), -42, 42)

	scale = torch.exp2((expo/estep).to(torch.float32 if exact else data.dtype))
	mant = torch.clip(data/scale, bmin, bmax)
	qerr = torch.amax(torch.abs(torch.round(mant*256)/256 * scale - data), dim=-1, keepdims=True)
	best = torch.argmin(qerr, dim=0, keepdims=True)
	expo = torch.take_along_dim(expo+eoff, best, dim=0)[0]
	mant = torch.take_along_dim(mant, best, dim=0)[0]
	mant /= 255/256
	mant += torch.where(mant < -1/510, 1, 0)

	mant = mant.reshape(mant.shape[0], -1).cpu().numpy()
	expo = F.pad(expo, (0,0,0,0,0,-expo.shape[-3] % 4)).reshape(-1, 4, expo.shape[1]).permute(0,2,1)
	expo = expo.reshape(expo.shape[0], -1).cpu().numpy()
	return mant, expo.astype(np.int8).astype(np.uint8), indices

def disable_exllama():
	from transformers import GPTQConfig
	post_init = GPTQConfig.post_init
	GPTQConfig.post_init = lambda self: setattr(self, "use_exllama", False) or post_init(self)

def unpack_gptq(layer):
	bias = layer.bias
	layer.bias = None
	weight = layer(torch.eye(layer.infeatures, dtype=layer.scales.dtype, device=layer.scales.device)).transpose(1,0)
	layer.bias = bias
	return weight

def export_gptq(layer):
	assert layer.group_size % 4 == 0, f"group_size {layer.group_size} should be a multiple of 4"
	assert 8 % layer.bits == 0, f"bits {layer.bits} should be a divisor of 8"
	assert str(type(layer)).find("exllama") < 0, "exllama backend is not supported"
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
		indices = torch.stack((indices, torch.argsort(indices))).float().cpu().numpy() # use float for int indices
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

	# apply weight parametrizations (in vits)
	for name, layer in model.named_modules():
		parametrizations = getattr(layer, "parametrizations", None)
		if isinstance(parametrizations, torch.nn.ModuleDict):
			for key in list(parametrizations):
				torch.nn.utils.parametrize.remove_parametrizations(layer, key)

	gptq_layers = {}
	unpack = lambda x: unpack_gptq(gptq_layers[id(x)]) if id(x) in gptq_layers else x

	state_dict = dict(model.state_dict())
	for name, layer in model.named_modules():
		# RotaryEmbedding => Linear
		if hasattr(layer, "inv_freq"):
			if hasattr(layer, "cos_cached"): # deprecated in torch 4.39
				cos_cached, sin_cached = layer.cos_cached, layer.sin_cached
			elif hasattr(layer, "_cached_cos"): # TODO: OpenELMRotaryEmbedding
				cos_cached, sin_cached = layer._cached_cos[0,0], layer._cached_sin[0,0]
			else:
				pos_id = torch.arange(min(model.config.max_position_embeddings, max_positions),
					dtype=torch.float32, device=model.device).unsqueeze(0)
				cos_cached, sin_cached = (x[0] for x in layer(pos_id, pos_id))
			half_dim = cos_cached.shape[-1]//2
			weight = torch.cat(( # pad rotary weights as complex numbers
				torch.nn.functional.pad(cos_cached[:max_positions, :half_dim], (0, -half_dim%2), value=1),
				torch.nn.functional.pad(sin_cached[:max_positions, :half_dim], (0, -half_dim%2), value=0)), dim=-1)
			name0 = re.sub(r"[.]\d+[.]", ".0.", name, count=1)
			if f"{name0}.weight" in state_dict and torch.allclose(weight, state_dict[f"{name0}.weight"]):
				pass # skip duplicate weights to save space
			else:
				state_dict[f"{name}.weight"] = weight
		# QuantLinear => Linear
		elif hasattr(layer, "qweight"):
			weight = torch.zeros((layer.outfeatures, layer.infeatures), dtype=layer.scales.dtype, device="meta")
			gptq_layers[id(weight)] = layer
			state_dict[f"{name}.weight"] = weight
			for key in ("qweight", "qzeros", "scales", "g_idx"):
				del state_dict[f"{name}.{key}"]

	# model-specific transform
	model_type = model.config.model_type
	if model_type in ["gpt2", "gpt_neo"]:
		for name, data in list(state_dict.items()):
			if name in ["transformer.wte.weight", "lm_head.weight"]:
				if name == "lm_head.weight" and torch.allclose(data, model.state_dict()["transformer.wte.weight"]):
					pass # skip duplicate weights to save space
				else:
					state_dict[f"{name}.T"] = unpack(data).T
			else:
				if model_type == "gpt2" and re.search(r"(c_fc|c_proj|c_attn)[.]weight$", name):
					# Conv1D => Linear
					state_dict[name] = unpack(data).transpose(0,1)
				continue
			del state_dict[name]
	elif model_type == "gpt_neox":
		for name, data in list(state_dict.items()):
			if name in ["gpt_neox.embed_in.weight", "embed_out.weight"]:
				state_dict[f"{name}.T"] = unpack(data).T
			else:
				if m := re.fullmatch(r"(.*[.]\d+[.]attention)[.]query_key_value[.](weight|bias)", name):
					state_dict[name] = unpack(data).view(model.config.num_attention_heads, 3, -1).\
						transpose(0,1).reshape_as(data)
				continue
			del state_dict[name]
	elif model_type in ["gemma", "llama", "mistral", "phi", "phi3", "qwen2", "stablelm"]:
		for name, data in list(state_dict.items()):
			if name in ["model.embed_tokens.weight", "lm_head.weight"]:
				if name == "lm_head.weight" and torch.allclose(data, model.state_dict()["model.embed_tokens.weight"]):
					pass # skip duplicate weights to save space
				else:
					state_dict[f"{name}.T"] = unpack(data).T
			else:
				if m := re.fullmatch(r"(.*[.]\d+[.]self_attn)[.]([qkv]_proj[.](weight|bias)|o_proj[.]weight)", name):
					head_dim = getattr(model.get_submodule(m[1]), "head_dim", 0)
					if head_dim%4 != 0: # pad head_dim and half rotary dim
						half_dim = model.get_submodule(m[1]).rotary_emb.dim//2
						view = unpack(data).view(data.shape[0], -1, head_dim, 1) if m[2].startswith("o_proj")\
							else unpack(data).view(1, -1, head_dim, data.numel() // data.shape[0])
						view = torch.cat((
							torch.nn.functional.pad(view[:, :, 0*half_dim:1*half_dim], (0,0, 0,-half_dim%2)),
							torch.nn.functional.pad(view[:, :, 1*half_dim:2*half_dim], (0,0, 0,-half_dim%2)),
							torch.nn.functional.pad(view[:, :, 2*half_dim:], (0,0, 0,-(head_dim-2*half_dim)%4)),
						), dim=2)
						state_dict[name] = view.view(data.shape[0], -1) if m[2].startswith("o_proj")\
										else view.view(-1, *data.shape[1:])
				elif m := re.fullmatch(r".*(_layernorm|\bnorm)[.]weight", name):
					if model_type == "gemma":
						state_dict[name] = 1.0 + unpack(data).float() # convert residue weight
				continue
			del state_dict[name]
	elif model_type in ["openelm"]:
		for name, data in list(state_dict.items()):
			if name in ["transformer.token_embeddings.weight", "lm_head.weight"]:
				state_dict[f"{name}.T"] = unpack(data).T
			else:
				continue
			del state_dict[name]
	elif model_type in ["t5"]:
		shared_weight = model.state_dict().get("shared.weight")
		for name, data in list(state_dict.items()):
			if name in ["shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]:
				if name != "shared.weight" and shared_weight is not None and torch.allclose(data, shared_weight):
					pass # skip duplicate weights to save space
				else:
					state_dict[f"{name}.T"] = unpack(data).T
			elif m := re.fullmatch(r"(.*[.]SelfAttention)[.]relative_attention_bias[.]weight", name):
				layer = model.get_submodule(m[1])
				relative_position_bucket = layer._relative_position_bucket(
					torch.arange(-layer.relative_attention_max_distance, layer.relative_attention_max_distance,
						dtype=torch.long, device=layer.q.weight.device),
					bidirectional=(not layer.is_decoder),
					num_buckets=layer.relative_attention_num_buckets,
					max_distance=layer.relative_attention_max_distance,
				)
				# bake buckets into weights
				state_dict[f"{name}.T"] = layer.relative_attention_bias(relative_position_bucket).permute(1, 0)[None,:,:]
			else:
				continue
			del state_dict[name]
	elif model_type in ["vits"]:
		for name, data in list(state_dict.items()):
			if name in ["text_encoder.embed_tokens.weight", "text_encoder.project.weight"]:
				state_dict[f"{name}.T"] = unpack(data).transpose(0,1)
			elif m := re.fullmatch(r".*attention[.]emb_rel_[kv]", name):
				state_dict[f"{m[0]}.weight"] = unpack(data).squeeze(0)
			elif re.fullmatch(r"(text_encoder|flow|decoder).*", name):
				continue
			del state_dict[name]
	else:
		raise NotImplementedError(f"{model_type=}")

	for name in list(state_dict.keys()):
		# reduce memory use
		data = state_dict.pop(name)
		print(f"\t{name}\t{tuple(data.shape)} {data.dtype}")
		assert (re.search(r"\.weight(\.T)?$", name) and len(data.shape) in (2,3))\
			or (re.search(r"\.(weight|bias)$", name) and len(data.shape) == 1)

		quantizable = bool(re.search(r"\.weight(\.T)?$", name)) and len(data.shape) == 2
		quantizable = quantizable and not re.search(r"(rotary_emb|pos_embedding|relative_attention_bias)\.weight", name)
		quantizable = quantizable and quantize is not None and quantize(name, data.shape)

		filenames = []
		if id(data) in gptq_layers:
			filenames = (f"{name}.png", f"{name}.q8.exr", f"{name}.q8.idx.exr")
		elif quantizable:
			filenames = (f"{name}.exr", f"{name}.q8.png", f"{name}.q8.idx.exr")
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
			arrays = export_custom_int8(data)
		else:
			arrays = (data.cpu().numpy(),)
		for filename, array in zip(filenames, arrays):
			if array is None:
				continue
			print(f"\t\t{filename}\t{tuple(array.shape)}")
			if len(array.shape) == 1:
				array = pad_align(array, [4]).reshape(1, -1, 4)
			elif len(array.shape) == 2:
				array = pad_align(array, [1, 4]).reshape(array.shape[0], -1, 4)
			elif len(array.shape) == 3:
				array = pad_align(array, [1, 4, 1] if array.shape[-1] == 1 else [1, 1, 4]).reshape(array.shape[0], -1, 4)
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
		try:
			fast_tokenizer = convert_slow_tokenizer(tokenizer)
		except ValueError:
			fast_tokenizer = None

	# extract vocab
	vocab = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer.get_vocab()))))
	fixed = [x.encode() if x in tokenizer.added_tokens_encoder else\
		bytes((int(x[1:-1], 16),)) if re.fullmatch(r"<0x[0-9A-F]{2}>", x) else None for x in vocab]
	if fast_tokenizer is None:
		vocab = [x.encode() for x in vocab]
	elif type(fast_tokenizer.decoder).__name__ == "ByteLevel":
		from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
		byte_decoder = {c: b for b, c in bytes_to_unicode().items()}
		vocab = [fixed[i] or bytes(byte_decoder[c] for c in x) for i,x in enumerate(vocab)]
	else:
		prefix_len = len(tokenizer.decode([0]))
		vocab = [fixed[i] or tokenizer.decode([0, i])[prefix_len:].encode() for i in range(len(vocab))]

	# extract model
	merges = None
	weights = None
	if fast_tokenizer is not None:
		config = json.loads(fast_tokenizer.to_str())
		if type(fast_tokenizer.model).__name__ == "BPE":
			merges = [tokenizer.convert_tokens_to_ids(x.split(" ", 1)) for x in config["model"]["merges"]]
		elif type(fast_tokenizer.model).__name__ == "Unigram":
			weights = [x[1] for x in config["model"]["vocab"]] # TODO
		else:
			raise NotImplementedError(f"model {type(fast_tokenizer.model)} is not supported")

	added_tokens = [k for k, v in sorted(tokenizer.added_tokens_encoder.items(), key=lambda item: item[1])]
	vocab = ["".join(chr(b) for b in token) for token in vocab]
	if merges is not None:
		merges = [f"{vocab[i]} {vocab[j]}" for i, j in merges]

	# extract chat_template
	chat_templates = None
	if tokenizer.chat_template is not None:
		messages = [
			{"role": "system", "content": "{0}"},
			{"role": "user", "content": "{0}"},
			{"role": "assistant", "content": "{0}"},
		]
		try:
			tokenizer.apply_chat_template(messages[:1], tokenize=False)
		except: # system role is not supported
			messages = (messages[1:]*2)[:3]
		last_text = ""
		chat_templates = {}
		for i in range(3):
			text = tokenizer.apply_chat_template(messages[:i+1], tokenize=False)
			chat_templates[messages[i]["role"]] = text[len(last_text):]
			last_text = text

	# workaround for broken json parser
	escape_brackets = lambda lst: [re.sub(r'[\[\]\{\}]', lambda m: f"\\u{ord(m.group()) :04X}", x) for x in lst]
	unescape_brackets = lambda x: x.replace(r"\\u00", r"\u00")
	output = unescape_brackets(json.dumps(dict(
		added_tokens = added_tokens,
		vocab        = escape_brackets(vocab),
		merges       = escape_brackets(merges) if merges is not None else None,
		weights      = weights,
		bos_token_id = tokenizer.bos_token_id,
		eos_token_id = tokenizer.eos_token_id,
		unk_token_id = tokenizer.unk_token_id,
		chat_templates = chat_templates or None,
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
	o = {}

	# run encoder
	encoder_outputs = None
	if hasattr(model, "encoder"):
		encoder_input_ids = tokenizer(prompt, return_tensors="pt", padding=False, add_special_tokens=False).input_ids.to(model.device)
		encoder_outputs = model.encoder(input_ids=encoder_input_ids, return_dict=True)
		o["encoder_input_ids"] = encoder_input_ids[0].tolist()
		o["encoder_hidden_states"] = encoder_outputs.last_hidden_state[0].reshape(-1).tolist()

	# run tokenizer
	input_ids = tokenizer(prompt, return_tensors="pt", padding=False, add_special_tokens=False).input_ids.to(model.device).long()
	if encoder_outputs is not None:
		decoder_start_token_id = model.generation_config.decoder_start_token_id
		if input_ids.shape[1] == 0 or (input_ids[:, 0] != decoder_start_token_id).all().item():	
			input_ids = torch.cat([torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device) * decoder_start_token_id, input_ids], dim=-1)

	# run model
	outputs = model(**model.prepare_inputs_for_generation(input_ids, return_dict=True,
		**(dict(encoder_outputs=encoder_outputs))), output_hidden_states=True)
	o["input_ids"] = input_ids[0].tolist()
	o["logits"] = outputs.logits[0,-1].tolist()
	o["hidden_states"] = (outputs.hidden_states if encoder_outputs is None else outputs.decoder_hidden_states)[-1][0,-1].tolist()

	os.makedirs(folder, exist_ok=True)
	print(folder/"testcase.json")
	with open(folder/"testcase.json", "w") as f:
		json.dump(o, f)

def main():
	from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForTextToWaveform, AutoTokenizer
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('model', help='model id or path. for example: roneneldan/TinyStories-1M')
	parser.add_argument('folder', help='save path. for example: ../Model/')
	parser.add_argument('--tokenizer', type=str)
	parser.add_argument('--device', type=str)
	parser.add_argument('--dtype', type=str)
	parser.add_argument('--trust', action='store_true')
	parser.add_argument('--force', action='store_true')
	parser.add_argument('--quantize', type=float)
	
	args = parser.parse_args()

	folder = Path(args.folder)
	if re.search(r"[/\\]$", args.folder):
		folder /= Path(args.model).name
	print(f"convert: {args.model} => {folder}")
	disable_exllama()
	errs = []
	for auto_cls in [AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForTextToWaveform]:
		try:
			model = auto_cls.from_pretrained(args.model, trust_remote_code=args.trust,
				device_map=args.device or None, torch_dtype=getattr(torch, args.dtype) if args.dtype else None)
		except ValueError as e:
			errs.append(e)
			continue
		else:
			tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
			quantize = (lambda name, shape: np.prod(shape) >= args.quantize*1024*1024) if args.quantize else None
			print(f"model: {type(model)}")
			export_lm(model, folder, force_write=bool(args.force), quantize=quantize)
			if tokenizer is not None:
				export_tokenizer(tokenizer, folder)
			if auto_cls in [AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
				export_testcase(model, tokenizer, folder, force_write=bool(args.force))
			return
	for e in errs:
		print(e)

if __name__ == '__main__':
	with torch.no_grad():
		main()