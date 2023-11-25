using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;
using System.Linq;

namespace ShaderGPT {
public class GPT2 : GPTBase {
	[System.Serializable]
	class Config {
		public int n_layer;
		public int n_head;
		public float layer_norm_epsilon;
		public string activation_function;
		public int n_positions;
		public int vocab_size;
	}
	Config config;

	public new void OnEnable() {
		config = JsonUtility.FromJson<Config>(configJson.text);
		maxLength = Mathf.Min(maxLength, config.n_positions);
		base.OnEnable();
	}
	public override int Run(int positionId) {
		var input = InputTensor(tokens, positionId);
		var (hidden_states, logits) = GPT2LMHeadModel(input, position_id:positionId);
		ctx.Release(input);
		ctx.Release(hidden_states);
		var next_tokens = MultinomialSample(logits, config.vocab_size, temperature);
		ctx.Release(logits);
		var data = BatchRelease(ctx.GetData((RenderTexture)MarkRelease(next_tokens)));
		return Mathf.RoundToInt(data[0]);
	}
	public override void Test(Testcase testcase) {
		var input = InputTensor(testcase.input_ids);
		var (hidden_states, logits) = GPT2LMHeadModel(input);
		ctx.Release(input);
		AssertData((RenderTexture)hidden_states, -1, testcase.hidden_states, 3e-4f);
		AssertData((RenderTexture)logits, -1, testcase.logits, 2e-3f);
		ctx.Release(hidden_states);
		ctx.Release(logits);
	}
	public override void Bake() {
		var input = ctx.PersistentGPUTensor("input", 1, 1);
		var (hidden_states, logits) = GPT2LMHeadModel(input);
		ctx.Release(hidden_states);
		var next_tokens = MultinomialSample(logits, config.vocab_size);
		ctx.Release(logits);
		nn.Copy(input, next_tokens, ctx.Size(input));
		ctx.Release(next_tokens);
	}

	void GPT2Attention(ref Texture hidden_states, string path, int position_id) {
		var query = nn.Linear(hidden_states, parameters[$"{path}.c_query.weight"]);
		var key   = nn.Linear(hidden_states, parameters[$"{path}.c_key.weight"]);
		var value = nn.Linear(hidden_states, parameters[$"{path}.c_value.weight"]);
		ctx.Release(hidden_states);
		query = BatchRelease(nn.AddAct(MarkRelease(query), parameters[$"{path}.c_query.bias"]));
		key   = BatchRelease(nn.AddAct(MarkRelease(key),   parameters[$"{path}.c_key.bias"]));
		value = BatchRelease(nn.AddAct(MarkRelease(value), parameters[$"{path}.c_value.bias"]));

		var keys   = ctx.PersistentGPUTensor($"{path}.k", maxLength, ctx.Size1(key), dtype:nn.dataType);
		var values = ctx.PersistentGPUTensor($"{path}.v", maxLength, ctx.Size1(value), dtype:nn.dataType);
		BatchRelease(nn.Copy(keys,   MarkRelease(key),   outputOffset:new Vector2Int(position_id, 0), size:ctx.Size(key)));
		BatchRelease(nn.Copy(values, MarkRelease(value), outputOffset:new Vector2Int(position_id, 0), size:ctx.Size(value)));

		var window_size = config.n_positions;
		var causal_mask = new Vector4(1, 1, position_id+1-window_size, position_id+1);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, head:config.n_head));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), group:config.n_head, rangeMask:causal_mask));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, head:config.n_head, transposeWeight:true));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), parameters[$"{path}.c_proj.weight"]));
		hidden_states = BatchRelease(nn.AddAct(MarkRelease(hidden_states), parameters[$"{path}.c_proj.bias"]));
	}
	void GPT2Block(ref Texture hidden_states, string path, int position_id) {
		var residual = hidden_states;
		hidden_states = nn.GroupNorm(residual, parameters[$"{path}.ln_1.weight"], parameters[$"{path}.ln_1.bias"], config.layer_norm_epsilon);
		GPT2Attention(ref hidden_states, path:$"{path}.attn", position_id:position_id);
		hidden_states = BatchRelease(nn.AddAct(MarkRelease(hidden_states), MarkRelease(residual)));

		residual = hidden_states;
		hidden_states = nn.GroupNorm(residual, parameters[$"{path}.ln_2.weight"], parameters[$"{path}.ln_2.bias"], config.layer_norm_epsilon);
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), parameters[$"{path}.mlp.c_fc.weight"]));
		hidden_states = BatchRelease(nn.AddAct(MarkRelease(hidden_states), parameters[$"{path}.mlp.c_fc.bias"], TensorNN.ActFn(config.activation_function)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), parameters[$"{path}.mlp.c_proj.weight"]));
		hidden_states = BatchRelease(nn.AddAct(MarkRelease(hidden_states), parameters[$"{path}.mlp.c_proj.bias"]));
		hidden_states = BatchRelease(nn.AddAct(MarkRelease(hidden_states), MarkRelease(residual)));
	}
	Texture GPT2Model(Texture input_ids, string path, int position_id) {
		var hidden_states = nn.Embedding(input_ids, parameters[$"{path}.wte.weight.T"], parameters[$"{path}.wpe.weight.T"], transposeWeight:true);
		for(int i=0; i<config.n_layer; i++)
			GPT2Block(ref hidden_states, path:$"{path}.h.{i}", position_id:position_id);
		hidden_states = BatchRelease(nn.GroupNorm(MarkRelease(hidden_states), parameters[$"{path}.ln_f.weight"], parameters[$"{path}.ln_f.bias"], config.layer_norm_epsilon));
		return hidden_states;
	}
	(Texture, Texture) GPT2LMHeadModel(Texture input_ids, int position_id=0) {
		var hidden_states = GPT2Model(input_ids, path:"transformer", position_id:position_id);
		var lm_logits = nn.Linear(hidden_states, parameters["transformer.wte.weight.T"], transposeWeight:true);
		return (hidden_states, lm_logits);
	}
}
}