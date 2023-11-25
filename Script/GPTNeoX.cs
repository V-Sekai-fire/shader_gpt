using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;
using System.Linq;

namespace ShaderGPT {
public class GPTNeoX : GPTBase {
	[System.Serializable]
	class Config {
		public int num_hidden_layers;
		public int num_attention_heads;
		public float layer_norm_eps;
		public string hidden_act;
		public int max_position_embeddings;
		public int vocab_size;
	}
	Config config;

	public new void OnEnable() {
		config = JsonUtility.FromJson<Config>(configJson.text);
		maxLength = Mathf.Min(maxLength, config.max_position_embeddings);
		base.OnEnable();
	}
	public override int Run(int positionId) {
		var input = InputTensor(tokens, positionId);
		var (hidden_states, logits) = GPTNeoXForCausalLM(input, position_id:positionId);
		ctx.Release(input);
		ctx.Release(hidden_states);
		var next_tokens = MultinomialSample(logits, config.vocab_size, temperature);
		ctx.Release(logits);
		var data = BatchRelease(ctx.GetData((RenderTexture)MarkRelease(next_tokens)));
		return Mathf.RoundToInt(data[0]);
	}
	public override void Test(Testcase testcase) {
		var input = InputTensor(testcase.input_ids);
		var (hidden_states, logits) = GPTNeoXForCausalLM(input);
		ctx.Release(input);
		AssertData((RenderTexture)hidden_states, -1, testcase.hidden_states, 3e-4f);
		AssertData((RenderTexture)logits, -1, testcase.logits, 2e-3f);
		ctx.Release(hidden_states);
		ctx.Release(logits);
	}
	public override void Bake() {
		var input = ctx.PersistentGPUTensor("input", 1, 1);
		var (hidden_states, logits) = GPTNeoXForCausalLM(input);
		ctx.Release(hidden_states);
		var next_tokens = MultinomialSample(logits, config.vocab_size);
		ctx.Release(logits);
		nn.Copy(input, next_tokens, ctx.Size(input));
		ctx.Release(next_tokens);
	}

	void GPTNeoXAttention(ref Texture hidden_states, string path, int position_id) {
		var query = nn.Linear(hidden_states, parameters[$"{path}.query.weight"]);
		var key   = nn.Linear(hidden_states, parameters[$"{path}.key.weight"]);
		var value = nn.Linear(hidden_states, parameters[$"{path}.value.weight"]);
		ctx.Release(hidden_states);
		query = BatchRelease(nn.AddAct(MarkRelease(query), parameters[$"{path}.query.bias"]));
		key   = BatchRelease(nn.AddAct(MarkRelease(key),   parameters[$"{path}.key.bias"]));
		value = BatchRelease(nn.AddAct(MarkRelease(value), parameters[$"{path}.value.bias"]));
		query = BatchRelease(nn.Rotary(MarkRelease(query), parameters[$"{path}.rotary_emb.weight"], new Vector2Int(position_id, 0), group:config.num_attention_heads));
		key   = BatchRelease(nn.Rotary(MarkRelease(key),   parameters[$"{path}.rotary_emb.weight"], new Vector2Int(position_id, 0), group:config.num_attention_heads));

		var keys   = ctx.PersistentGPUTensor($"{path}.k", maxLength, ctx.Size1(key), dtype:nn.dataType);
		var values = ctx.PersistentGPUTensor($"{path}.v", maxLength, ctx.Size1(value), dtype:nn.dataType);
		BatchRelease(nn.Copy(keys,   MarkRelease(key),   outputOffset:new Vector2Int(position_id, 0), size:ctx.Size(key)));
		BatchRelease(nn.Copy(values, MarkRelease(value), outputOffset:new Vector2Int(position_id, 0), size:ctx.Size(value)));

		var window_size = config.max_position_embeddings;
		var causal_mask = new Vector4(1, 1, position_id+1-window_size, position_id+1);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, head:config.num_attention_heads));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), group:config.num_attention_heads, rangeMask:causal_mask));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, head:config.num_attention_heads, transposeWeight:true));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), parameters[$"{path}.dense.weight"]));
		hidden_states = BatchRelease(nn.AddAct(MarkRelease(hidden_states), parameters[$"{path}.dense.bias"]));
	}
	void GPTNeoXLayer(ref Texture hidden_states, string path, int position_id) {
		var attn_output = nn.GroupNorm(hidden_states, parameters[$"{path}.input_layernorm.weight"], parameters[$"{path}.input_layernorm.bias"], config.layer_norm_eps);
		GPTNeoXAttention(ref attn_output, path:$"{path}.attention", position_id:position_id);

		var mlp_output = nn.GroupNorm(hidden_states, parameters[$"{path}.post_attention_layernorm.weight"], parameters[$"{path}.post_attention_layernorm.bias"], config.layer_norm_eps);
		mlp_output = BatchRelease(nn.Linear(MarkRelease(mlp_output), parameters[$"{path}.mlp.dense_h_to_4h.weight"]));
		mlp_output = BatchRelease(nn.AddAct(MarkRelease(mlp_output), parameters[$"{path}.mlp.dense_h_to_4h.bias"], TensorNN.ActFn(config.hidden_act)));
		mlp_output = BatchRelease(nn.Linear(MarkRelease(mlp_output), parameters[$"{path}.mlp.dense_4h_to_h.weight"]));
		mlp_output = BatchRelease(nn.AddAct(MarkRelease(mlp_output), parameters[$"{path}.mlp.dense_4h_to_h.bias"]));

		hidden_states = BatchRelease(nn.AddAct(MarkRelease(hidden_states), MarkRelease(attn_output)));
		hidden_states = BatchRelease(nn.AddAct(MarkRelease(hidden_states), MarkRelease(mlp_output)));
	}
	Texture GPTNeoXModel(Texture input_ids, string path, int position_id) {
		var hidden_states = nn.Embedding(input_ids, parameters[$"{path}.embed_in.weight.T"], transposeWeight:true);
		for(int i=0; i<config.num_hidden_layers; i++)
			GPTNeoXLayer(ref hidden_states, path:$"{path}.layers.{i}", position_id:position_id);
		hidden_states = BatchRelease(nn.GroupNorm(MarkRelease(hidden_states), parameters[$"{path}.final_layer_norm.weight"], parameters[$"{path}.final_layer_norm.bias"], config.layer_norm_eps));
		return hidden_states;
	}
	(Texture, Texture) GPTNeoXForCausalLM(Texture input_ids, int position_id=0) {
		var hidden_states = GPTNeoXModel(input_ids, path:"gpt_neox", position_id:position_id);
		var lm_logits = nn.Linear(hidden_states, parameters["embed_out.weight.T"], transposeWeight:true);
		return (hidden_states, lm_logits);
	}
}
}