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
		var (hidden_states, logits) = GPTNeoXForCausalLM(input);
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
		AssertData((RenderTexture)hidden_states, -1, testcase.hidden_states, 4e-3f);
		AssertData((RenderTexture)logits, -1, testcase.logits, 4e-3f);
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

	void GPTNeoXAttention(ref Texture hidden_states, Texture input_ids, string path) {
		var query = nn.Linear(hidden_states, parameters[$"{path}.query.weight"]);
		var key   = nn.Linear(hidden_states, parameters[$"{path}.key.weight"]);
		var value = nn.Linear(hidden_states, parameters[$"{path}.value.weight"]);
		ctx.Release(hidden_states);
		query = BatchRelease(nn.Fusion(MarkRelease(query), add:parameters[$"{path}.query.bias"]));
		key   = BatchRelease(nn.Fusion(MarkRelease(key),   add:parameters[$"{path}.key.bias"]));
		value = BatchRelease(nn.Fusion(MarkRelease(value), add:parameters[$"{path}.value.bias"]));

		var rotary = nn.Embedding(input_ids, null, parameters[$"{path}.rotary_emb.weight"]);
		query = BatchRelease(nn.Rotary(MarkRelease(query), rotary, groups:config.num_attention_heads));
		key   = BatchRelease(nn.Rotary(MarkRelease(key),   rotary, groups:config.num_attention_heads));
		ctx.Release(rotary);

		var keys   = ctx.PersistentGPUTensor($"{path}.k", maxLength, ctx.Size1(key), dtype:nn.dataType);
		var values = ctx.PersistentGPUTensor($"{path}.v", maxLength, ctx.Size1(value), dtype:nn.dataType);
		BatchRelease(nn.Scatter(keys,   input_ids, MarkRelease(key),   indexMask:new Vector2(0,1)));
		BatchRelease(nn.Scatter(values, input_ids, MarkRelease(value), indexMask:new Vector2(0,1)));

		var window_size = config.max_position_embeddings;
		var norm_factor = 1f / Mathf.Sqrt(ctx.Size1(query)*4 / config.num_attention_heads);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.num_attention_heads));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.num_attention_heads, indexRange:new Vector4(1-window_size, 1, 0, 1), rangeOffset:input_ids));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.num_attention_heads, transposeWeight:true));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), parameters[$"{path}.dense.weight"]));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:parameters[$"{path}.dense.bias"]));
	}
	void GPTNeoXLayer(ref Texture hidden_states, Texture input_ids, string path) {
		var attn_states = nn.GroupNorm(hidden_states, parameters[$"{path}.input_layernorm.weight"], parameters[$"{path}.input_layernorm.bias"], config.layer_norm_eps);
		GPTNeoXAttention(ref attn_states, input_ids, path:$"{path}.attention");

		var mlp_states = nn.GroupNorm(hidden_states, parameters[$"{path}.post_attention_layernorm.weight"], parameters[$"{path}.post_attention_layernorm.bias"], config.layer_norm_eps);
		mlp_states = BatchRelease(nn.Linear(MarkRelease(mlp_states), parameters[$"{path}.mlp.dense_h_to_4h.weight"]));
		mlp_states = BatchRelease(nn.Fusion(MarkRelease(mlp_states), add:parameters[$"{path}.mlp.dense_h_to_4h.bias"], func:TensorNN.ActFn(config.hidden_act)));
		mlp_states = BatchRelease(nn.Linear(MarkRelease(mlp_states), parameters[$"{path}.mlp.dense_4h_to_h.weight"]));
		mlp_states = BatchRelease(nn.Fusion(MarkRelease(mlp_states), add:parameters[$"{path}.mlp.dense_4h_to_h.bias"]));

		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture GPTNeoXModel(Texture input_ids, string path) {
		var hidden_states = nn.Embedding(input_ids, parameters[$"{path}.embed_in.weight.T"], transposeWeight:true);
		for(int i=0; i<config.num_hidden_layers; i++)
			GPTNeoXLayer(ref hidden_states, input_ids, path:$"{path}.layers.{i}");
		hidden_states = BatchRelease(nn.GroupNorm(MarkRelease(hidden_states), parameters[$"{path}.final_layer_norm.weight"], parameters[$"{path}.final_layer_norm.bias"], config.layer_norm_eps));
		return hidden_states;
	}
	(Texture, Texture) GPTNeoXForCausalLM(Texture input_ids) {
		var hidden_states = GPTNeoXModel(input_ids, path:"gpt_neox");
		var lm_logits = nn.Linear(hidden_states, parameters["embed_out.weight.T"], transposeWeight:true);
		return (hidden_states, lm_logits);
	}
}
}