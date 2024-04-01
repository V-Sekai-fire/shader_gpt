using UnityEngine;
using System.Text.RegularExpressions;

namespace ShaderGPT {
public class Phi : GPTBase {
	[System.Serializable]
	class Config {
		public int num_hidden_layers;
		public int num_attention_heads;
		public int num_key_value_heads;
		public float layer_norm_eps;
		public string hidden_act;
		public int max_position_embeddings;
	}
	Config config;

	public new void OnEnable() {
		config = JsonUtility.FromJson<Config>(configJson.text);
		maxLength = Mathf.Min(maxLength, config.max_position_embeddings);
		base.OnEnable();
	}
	public override int Run(int positionId) {
		var input = InputTensor(tokens, positionId);
		var (hidden_states, logits) = PhiForCausalLM(input);
		ctx.Release(hidden_states);
		var next_tokens = Generate(input, ref logits);
		ctx.Release(input);
		var data = BatchRelease(ctx.GetData((RenderTexture)MarkRelease(next_tokens)));
		return Mathf.RoundToInt(data[0]);
	}
	public override void Test(Testcase testcase) {
		var input = InputTensor(testcase.input_ids);
		var (hidden_states, logits) = PhiForCausalLM(input);
		ctx.Release(input);
		AssertData((RenderTexture)hidden_states, -1, testcase.hidden_states, 1e-5f);
		AssertData((RenderTexture)logits, -1, testcase.logits, 5e-5f);
		ctx.Release(hidden_states);
		ctx.Release(logits);
	}
	public override void Bake() {
		var input = ctx.PersistentGPUTensor("input", 1, 1);
		var (hidden_states, logits) = PhiForCausalLM(input);
		ctx.Release(hidden_states);
		var next_tokens = Generate(input, ref logits);
		nn.Copy(input, next_tokens, ctx.Size(input));
		ctx.Release(next_tokens);
	}

	void PhiAttention(ref Texture hidden_states, Texture input_ids, string path) {
		var query = nn.Linear(hidden_states, parameters[$"{path}.q_proj.weight"], parameters[$"{path}.q_proj.bias"]);
		var key   = nn.Linear(hidden_states, parameters[$"{path}.k_proj.weight"], parameters[$"{path}.k_proj.bias"]);
		var value = nn.Linear(hidden_states, parameters[$"{path}.v_proj.weight"], parameters[$"{path}.v_proj.bias"]);
		ctx.Release(hidden_states);

		parameters.TryGetValue(Regex.Replace($"{path}.rotary_emb.weight", @"[.]\d+[.]", ".0."), out var rotary_emb);
		var rotary = nn.Embedding(input_ids, rotary_emb ?? parameters[$"{path}.rotary_emb.weight"], chan:1);
		query = BatchRelease(nn.Rotary(MarkRelease(query), rotary, groups:config.num_attention_heads));
		key   = BatchRelease(nn.Rotary(MarkRelease(key),   rotary, groups:config.num_key_value_heads));
		ctx.Release(rotary);

		var keys   = ctx.PersistentGPUTensor($"{path}.k", maxLength, ctx.Size1(key), dtype:ctx.DType(key));
		var values = ctx.PersistentGPUTensor($"{path}.v", maxLength, ctx.Size1(value), dtype:ctx.DType(value));
		BatchRelease(nn.Scatter(keys,   input_ids, MarkRelease(key),   chan:1));
		BatchRelease(nn.Scatter(values, input_ids, MarkRelease(value), chan:1));

		var window_size = config.max_position_embeddings;
		var norm_factor = 1f / Mathf.Sqrt(ctx.Size1(query)*4 / config.num_attention_heads);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.num_attention_heads, weightHeads:config.num_key_value_heads));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.num_attention_heads, window:new Vector4(1-window_size, 1, 0, 1), offset:input_ids));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.num_attention_heads, weightHeads:config.num_key_value_heads, transposeWeight:true));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), parameters[$"{path}.dense.weight"], parameters[$"{path}.dense.bias"]));
	}
	void PhiDecoderLayer(ref Texture hidden_states, Texture input_ids, string path) {
		var attn_states = nn.GroupNorm(hidden_states, parameters[$"{path}.input_layernorm.weight"], parameters[$"{path}.input_layernorm.bias"], config.layer_norm_eps);
		var mlp_states = BatchRelease(nn.Linear(attn_states, parameters[$"{path}.mlp.fc1.weight"], parameters[$"{path}.mlp.fc1.bias"]));
		mlp_states = BatchRelease(nn.Fusion(MarkRelease(mlp_states), func:TensorNN.ActFn(config.hidden_act)));
		mlp_states = BatchRelease(nn.Linear(MarkRelease(mlp_states), parameters[$"{path}.mlp.fc2.weight"], parameters[$"{path}.mlp.fc2.bias"]));

		PhiAttention(ref attn_states, input_ids, path:$"{path}.self_attn");
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture PhiModel(Texture input_ids, string path) {
		var hidden_states = nn.Embedding(input_ids, parameters[$"{path}.embed_tokens.weight.T"], transposeWeight:true);
		for(int i=0; i<config.num_hidden_layers; i++)
			PhiDecoderLayer(ref hidden_states, input_ids, path:$"{path}.layers.{i}");
		hidden_states = BatchRelease(nn.GroupNorm(MarkRelease(hidden_states), parameters[$"{path}.final_layernorm.weight"], parameters[$"{path}.final_layernorm.bias"], config.layer_norm_eps));
		return hidden_states;
	}
	(Texture, Texture) PhiForCausalLM(Texture input_ids) {
		var hidden_states = PhiModel(input_ids, path:"model");
		var lm_logits = nn.Linear(hidden_states, parameters["lm_head.weight.T"], parameters["lm_head.bias"], transposeWeight:true);
		return (hidden_states, lm_logits);
	}
}
}