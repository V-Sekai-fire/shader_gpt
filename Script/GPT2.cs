using UnityEngine;

namespace ShaderGPT {
public class GPT2 : GPTBase {
	[System.Serializable]
	class Config {
		public int n_layer;
		public int n_head;
		public float layer_norm_epsilon;
		public string activation_function;
		public int n_positions;
	}
	Config config;

	public new void OnEnable() {
		config = JsonUtility.FromJson<Config>(configJson.text);
		maxLength = Mathf.Min(maxLength, config.n_positions);
		base.OnEnable();
	}
	public override int Run(int positionId) {
		var input = InputTensor(tokens, positionId);
		var (hidden_states, logits) = GPT2LMHeadModel(input);
		ctx.Release(hidden_states);
		var next_tokens = Generate(input, ref logits);
		ctx.Release(input);
		var data = BatchRelease(ctx.GetData((RenderTexture)MarkRelease(next_tokens)));
		return Mathf.RoundToInt(data[0]);
	}
	public override void Test(Testcase testcase) {
		var input = InputTensor(testcase.input_ids);
		var (hidden_states, logits) = GPT2LMHeadModel(input);
		ctx.Release(input);
		AssertData((RenderTexture)hidden_states, -1, testcase.hidden_states, 8e-5f);
		AssertData((RenderTexture)logits, -1, testcase.logits, 2e-4f);
		ctx.Release(hidden_states);
		ctx.Release(logits);
	}
	public override void Bake() {
		var input = ctx.PersistentGPUTensor("input", 1, 1);
		var (hidden_states, logits) = GPT2LMHeadModel(input);
		ctx.Release(hidden_states);
		var next_tokens = Generate(input, ref logits);
		nn.Copy(input, next_tokens, ctx.Size(input));
		ctx.Release(next_tokens);
	}

	void GPT2Attention(ref Texture hidden_states, Texture input_ids, string path) {
		var query = nn.Linear(hidden_states, parameters[$"{path}.c_query.weight"], parameters[$"{path}.c_query.bias"]);
		var key   = nn.Linear(hidden_states, parameters[$"{path}.c_key.weight"],   parameters[$"{path}.c_key.bias"]);
		var value = nn.Linear(hidden_states, parameters[$"{path}.c_value.weight"], parameters[$"{path}.c_value.bias"]);
		ctx.Release(hidden_states);

		var keys   = ctx.PersistentGPUTensor($"{path}.k", maxLength, ctx.Size1(key), dtype:ctx.DType(key));
		var values = ctx.PersistentGPUTensor($"{path}.v", maxLength, ctx.Size1(value), dtype:ctx.DType(value));
		BatchRelease(nn.IndexCopy(keys,   (input_ids, 1), MarkRelease(key)));
		BatchRelease(nn.IndexCopy(values, (input_ids, 1), MarkRelease(value)));

		var window_size = config.n_positions;
		var norm_factor = 1f / Mathf.Sqrt(ctx.Size1(query)*4 / config.n_head);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.n_head));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.n_head, window:new Vector4(1-window_size, 1, 0, 1), offset:input_ids));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.n_head, weightT:true));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), parameters[$"{path}.c_proj.weight"], parameters[$"{path}.c_proj.bias"]));
	}
	void GPT2Block(ref Texture hidden_states, Texture input_ids, string path) {
		var attn_states = nn.GroupNorm(hidden_states, parameters[$"{path}.ln_1.weight"], parameters[$"{path}.ln_1.bias"], config.layer_norm_epsilon);
		GPT2Attention(ref attn_states, input_ids, path:$"{path}.attn");
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));

		var mlp_states = nn.GroupNorm(hidden_states, parameters[$"{path}.ln_2.weight"], parameters[$"{path}.ln_2.bias"], config.layer_norm_epsilon);
		mlp_states = BatchRelease(nn.Linear(MarkRelease(mlp_states), parameters[$"{path}.mlp.c_fc.weight"], parameters[$"{path}.mlp.c_fc.bias"]));
		mlp_states = BatchRelease(nn.Fusion(MarkRelease(mlp_states), func:TensorNN.ActFn(config.activation_function)));
		mlp_states = BatchRelease(nn.Linear(MarkRelease(mlp_states), parameters[$"{path}.mlp.c_proj.weight"], parameters[$"{path}.mlp.c_proj.bias"]));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture GPT2Model(Texture input_ids, string path) {
		var inputs_embeds   = nn.IndexSelect(parameters[$"{path}.wte.weight.T"], (input_ids, 0), inputT:true);
		var position_embeds = nn.IndexSelect(parameters[$"{path}.wpe.weight.T"], (input_ids, 1), inputT:true);
		var hidden_states   = BatchRelease(nn.Fusion(MarkRelease(inputs_embeds), add:MarkRelease(position_embeds)));
		for(int i=0; i<config.n_layer; i++)
			GPT2Block(ref hidden_states, input_ids, path:$"{path}.h.{i}");
		hidden_states = BatchRelease(nn.GroupNorm(MarkRelease(hidden_states), parameters[$"{path}.ln_f.weight"], parameters[$"{path}.ln_f.bias"], config.layer_norm_epsilon));
		return hidden_states;
	}
	(Texture, Texture) GPT2LMHeadModel(Texture input_ids) {
		var hidden_states = GPT2Model(input_ids, path:"transformer");
		parameters.TryGetValue("lm_head.weight.T", out var lm_head);
		var lm_logits = nn.Linear(hidden_states, lm_head ?? parameters["transformer.wte.weight.T"], weightT:true);
		return (hidden_states, lm_logits);
	}
}
}