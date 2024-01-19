using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;
using System.Linq;

namespace ShaderGPT {
public class Phi : GPTBase {
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
		var (hidden_states, logits) = PhiForCausalLM(input);
		ctx.Release(input);
		ctx.Release(hidden_states);
		var next_tokens = MultinomialSample(logits, config.vocab_size, temperature);
		ctx.Release(logits);
		var data = BatchRelease(ctx.GetData((RenderTexture)MarkRelease(next_tokens)));
		return Mathf.RoundToInt(data[0]);
	}
	public override void Test(Testcase testcase) {
		var input = InputTensor(testcase.input_ids);
		var (hidden_states, logits) = PhiForCausalLM(input);
		ctx.Release(input);
		AssertData((RenderTexture)hidden_states, -1, testcase.hidden_states, 1e-4f);
		AssertData((RenderTexture)logits, -1, testcase.logits, 1e-4f);
		ctx.Release(hidden_states);
		ctx.Release(logits);
	}
	public override void Bake() {
		var input = ctx.PersistentGPUTensor("input", 1, 1);
		var (hidden_states, logits) = PhiForCausalLM(input);
		ctx.Release(hidden_states);
		var next_tokens = MultinomialSample(logits, config.vocab_size);
		ctx.Release(logits);
		nn.Copy(input, next_tokens, ctx.Size(input));
		ctx.Release(next_tokens);
	}

	void MHA(ref Texture hidden_states, Texture input_ids, string path) {
		var query = nn.Linear(hidden_states, parameters[$"{path}.Wq.weight"]);
		var key   = nn.Linear(hidden_states, parameters[$"{path}.Wk.weight"]);
		var value = nn.Linear(hidden_states, parameters[$"{path}.Wv.weight"]);
		ctx.Release(hidden_states);
		query = BatchRelease(nn.Fusion(MarkRelease(query), add:parameters[$"{path}.Wq.bias"]));
		key   = BatchRelease(nn.Fusion(MarkRelease(key),   add:parameters[$"{path}.Wk.bias"]));
		value = BatchRelease(nn.Fusion(MarkRelease(value), add:parameters[$"{path}.Wv.bias"]));

		var rotary = nn.Embedding(input_ids, null, parameters[$"{path}.rotary_emb.weight"]);
		query = BatchRelease(nn.Rotary(MarkRelease(query), rotary, groups:config.n_head));
		key   = BatchRelease(nn.Rotary(MarkRelease(key),   rotary, groups:config.n_head));
		ctx.Release(rotary);

		var keys   = ctx.PersistentGPUTensor($"{path}.k", maxLength, ctx.Size1(key), dtype:nn.dataType);
		var values = ctx.PersistentGPUTensor($"{path}.v", maxLength, ctx.Size1(value), dtype:nn.dataType);
		BatchRelease(nn.Scatter(keys,   input_ids, MarkRelease(key),   indexMask:new Vector2(0,1)));
		BatchRelease(nn.Scatter(values, input_ids, MarkRelease(value), indexMask:new Vector2(0,1)));

		var window_size = config.n_positions;
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.n_head));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), groups:config.n_head,
			indexRange:new Vector4(1-window_size, 1, 0, 1), rangeOffset:input_ids));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.n_head, transposeWeight:true));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), parameters[$"{path}.out_proj.weight"]));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:parameters[$"{path}.out_proj.bias"]));
	}
	void ParallelBlock(ref Texture hidden_states, Texture input_ids, string path) {
		var attn_states = nn.GroupNorm(hidden_states, parameters[$"{path}.ln.weight"], parameters[$"{path}.ln.bias"], config.layer_norm_epsilon);
		var mlp_states = BatchRelease(nn.Linear(attn_states, parameters[$"{path}.mlp.fc1.weight"]));
		mlp_states = BatchRelease(nn.Fusion(MarkRelease(mlp_states), add:parameters[$"{path}.mlp.fc1.bias"], func:TensorNN.ActFn(config.activation_function)));
		mlp_states = BatchRelease(nn.Linear(MarkRelease(mlp_states), parameters[$"{path}.mlp.fc2.weight"]));
		mlp_states = BatchRelease(nn.Fusion(MarkRelease(mlp_states), add:parameters[$"{path}.mlp.fc2.bias"]));

		MHA(ref attn_states, input_ids, path:$"{path}.mixer");
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture PhiModel(Texture input_ids, string path) {
		var hidden_states = nn.Embedding(input_ids, parameters[$"{path}.embd.wte.weight.T"], transposeWeight:true);
		for(int i=0; i<config.n_layer; i++)
			ParallelBlock(ref hidden_states, input_ids, path:$"{path}.h.{i}");
		// in modeling_phi.py, layernorm is implemented in PhiForCausalLM below
		return hidden_states;
	}
	(Texture, Texture) PhiForCausalLM(Texture input_ids) {
		var hidden_states = PhiModel(input_ids, path:"transformer");
		var lm_logits = BatchRelease(nn.GroupNorm(hidden_states, parameters["lm_head.ln.weight"], parameters["lm_head.ln.bias"], config.layer_norm_epsilon));
		lm_logits = nn.Linear(MarkRelease(lm_logits), parameters["lm_head.linear.weight.T"], transposeWeight:true);
		lm_logits = BatchRelease(nn.Fusion(MarkRelease(lm_logits), add:parameters["lm_head.linear.bias"]));
		return (hidden_states, lm_logits);
	}
}
}