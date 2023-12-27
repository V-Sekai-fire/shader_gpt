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
		var (hidden_states, logits) = PhiForCausalLM(input, position_id:positionId);
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

	void MHA(ref Texture hidden_states, string path, int position_id) {
		var query = nn.Linear(hidden_states, parameters[$"{path}.Wq.weight"]);
		var key   = nn.Linear(hidden_states, parameters[$"{path}.Wk.weight"]);
		var value = nn.Linear(hidden_states, parameters[$"{path}.Wv.weight"]);
		ctx.Release(hidden_states);
		query = BatchRelease(nn.AddAct(MarkRelease(query), parameters[$"{path}.Wq.bias"]));
		key   = BatchRelease(nn.AddAct(MarkRelease(key),   parameters[$"{path}.Wk.bias"]));
		value = BatchRelease(nn.AddAct(MarkRelease(value), parameters[$"{path}.Wv.bias"]));
		query = BatchRelease(nn.Rotary(MarkRelease(query), parameters[$"{path}.rotary_emb.weight"], new Vector2Int(position_id, 0), group:config.n_head));
		key   = BatchRelease(nn.Rotary(MarkRelease(key),   parameters[$"{path}.rotary_emb.weight"], new Vector2Int(position_id, 0), group:config.n_head));

		var keys   = ctx.PersistentGPUTensor($"{path}.k", maxLength, ctx.Size1(key), dtype:nn.dataType);
		var values = ctx.PersistentGPUTensor($"{path}.v", maxLength, ctx.Size1(value), dtype:nn.dataType);
		BatchRelease(nn.Copy(keys,   MarkRelease(key),   outputOffset:new Vector2Int(position_id, 0), size:ctx.Size(key)));
		BatchRelease(nn.Copy(values, MarkRelease(value), outputOffset:new Vector2Int(position_id, 0), size:ctx.Size(value)));

		var window_size = config.n_positions;
		var causal_mask = new Vector4(1, 1, position_id+1-window_size, position_id+1);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, head:config.n_head));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), group:config.n_head, rangeMask:causal_mask));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, head:config.n_head, transposeWeight:true));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), parameters[$"{path}.out_proj.weight"]));
		hidden_states = BatchRelease(nn.AddAct(MarkRelease(hidden_states), parameters[$"{path}.out_proj.bias"]));
	}
	void ParallelBlock(ref Texture hidden_states, string path, int position_id) {
		var residual = hidden_states;
		hidden_states = nn.GroupNorm(hidden_states, parameters[$"{path}.ln.weight"], parameters[$"{path}.ln.bias"], config.layer_norm_epsilon);
		var mlp_output = BatchRelease(nn.Linear(hidden_states, parameters[$"{path}.mlp.fc1.weight"]));
		mlp_output = BatchRelease(nn.AddAct(MarkRelease(mlp_output), parameters[$"{path}.mlp.fc1.bias"], TensorNN.ActFn(config.activation_function)));
		mlp_output = BatchRelease(nn.Linear(MarkRelease(mlp_output), parameters[$"{path}.mlp.fc2.weight"]));
		mlp_output = BatchRelease(nn.AddAct(MarkRelease(mlp_output), parameters[$"{path}.mlp.fc2.bias"]));

		MHA(ref hidden_states, path:$"{path}.mixer", position_id:position_id);
		hidden_states = BatchRelease(nn.AddAct(MarkRelease(hidden_states), MarkRelease(mlp_output)));
		hidden_states = BatchRelease(nn.AddAct(MarkRelease(hidden_states), MarkRelease(residual)));
	}
	Texture PhiModel(Texture input_ids, string path, int position_id) {
		var hidden_states = nn.Embedding(input_ids, parameters[$"{path}.embd.wte.weight.T"], transposeWeight:true);
		for(int i=0; i<config.n_layer; i++)
			ParallelBlock(ref hidden_states, path:$"{path}.h.{i}", position_id:position_id);
		// in modeling_phi.py, layernorm is implemented in PhiForCausalLM below
		return hidden_states;
	}
	(Texture, Texture) PhiForCausalLM(Texture input_ids, int position_id=0) {
		var hidden_states = PhiModel(input_ids, path:"transformer", position_id:position_id);
		var lm_logits = BatchRelease(nn.GroupNorm(hidden_states, parameters["lm_head.ln.weight"], parameters["lm_head.ln.bias"], config.layer_norm_epsilon));
		lm_logits = nn.Linear(MarkRelease(lm_logits), parameters["lm_head.linear.weight.T"], transposeWeight:true);
		lm_logits = BatchRelease(nn.AddAct(MarkRelease(lm_logits), parameters["lm_head.linear.bias"]));
		return (hidden_states, lm_logits);
	}
}
}