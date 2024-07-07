using UnityEngine;
using System.Collections.Generic;
using System.Linq;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace ShaderGPT {
public class BasicLM : MonoBehaviour {
	[Header("Model")]
	public Shader[] shaders;
	public Texture[] textures;
	public TextAsset configJson;
	public TextAsset tokenizerJson;
	public TextAsset testcaseJson;
	public GenerationConfig generationConfig;

	public enum Task {
		Run = 0,
		Test,
		Bake,
	}
	[Header("Task")]
	public Task task;
	public UnityEngine.UI.Text outputText;
	public float interval = 0.1f;

	private TensorNN nn;
	private TensorContext ctx {
		get => nn.ctx;
		set { nn.ctx = value; }
	}
	private ModelForCausalLM model;
	private Tokenizer tokenizer;

	private List<int> tokens;
	private float nextTime;
	private int positionId;
	
	public void OnEnable() {
		nn = new TensorNN(){
			ctx = new TensorContext(),
			kernels = shaders.ToDictionary(x => x.name.Split('/')[1], x => x),
		};
		model = AutoModelForCausalLM.FromPretrained(nn, configJson, textures);
		model.generation_config = generationConfig;
		tokenizer = JsonUtility.FromJson<Tokenizer>(tokenizerJson.text);
		var testcase = testcaseJson ? JsonUtility.FromJson<Testcase>(testcaseJson.text) : null;

		if(task == Task.Run) {
			if(model is ModelForSeq2SeqLM)
				RunEncoder(testcase);
			nextTime = Time.time;
			positionId = 0;
			if(tokens == null)
				tokens = new List<int>(testcase.input_ids);

			var text = "";
			for(int i=0; i<tokens.Count; i++)
				text += tokenizer.vocab[tokens[i]];
			if(outputText)
				outputText.text = text;
			else
				Debug.Log(text);
		} else if(task == Task.Test) {
			Debug.Log($"Testing {this.name} ({model})");
			Test(testcase);
			Debug.Assert(ctx.TensorCount() == model.cache.Count);
		} else if(task == Task.Bake) {
#if UNITY_EDITOR
			var path = System.IO.Path.GetDirectoryName(AssetDatabase.GetAssetPath(configJson)) + ".prefab";
			Debug.Log(path);

			ctx = new TensorTracer();
			Bake();
			Debug.Assert(ctx.TensorCount() == model.cache.Count);
			EditorGUIUtility.PingObject(((TensorTracer)ctx).Export(path));
#endif
		}
		RenderTexture.active = null; // suppress warning
	}
	public void OnDisable() {
		model.CacheClear();
	}
	public void Update() {
		if(task == Task.Run) {
			if(tokens.Count >= generationConfig.max_length)
				return;
			if(Time.time < nextTime)
				return;
			nextTime = Time.time + interval;
			
			var token = Run(positionId);
			Debug.Assert(ctx.TensorCount() == model.cache.Count);
			positionId = tokens.Count;
			tokens.Add(token);
			if(outputText)
				outputText.text += tokenizer.vocab[token];
			else
				Debug.Log(tokenizer.vocab[token]);
		}
	}
	static Dictionary<System.Type, (float,float)> testErrMap = new Dictionary<System.Type, (float,float)>() {
		{typeof(Models.GPT2), (8e-5f, 2e-4f)},
		{typeof(Models.GPTNeo), (5e-5f, 2e-4f)},
		{typeof(Models.GPTNeoX), (6e-3f, 6e-3f)},
		{typeof(Models.Llama), (1e-4f, 4e-5f)},
		{typeof(Models.Phi), (3e-5f, 5e-5f)},
		{typeof(Models.OpenELM), (4e-5f, 1e-4f)},
		{typeof(Models.T5), (3e-5f, 2e-4f)},
	};
	int encoder_input_length;
	void RunEncoder(Testcase testcase) {
		encoder_input_length = testcase.encoder_input_ids.Length;
		var encoder_input = InputTensor(testcase.encoder_input_ids, chan2:encoder_input_length);
		var encoder_output = ((ModelForSeq2SeqLM)model).ForSeq2SeqLM(encoder_input, null);
		ctx.Release(encoder_output.encoder_hidden_states);
		ctx.Release(encoder_input);
	}
	int Run(int positionId) {
		var input = InputTensor(tokens, positionId, chan2:encoder_input_length);
		var output = model.ForCausalLM(input);
		ctx.Release(output.hidden_states);
		var next_tokens = model.Generate(input, ref output.logits);
		ctx.Release(input);
		var data = ctx.GetData((RenderTexture)next_tokens);
		ctx.Release(next_tokens);
		return Mathf.RoundToInt(data[0]);
	}
	void Test(Testcase testcase) {
		var (hidden_states_err, logits_err) = testErrMap[model.GetType()];
		if(model is ModelForSeq2SeqLM) {
			encoder_input_length = testcase.encoder_input_ids.Length;
			var encoder_input = InputTensor(testcase.encoder_input_ids, chan2:encoder_input_length);
			var encoder_output = ((ModelForSeq2SeqLM)model).ForSeq2SeqLM(encoder_input, null);
			ctx.Release(encoder_input);
			AssertData((RenderTexture)encoder_output.encoder_hidden_states, 0, testcase.encoder_hidden_states, 1e-3f);
			ctx.Release(encoder_output.encoder_hidden_states);
		}
		var input = InputTensor(testcase.input_ids, chan2:encoder_input_length);
		var output = model.ForCausalLM(input);
		ctx.Release(input);
		AssertData((RenderTexture)output.hidden_states, -1, testcase.hidden_states, hidden_states_err);
		AssertData((RenderTexture)output.logits, -1, testcase.logits, logits_err);
		ctx.Release(output.hidden_states);
		ctx.Release(output.logits);
	}
	void Bake() {
		var input = ctx.GPUTensor(1, 1);
		if(model is ModelForSeq2SeqLM) {
			var encoder_input = ctx.CPUTensor(generationConfig.max_length, 1);
			nn.Copy(input, ctx.Slice(encoder_input,  1, 1));
			var encoder_output = ((ModelForSeq2SeqLM)model).ForSeq2SeqLM(encoder_input, null);
			ctx.Release(encoder_output.encoder_hidden_states);
			ctx.Release(encoder_input);
			((TensorTracer)ctx).Split();
		}
		var output = model.ForCausalLM(input);
		ctx.Release(output.hidden_states);
		var next_tokens = model.Generate(input, ref output.logits);
		nn.Copy(input, next_tokens);
		ctx.Release(next_tokens);
		ctx.Release(input);
	}

	Texture InputTensor(IList<int> input_ids, int position_id=0, int? size0=null, int chan2=0) {
		var n = input_ids.Count-position_id;
		var inputData = new float[(size0??n)*4];
		for(int i=0; i<n; i++) {
			inputData[i*4+0] = input_ids[i+position_id];
			inputData[i*4+1] = i+position_id;
			inputData[i*4+2] = chan2;
		}
		var input = ctx.CPUTensor((size0??n), 1);
		ctx.SetData(input, inputData);
		return input;
	}

	void AssertData(RenderTexture rt, int row, float[] value, float eps) {
		var col = ctx.Size1(rt) * 4;
		var offset = (row>=0 ? row : ctx.Size0(rt)+row) * col;
		var count = Mathf.Min(col, value.Length);
		var data = ctx.GetData(rt);
		var errorL1 = 0f;
		var errorL2 = 0f;
		var errorLi = 0f;
		for(int i=0; i<count; i++) {
			var error = Mathf.Abs(data[offset+i] - value[i]);
			errorL1 += error;
			errorL2 += error*error;
			errorLi = Mathf.Max(errorLi, error);
		}
		errorL1 = errorL1/count;
		errorL2 = Mathf.Sqrt(errorL2/count);
		if(Mathf.Abs(errorLi) < eps) {
			Debug.Log($"error: L1={errorL1}, L2={errorL2}, Li={errorLi}");
		} else {
			Debug.LogError($"error: L1={errorL1}, L2={errorL2}, Li={errorLi}");
			ctx.DebugTensor(rt);
		}
	}

	[System.Serializable]
	class Tokenizer {
		public string[] vocab;
	}
	[System.Serializable]
	class Testcase {
		public int[] encoder_input_ids;
		public float[] encoder_hidden_states;
		public int[] input_ids;
		public float[] hidden_states;
		public float[] logits;
	}
}
}