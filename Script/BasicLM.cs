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

	[Header("Generation")]
	public int maxLength = 2048;
	public float temperature = 0;
	public float repetitionPenalty = 1f;
	public float interval = 0.1f;

	public enum Task {
		Run = 0,
		Test,
		Bake,
	}
	[Header("Task")]
	public Task task;
	public UnityEngine.UI.Text outputText;

	protected TensorNN nn;
	protected TensorContext ctx {
		get => nn.ctx;
		set { nn.ctx = value; }
	}
	private ModelForCausalLM model;
	private ModelForCausalLMConfig config;
	private Tokenizer tokenizer;

	private List<int> tokens;
	private float nextTime;
	private int positionId;
	
	public void OnEnable() {
		nn = new TensorNN(){
			ctx = new TensorContext(),
			kernels = shaders.ToDictionary(x => x.name.Split('/')[1], x => x),
		};
		model = ModelForCausalLM.FromPretrained(nn, configJson, textures);
		config = JsonUtility.FromJson<ModelForCausalLMConfig>(configJson.text);
		tokenizer = JsonUtility.FromJson<Tokenizer>(tokenizerJson.text);
		var testcase = testcaseJson ? JsonUtility.FromJson<Testcase>(testcaseJson.text) : null;

		if(task == Task.Run) {
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
			Debug.Assert(ctx.TensorCount() == 0);
		} else if(task == Task.Bake) {
#if UNITY_EDITOR
			var path = System.IO.Path.GetDirectoryName(AssetDatabase.GetAssetPath(configJson)) + ".prefab";
			Debug.Log(path);

			ctx = new TensorTracer();
			Bake();
			Debug.Assert(ctx.TensorCount() == 0);
			EditorGUIUtility.PingObject(((TensorTracer)ctx).Export(path));
#endif
		}
		RenderTexture.active = null; // suppress warning
	}
	public void OnDisable() {
		ctx.ReleasePersistent();
	}
	public void Update() {
		if(task == Task.Run) {
			if(tokens.Count >= maxLength)
				return;
			if(Time.time < nextTime)
				return;
			nextTime = Time.time + interval;
			
			var token = Run(positionId);
			Debug.Assert(ctx.TensorCount() == 0);
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
		{typeof(Models.GPTNeoX), (4e-3f, 6e-3f)},
		{typeof(Models.Llama), (4e-5f, 4e-5f)},
		{typeof(Models.Phi), (1e-5f, 5e-5f)},
	};
	public int Run(int positionId) {
		var input = InputTensor(tokens, positionId);
		var (hidden_states, logits) = model.ForCausalLM(input);
		ctx.Release(hidden_states);
		var next_tokens = Generate(input, ref logits);
		ctx.Release(input);
		var data = BatchRelease(ctx.GetData((RenderTexture)MarkRelease(next_tokens)));
		return Mathf.RoundToInt(data[0]);
	}
	public void Test(Testcase testcase) {
		var (hidden_states_err, logits_err) = testErrMap[model.GetType()];
		var input = InputTensor(testcase.input_ids);
		var (hidden_states, logits) = model.ForCausalLM(input);
		ctx.Release(input);
		AssertData((RenderTexture)hidden_states, -1, testcase.hidden_states, hidden_states_err);
		AssertData((RenderTexture)logits, -1, testcase.logits, logits_err);
		ctx.Release(hidden_states);
		ctx.Release(logits);
	}
	public void Bake() {
		var input = ctx.PersistentGPUTensor("input", 1, 1);
		var (hidden_states, logits) = model.ForCausalLM(input);
		ctx.Release(hidden_states);
		var next_tokens = Generate(input, ref logits);
		nn.Copy(input, next_tokens, ctx.Size(input));
		ctx.Release(next_tokens);
	}
	
	protected Texture InputTensor(IList<int> input_ids, int position_id=0) {
		var n = input_ids.Count-position_id;
		var inputData = new float[n*4];
		for(int i=0; i<n; i++) {
			inputData[i*4+0] = input_ids[i+position_id];
			inputData[i*4+1] = i+position_id;
		}
		var input = ctx.CPUTensor(n, 1);
		ctx.SetData(input, inputData);
		return input;
	}
	protected void RepetitionPenaltyLogitsProcessor(Texture input_ids, ref Texture scores, float penalty, Texture last_input_ids) {
		var inputs_T = nn.Transpose(input_ids, 1);
		inputs_T = BatchRelease(nn.Fusion(MarkRelease(inputs_T), @default:4*ctx.Size1(scores)*Vector4.one,
			window:new Vector4(-maxLength, ctx.Size0(last_input_ids), 0, 1), offset:last_input_ids));
		var mask = nn.Fusion(scores, scale:0f);
		BatchRelease(nn.IndexCopy((RenderTexture)mask, (MarkRelease(inputs_T), 0), null, fill:1f, axis1:true));
		var penal = nn.Fusion(scores, func:TensorNN.Keyword.FUNC_RELU, eps:penalty*penalty, scale:1f/penalty);
		var diff = BatchRelease(nn.Fusion(scores, scale:-1, add:MarkRelease(penal)));
		scores = BatchRelease(nn.Fusion(MarkRelease(mask), mul:MarkRelease(diff), add:MarkRelease(scores)));
	}
	protected Texture Generate(Texture input, ref Texture scores) {
		var inputs = ctx.PersistentGPUTensor("inputs", maxLength, 1);
		nn.IndexCopy(inputs, (input, 1), input);
		if(ctx.Size0(scores) > 1)
			scores = BatchRelease(nn.Copy(null, MarkRelease(scores),
				size:new Vector2Int(1, ctx.Size1(scores)), inputOffset:new Vector2Int(ctx.Size0(scores)-1, 0)));
		RepetitionPenaltyLogitsProcessor(inputs, ref scores, repetitionPenalty, input);
		var gumbel = BatchRelease(nn.Gumbel(MarkRelease(scores), temperature));
		return BatchRelease(nn.ArgMax(MarkRelease(gumbel), window:new Vector2(0, config.vocab_size)));
	}

	// utilities
	List<Texture> releaseList = new List<Texture>();
	protected T MarkRelease<T>(T tex) where T: Texture {
		releaseList.Add(tex);
		return tex;
	}
	protected T BatchRelease<T>(T x) {
		foreach(var tex in releaseList)
			ctx.Release(tex);
		releaseList.Clear();
		return x;
	}
	protected void AssertData(RenderTexture rt, int row, float[] value, float eps) {
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
		Debug.Log($"error: L1={errorL1}, L2={errorL2}, Li={errorLi}");
		Debug.Assert(Mathf.Abs(errorLi) < eps);
		if(Mathf.Abs(errorLi) >= eps)
			ctx.DebugTensor(rt);
	}

	[System.Serializable]
	public class Tokenizer {
		public string[] vocab;
	}
	[System.Serializable]
	public class Testcase {
		public int[] input_ids;
		public float[] hidden_states;
		public float[] logits;
	}
}
}