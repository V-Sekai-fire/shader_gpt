using UnityEngine;
using System.Collections.Generic;
using System.Linq;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace ShaderGPT {
public class BasicTTS : MonoBehaviour {
	[Header("Model")]
	public Shader[] shaders;
	public Texture[] textures;
	public TextAsset configJson;
	public TextAsset tokenizerJson;
	public TextAsset testcaseJson;

	public enum Task {
		Run = 0,
		Test,
		Bake,
	}
	[Header("Task")]
	public Task task;
	public UnityEngine.UI.Text inputText;
	public AudioSource outputSource;

	private TensorNN nn;
	private TensorContext ctx {
		get => nn.ctx;
		set { nn.ctx = value; }
	}
	private Models.Vits model;
	private Tokenizer tokenizer;

	public void OnEnable() {
		nn = new TensorNN(){
			ctx = new TensorContext(),
			kernels = shaders.ToDictionary(x => x.name.Split('/')[1], x => x),
		};
		model = new Models.Vits(nn, Models.VitsConfig.FromPretrained(configJson));
		model.LoadStateDict(textures);
		tokenizer = JsonUtility.FromJson<Tokenizer>(tokenizerJson.text);
		var testcase = testcaseJson ? JsonUtility.FromJson<Testcase>(testcaseJson.text) : null;

		if(task == Task.Run) {
			Run();
			Debug.Assert(ctx.TensorCount() == 0);
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
		if(outputSource && outputSource.clip)
			Object.Destroy(outputSource.clip);
	}

	void Run() {
		var text = inputText.text.Normalize(System.Text.NormalizationForm.FormKD).ToLower();
		var input_ids_data = new List<int>();
		foreach(char ch in text) {
			var chs = ch.ToString();
			var idx = System.Array.IndexOf(tokenizer.vocab, chs);
			if(idx < 0)
				idx = System.Array.IndexOf(tokenizer.vocab, " ");
			if(idx >= 0) {
				input_ids_data.Add(0);
				input_ids_data.Add(idx);
			}
		}
		input_ids_data.Add(0);

		var indices_data = new List<int>();
		for(int i=0; i<input_ids_data.Count; i++)
			for(int j=i%2==0?1 : 3; j>0; j--)
				indices_data.Add(i);

		var input_ids = InputTensor(input_ids_data);
		var indices = InputTensor(indices_data);
		Debug.Log($"lengths: {ctx.Size0(input_ids)}, {ctx.Size0(indices)}");
		var o = model.VitsModel(input_ids, indices);
		ctx.Release(input_ids);
		ctx.Release(indices);

		var waveform = ctx.GetData((RenderTexture)o.waveform);
		ctx.Release(o.hidden_states);
		ctx.Release(o.spectrogram);
		ctx.Release(o.waveform);

		if(outputSource.clip)
			Object.Destroy(outputSource.clip);
		var clip = AudioClip.Create("tts", waveform.Length, 1, model.config.sampling_rate, false);
		clip.SetData(waveform, 0);
		outputSource.clip = clip;
		outputSource.Play();
	}
	void Test(Testcase testcase) {
		model.config.noise_scale = 0;

		var input_ids = InputTensor(testcase.input_ids);
		var indices = InputTensor(testcase.indices);
		Debug.Log($"lengths: {ctx.Size0(input_ids)}, {ctx.Size0(indices)}");
		var o = model.VitsModel(input_ids, indices);
		ctx.Release(input_ids);
		ctx.Release(indices);

		var stopWatch = System.Diagnostics.Stopwatch.StartNew();
		ctx.GetData((RenderTexture)o.waveform);
		stopWatch.Stop();
		Debug.Log($"ElapsedMilliseconds: {stopWatch.ElapsedMilliseconds}");

		AssertData((RenderTexture)o.hidden_states, 0, testcase.hidden_states, 3e-6f);
		AssertData((RenderTexture)o.spectrogram, 0, testcase.spectrogram, 1e-5f);
		AssertData((RenderTexture)o.waveform, 0, testcase.waveform, 5e-4f);
		ctx.Release(o.hidden_states);
		ctx.Release(o.spectrogram);
		ctx.Release(o.waveform);
	}
	void Bake() {
		var max_length = 768;
		var max_index_length = max_length*2;
		var max_waveform_length = max_index_length;
		foreach(var x in model.config.upsample_rates)
			max_waveform_length *= x;

		var inputs = ctx.CPUTensor(max_length, 1);
		var indices = ctx.CPUTensor(max_index_length, 1);
		var lengths = ctx.CPUTensor(1, 1);
		var waveform = ctx.GPUTensor(max_waveform_length/65536, 16384);
		nn.Copy(ctx.Slice(waveform,  1, 1), ctx.Slice(inputs,  1, 1));
		nn.Copy(ctx.Slice(waveform,  1, 1), ctx.Slice(indices, 1, 1));
		nn.Copy(ctx.Slice(waveform,  1, 1), ctx.Slice(lengths, 1, 1));

		var o = model.VitsModel(inputs, indices,
			input_padding_mask: (new Vector4(-ctx.Size0(inputs),  0, 1, 0), lengths),
			output_padding_mask:(new Vector4(-ctx.Size0(indices), 0, 1, 1), lengths));
		ctx.Release(inputs);
		ctx.Release(indices);
		ctx.Release(lengths);

		nn.Copy(waveform, o.waveform, reshape:true);
		ctx.Release(o.hidden_states);
		ctx.Release(o.spectrogram);
		ctx.Release(o.waveform);
		ctx.Release(waveform);
	}

	Texture InputTensor(IList<int> input_ids, int position_id=0) {
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
		public int[] input_ids;
		public int[] indices;
		public float[] hidden_states;
		public float[] spectrogram;
		public float[] waveform;
	}
}
}