using UnityEngine;
#if UDON
using VRC.SDK3.Data;
#endif

namespace ShaderGPT.Udon {
[DefaultExecutionOrder(-1)] // make it run before pipeline
public class GPTTokenizer : UdonMonoBehaviour {
	[Header("Tokenizer")]
	public TextAsset tokenizerJson;

	private string[] added_tokens;
	private string[] vocab;
	private string[] merges;
	private float [] weights;
	[System.NonSerialized] public int bos_token_id;
	[System.NonSerialized] public int eos_token_id;
	[System.NonSerialized] public int unk_token_id;
	void LoadTokenizer() {
#if UDON
		VRCJson.TryDeserializeFromJson(tokenizerJson.text, out var tokenizer);
		added_tokens = tokenizer.GetStringArray(nameof(added_tokens));
		vocab        = tokenizer.GetStringArray(nameof(vocab));
		merges       = tokenizer.GetStringArray(nameof(merges));
		weights      = tokenizer.GetFloatArray (nameof(weights));
		bos_token_id = tokenizer.GetInt        (nameof(bos_token_id), -1);
		eos_token_id = tokenizer.GetInt        (nameof(eos_token_id), -1);
		unk_token_id = tokenizer.GetInt        (nameof(unk_token_id), -1);
#else
		var tokenizer = JsonUtility.FromJson<Tokenizer>(tokenizerJson.text);
		added_tokens = tokenizer.added_tokens;
		vocab        = tokenizer.vocab;
		merges       = tokenizer.merges;
		weights      = tokenizer.weights;
		bos_token_id = tokenizer.bos_token_id;
		eos_token_id = tokenizer.eos_token_id;
		unk_token_id = tokenizer.unk_token_id;
	}
	[System.Serializable]
	class Tokenizer {
		public string[] added_tokens;
		public string[] vocab;
		public string[] merges;
		public float [] weights;
		public int bos_token_id = -1;
		public int eos_token_id = -1;
		public int unk_token_id = -1;
#endif
	}

	public void OnEnable() {
		if(vocab == null)
			LoadTokenizer();
	}

	[System.NonSerialized] public int decodeState;
	public string Decode(int token) {
		if(token < 0 || token >= vocab.Length)
			return string.Format("<error:{0}>", token);
		var s = vocab[token];
		var n = s.Length;
		var text = "";
		for(int i=0; i<n; i++) {
			var b = char.ConvertToUtf32(s, i);
			if(b <= 0b01111111)
				text += char.ConvertFromUtf32(b);
			else if(b > 0b10111111)
				decodeState = b;
			else {
				decodeState = (decodeState << 6) | (0b00111111 & b);
				if((decodeState & -0x800) == 0x3000)
					text += char.ConvertFromUtf32(decodeState - 0x3000);
				else if((decodeState & -0x10000) == 0xe0000)
					text += char.ConvertFromUtf32(decodeState - 0xe0000);
				else if((decodeState & -0x200000) == 0x3c00000)
					text += char.ConvertFromUtf32(decodeState - 0x3c00000);
			}
		}
		return text;
	}
	
	const int MAX_TOKENS = 16384;
	private int[] tokenArray = new int[MAX_TOKENS];
	private int tokenCount;
	public int[] Encode(string text) {
		tokenCount = 0;
		var textLen = text.Length;
		for(int i=0; i<textLen; ) {
			var j = textLen;
			var token = default(string);
			if(added_tokens != null)
				foreach(var t in added_tokens) {
					var k = text.IndexOf(t, i) & 0x7FFFFFFF;
					if(k < j) {
						j = k;
						token = t;
					}
				}
			while(i < j) {
				var k = PreTokenize(text, i, j);
				if(weights != null && weights.Length > 0)
					UnigramEncode(System.Text.Encoding.UTF8.GetBytes(text, i, k-i));
				else
					BytePairEncode(System.Text.Encoding.UTF8.GetBytes(text, i, k-i));
				i = k;
			}
			if(token != null) {
				tokenArray[tokenCount++] = System.Array.IndexOf(vocab, token);
				i += token.Length;
			}
		}
		return Take(tokenArray, tokenCount);
	}
	private string[] partArray = new string[MAX_TOKENS];
	private void BytePairEncode(byte[] bytes) {
		var n = bytes.Length;
		var chars = new char[n];
		System.Array.Copy(bytes, chars, n);
		var bstr = new string(chars);
		var parts = partArray;
		var npart = 0;
		for(int i=0; i<n; i++) {
			var b = bstr[i];
			var k = b >= 0b11110000 ? 4 : b >= 0b11100000 ? 3 : b >= 0b11000000 ? 2 : 1;
			if(i+k <= n && System.Array.IndexOf(vocab, bstr.Substring(i, k)) >= 0) { // prefer full codepoint
				parts[npart++] = bstr.Substring(i, k);
				i += k-1;
			} else // byte fallback
				parts[npart++] = bstr.Substring(i, 1);
		}
		BytePairMerge(parts, npart);
	}
	private void BytePairMerge(string[] parts, int n) {
		while(true) {
			var minRank = 0x7FFFFFFF;
			var minPos = -1;
			for(int i=1; i<n; i++) {
				var pair = string.Format("{0} {1}", parts[i-1], parts[i]);
				var rank = System.Array.IndexOf(merges, pair) & 0x7FFFFFFF;
				if(rank < minRank) {
					minRank = rank;
					minPos = i;
				}
			}
			if(minPos < 0) {
				for(int i=0; i<n; i++)
					tokenArray[tokenCount++] = System.Array.LastIndexOf(vocab, parts[i]); // avoid byte fallback tokens
				return;
			}
			parts[minPos-1] += parts[minPos];
			n--;
			System.Array.Copy(parts, minPos+1, parts, minPos, n-minPos);
		}
	}
	private void UnigramEncode(byte[] bytes) {
		var n = bytes.Length;
		var chars = new char[n];
		System.Array.Copy(bytes, chars, n);
		var bstr = new string(chars);
		var dp = new Vector2[n];
		for(int i=n-1; i>=0; i--) {
			var bestSum = float.NegativeInfinity;
			var bestIdx = i+1;
			var dpJ = default(Vector2);
			for(int j=n; j>i; dpJ=dp[--j]) {
				var chunk = bstr.Substring(i, j-i);
				var index = System.Array.IndexOf(vocab, chunk);
				if(index < 0)
					continue;
				var sum = dpJ.x + weights[index];
				if(sum > bestSum) {
					bestSum = sum;
					bestIdx = j;
				}
			}
			if(float.IsInfinity(bestSum)) {
				int j = i+1;
				while(j<n && char.ConvertToUtf32(bstr, j) < 0xC0)
					j ++;
				dpJ = j<n ? dp[j] : default;
				bestSum = dpJ.x + weights[unk_token_id];
				bestIdx = j;
			}
			dp[i] = new Vector2(bestSum, bestIdx);
			// Debug.Log($"dp[{i}]: {bestSum} {bestIdx}");
		}
		for(int i=0; i<n;) {
			var j = (int)dp[i].y;
			var chunk = bstr.Substring(i, j-i);
			var index = System.Array.IndexOf(vocab, chunk);
			if(index < 0)
				index = unk_token_id;
			tokenArray[tokenCount++] = index;
			i = j;
		}
	}

	static string Normalize(string text) {
		return text.Normalize(); // NFC normalizer
	}
	static int PreTokenize(string text, int start, int stop) {
		var i = start;
		// consume \s*(?!(?<= )\S)| ?(?=\S)
		while(i < stop && char.IsWhiteSpace(text, i))
			i ++;
		if(start < i && i < stop && char.ConvertToUtf32(text, i-1) == ' ')
			if(start < i-1) // split space for Metaspace
				return i-1;
		// split (?<=\p{L})[^\p{L}\p{N}\p{M}>]|(?<=\p{N})[^\p{L}\p{N}\p{M}>\p{P}] for ByteLevel
		var lastIsLN = false;
		while(i < stop && !char.IsWhiteSpace(text, i)) {
			var isLN = char.IsLetter(text, i) || char.IsNumber(text, i);
			if(!isLN && lastIsLN && !(char.IsNumber(text, i-1) && char.IsPunctuation(text, i))) {
				var cat = (int)char.GetUnicodeCategory(text, i);
				if((5 > cat || cat > 7) && char.ConvertToUtf32(text, i) != '>') // exclude marks and html tags
					return i;
			}
			lastIsLN = isLN;
			i ++;
		}
		// consume \s*(?!(?<= )\S)
		while(i < stop && char.IsWhiteSpace(text, i))
			i ++;
		if(start < i && i < stop && char.ConvertToUtf32(text, i-1) == ' ')
			i --;
		return i;
	}
	static T[] Take<T>(T[] src, int n) {
		var dst = new T[n];
		System.Array.Copy(src, dst, n);
		return dst;
	}
}
}