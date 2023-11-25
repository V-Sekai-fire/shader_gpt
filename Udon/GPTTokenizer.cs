using UnityEngine;
#if UDON
using VRC.SDK3.Data;
#endif

namespace ShaderGPT.Udon {
[DefaultExecutionOrder(-1)] // make it run before pipeline
#if UDON
[UdonSharp.UdonBehaviourSyncMode(UdonSharp.BehaviourSyncMode.None)]
public class GPTTokenizer : UdonSharp.UdonSharpBehaviour
#else
public class GPTTokenizer : MonoBehaviour
#endif
{
	[Header("Tokenizer")]
	public TextAsset tokenizerJson;

	private string[] vocab;
	private string[] merges;
	[System.NonSerialized] public int eos_token_id;
	void LoadTokenizer() {
#if UDON
		if(!VRCJson.TryDeserializeFromJson(tokenizerJson.text, out var tokenizer_))
			Debug.LogError($"tokenizer_: {tokenizer_}");
		else if(!tokenizer_.DataDictionary.TryGetValue("vocab", TokenType.DataList, out var vocab_))
			Debug.LogError($"vocab_: {vocab_}");
		else if(!tokenizer_.DataDictionary.TryGetValue("merges", TokenType.DataList, out var merges_))
			Debug.LogError($"merges_: {merges_}");
		else if(!tokenizer_.DataDictionary.TryGetValue("eos_token_id", TokenType.Double, out var eos_token_id_))
			Debug.LogError($"eos_token_id_: {eos_token_id_}");
		else {
			merges = ToStringArray(merges_.DataList);
			vocab = ToStringArray(vocab_.DataList);
			eos_token_id = (int)eos_token_id_.Double;
		}
#else
		var tokenizer = JsonUtility.FromJson<Tokenizer>(tokenizerJson.text);
		vocab = tokenizer.vocab;
		merges = tokenizer.merges;
		eos_token_id = tokenizer.eos_token_id;
#endif
	}

	public string testText;
	void OnEnable() {
		if(vocab == null)
			LoadTokenizer();

		var tokens = Encode(testText);
		var text = "";
		for(int i=0; i<tokens.Length; i++)
			text = string.Format("{0}{1},", text, tokens[i]);
		Debug.Log(text);
	}

	private int utfCode, utfStep;
	public void ResetDecode() {
		utfStep = utfCode = 0;
	}
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
			else if(b <= 0b10111111) {
				utfCode = (b & 0b111111) | (utfCode << 6);
				if(--utfStep == 0) {
					if(utfCode <= 0xD7FF)
						text += char.ConvertFromUtf32(utfCode);
					else if(utfCode >= 0xE000 && utfCode <= 0x10FFFF)
						text += char.ConvertFromUtf32(utfCode);
				}
			} else if(b <= 0b11011111) {
				utfCode = (b & 0b11111);
				utfStep = 1;
			} else if(b <= 0b11101111) {
				utfCode = (b & 0b1111);
				utfStep = 2;
			} else {
				utfCode = (b & 0b111);
				utfStep = 3;
			}
		}
		return text;
	}
	
	private int[] tokenArray = new int[2048];
	private int tokenCount;
	public int[] Encode(string text) {
		tokenCount = 0;
		var textLen = text.Length;
		for(int i=0; i<textLen; ) {
			var j = SplitText(text, textLen, i);
			var s = ConvertToUtf8(text, i, j);
			var n = s.Length;
			var parts = new string[n];
			for(int k=0; k<n; k++)
				parts[k] = s.Substring(k, 1);
			BytePairEncode(parts);
			i = j;
		}
		return Resize(tokenArray, tokenCount);
	}
	void BytePairEncode(string[] parts) {
		var n = parts.Length;
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
					tokenArray[tokenCount++] = System.Array.IndexOf(vocab, parts[i]);
				return;
			}
			parts[minPos-1] += parts[minPos];
			n--;
			System.Array.Copy(parts, minPos+1, parts, minPos, n-minPos);
		}
	}

	static int SplitText(string text, int textLen, int i) {
		var ch = char.ConvertToUtf32(text, i);
		if(ch == ' ') {
			i++;
			if(i == textLen)
				return i;
		} else if(ch == '\'') {
			i++;
			if(i == textLen)
				return i;
			if(!char.IsLetter(text, i))
				return i;
		}
		if(char.IsLetter(text, i)) {
			do {
				if(++i >= textLen) break;
			} while(char.IsLetter(text, i));
		} else if(char.IsNumber(text, i)) {
			do {
				if(++i >= textLen) break;
			} while(char.IsNumber(text, i));
		} else if(char.IsWhiteSpace(text, i)) {
			do {
				if(++i >= textLen) break;
			} while(char.IsWhiteSpace(text, i));
		} else {
			do {
				if(++i >= textLen) break;
				if(char.IsLetterOrDigit(text, i)) break;
				if(char.IsWhiteSpace(text, i)) break;
			} while(true);
		}
		return i;
	}
	static string ConvertToUtf8(string text, int i, int j) {
		var s = "";
		while(i<j) {
			var c = char.ConvertToUtf32(text, i);
			if (c <= 0x7F) {
				s += char.ConvertFromUtf32(c);
			} else if (c <= 0x07FF) {
				s += char.ConvertFromUtf32(0xC0 | (c >> 6));
				s += char.ConvertFromUtf32(0x80 | (c & 0x3F));
			} else {
				s += char.ConvertFromUtf32(0xE0 | (c >> 12));
				s += char.ConvertFromUtf32(0x80 | ((c >> 6) & 0x3F));
				s += char.ConvertFromUtf32(0x80 | (c & 0x3F));
			}
			i += c < 0x10000 ? 1 : 2;
		}
		return s;
	}
	static int[] Resize(int[] src, int n) {
		var dst = new int[n];
		System.Array.Copy(src, dst, n);
		return dst;
	}
#if UDON
	static string[] ToStringArray(DataList lst) {
		var n = lst.Count;
		var arr = new string[n];
		for(int i=0; i<n; i++)
			arr[i] = lst[i].String;
		return arr;
	}
#else
	[System.Serializable]
	public class Tokenizer {
		public string[] vocab;
		public string[] merges;
		public int eos_token_id;
	}
#endif
}
}