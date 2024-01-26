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
	private string[] added_tokens;
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
			vocab = ToStringArray(vocab_.DataList);
			merges = ToStringArray(merges_.DataList);
			eos_token_id = (int)eos_token_id_.Double;
			if(tokenizer_.DataDictionary.TryGetValue("added_tokens", TokenType.DataList, out var added_tokens_))
				added_tokens = ToStringArray(added_tokens_.DataList);
			else
				added_tokens = null;
		}
#else
		var tokenizer = JsonUtility.FromJson<Tokenizer>(tokenizerJson.text);
		vocab = tokenizer.vocab;
		merges = tokenizer.merges;
		eos_token_id = tokenizer.eos_token_id;
		added_tokens = tokenizer.added_tokens;
#endif
	}

	void OnEnable() {
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
	
	private int[] tokenArray = new int[2048];
	private int tokenCount;
	public int[] Encode(string text) {
		tokenCount = 0;
		var textLen = text.Length;
		for(int i=0; i<textLen; ) {
			var found = false;
			if(added_tokens != null)
				foreach(var token in added_tokens)
					if(text[i] == token[0] && text.Substring(i).StartsWith(token)) {
						tokenArray[tokenCount++] = System.Array.LastIndexOf(vocab, token);
						i += token.Length;
						found = true;
						break;
					}
			if(found)
				continue;
			var j = SplitText(text, textLen, i);
			Tokenize(ConvertToUtf8(text, i, j));
			i = j;
		}
		return Take(tokenArray, tokenCount);
	}
	private string[] partArray = new string[2048];
	void Tokenize(string bytes) {
		var n = bytes.Length;
		var parts = partArray;
		var npart = 0;
		for(int i=0; i<n; i++) {
			var b = char.ConvertToUtf32(bytes, i);
			var k = b >= 0b11110000 ? 4 : b >= 0b11100000 ? 3 : b >= 0b11000000 ? 2 : 1;
			if(i+k <= n && System.Array.IndexOf(vocab, bytes.Substring(i, k)) >= 0) { // prefer full codepoint
				parts[npart++] = bytes.Substring(i, k);
				i += k-1;
			} else // byte fallback
				parts[npart++] = bytes.Substring(i, 1);
		}
		BytePairEncode(parts, npart);
	}
	void BytePairEncode(string[] parts, int n) {
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
				s += char.ConvertFromUtf32(0b11000000 | (c >> 6));
				s += char.ConvertFromUtf32(0b10000000 | (0b00111111 & c));
			} else {
				s += char.ConvertFromUtf32(0b11100000 | (c >> 12));
				s += char.ConvertFromUtf32(0b10000000 | (0b00111111 & (c >> 6)));
				s += char.ConvertFromUtf32(0b10000000 | (0b00111111 & c));
			}
			i += c < 0x10000 ? 1 : 2;
		}
		return s;
	}
	static T[] Take<T>(T[] src, int n) {
		var dst = new T[n];
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
		public string[] added_tokens;
		public int eos_token_id;
	}
#endif
}
}