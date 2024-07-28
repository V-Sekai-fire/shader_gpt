using UnityEngine;
using UnityEngine.Rendering;
#if UDON
using VRC.SDK3.Data;
using VRC.SDK3.Rendering;
#endif

namespace ShaderGPT.Udon {
#if UDON
[UdonSharp.UdonBehaviourSyncMode(UdonSharp.BehaviourSyncMode.None)]
public class UdonMonoBehaviour : UdonSharp.UdonSharpBehaviour
#else
public class UdonMonoBehaviour : MonoBehaviour
#endif
{
#if !UDON
	public UdonMonoBehaviour AsUdonBehaviour() => (UdonMonoBehaviour)this;
	public virtual void OnAsyncGpuReadbackComplete(AsyncGPUReadbackRequest request) {}
	protected AsyncGPUReadbackRequest AsyncGPUReadback_Request(Texture src, int mipIndex, TextureFormat dstFormat)
		=> AsyncGPUReadback.Request(src, mipIndex, dstFormat, OnAsyncGpuReadbackComplete);
#else
	public VRC.Udon.UdonBehaviour AsUdonBehaviour() => (VRC.Udon.UdonBehaviour)(Component)this;
	protected VRCAsyncGPUReadbackRequest AsyncGPUReadback_Request(Texture src, int mipIndex, TextureFormat dstFormat)
		=> VRCAsyncGPUReadback.Request(src, mipIndex, dstFormat, (VRC.Udon.Common.Interfaces.IUdonEventReceiver)this);
#endif
}
#if UDON
static class UdonExtensions {
	public static void SendMessage(this VRC.Udon.Common.Interfaces.IUdonEventReceiver r, string methodName) {
		r.SendCustomEvent(methodName);
	}
	public static VRCAsyncGPUReadbackRequest GetData<T>(this VRCAsyncGPUReadbackRequest r) {
		return r;
	}
	public static void CopyTo(this VRCAsyncGPUReadbackRequest r, float[] dst) {
		r.TryGetData(dst);
	}

	public static int GetInt(this DataToken dict, string key, int @default=0) {
		if(!dict.DataDictionary.TryGetValue(key, TokenType.Double, out var value_))
			return @default;
		return (int)value_.Double;
	}
	public static float GetFloat(this DataToken dict, string key, float @default=0) {
		if(!dict.DataDictionary.TryGetValue(key, TokenType.Double, out var value_))
			return @default;
		return (float)value_.Double;
	}
	public static string GetString(this DataToken dict, string key, string @default=null) {
		if(!dict.DataDictionary.TryGetValue(key, TokenType.String, out var value_))
			return @default;
		return value_.String;
	}
	public static int[] GetIntArray(this DataToken dict, string key) {
		if(!dict.DataDictionary.TryGetValue(key, TokenType.DataList, out var value_))
			return null;
		var list = value_.DataList;
		var cnt = list.Count;
		var arr = new int[cnt];
		for(int i=0; i<cnt; i++)
			arr[i] = (int)list[i].Double;
		return arr;
	}
	public static float[] GetFloatArray(this DataToken dict, string key) {
		if(!dict.DataDictionary.TryGetValue(key, TokenType.DataList, out var value_))
			return null;
		var list = value_.DataList;
		var cnt = list.Count;
		var arr = new float[cnt];
		for(int i=0; i<cnt; i++)
			arr[i] = (float)list[i].Double;
		return arr;
	}
	public static string[] GetStringArray(this DataToken dict, string key) {
		if(!dict.DataDictionary.TryGetValue(key, TokenType.DataList, out var value_))
			return null;
		var list = value_.DataList;
		var cnt = list.Count;
		var arr = new string[cnt];
		for(int i=0; i<cnt; i++)
			arr[i] = list[i].String;
		return arr;
	}
}
#endif
}