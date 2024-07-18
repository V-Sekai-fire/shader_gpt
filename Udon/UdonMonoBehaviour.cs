using UnityEngine;
using UnityEngine.Rendering;

namespace ShaderGPT.Udon {
#if UDON
[UdonSharp.UdonBehaviourSyncMode(UdonSharp.BehaviourSyncMode.None)]
public class UdonMonoBehaviour : UdonSharp.UdonSharpBehaviour
#else
public class UdonMonoBehaviour : MonoBehaviour
#endif
{
#if !UDON
	public virtual void OnAsyncGpuReadbackComplete(AsyncGPUReadbackRequest request) {}
	protected AsyncGPUReadbackRequest AsyncGPUReadback_Request(Texture src, int mipIndex, TextureFormat dstFormat)
		=> AsyncGPUReadback.Request(src, mipIndex, dstFormat, OnAsyncGpuReadbackComplete);
#else
	protected VRC.SDK3.Rendering.VRCAsyncGPUReadbackRequest AsyncGPUReadback_Request(Texture src, int mipIndex, TextureFormat dstFormat)
		=> VRC.SDK3.Rendering.VRCAsyncGPUReadback.Request(src, mipIndex, dstFormat, (VRC.Udon.Common.Interfaces.IUdonEventReceiver)this);
#endif
}
}