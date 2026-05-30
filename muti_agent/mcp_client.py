# muti_agent/mcp_client.py
import sys
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

_mcp_client: Optional[MultiServerMCPClient] = None
_mcp_tools: Dict[str, Any] = {}
_initialized = False


def find_server_file(filename: str) -> Optional[Path]:
    """在项目根目录或 app 子目录中查找服务器文件"""
    base = Path(__file__).parent.parent
    candidates = [
        base / filename,
        base / "app" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


async def init_mcp_tools() -> Dict[str, Any]:
    global _mcp_client, _mcp_tools, _initialized
    if _initialized:
        return _mcp_tools

    leaf_path = find_server_file("leaf_mcp_server.py")
    lesion_path = find_server_file("lesion_mcp_server.py")
    #sam_path = find_server_file("sam_mcp_server.py")  # 新增

    if not leaf_path or not lesion_path: #or not sam_path:
        print("[MCP Client] 未找到 MCP 服务器文件，将使用本地工具")
        _initialized = True
        return {}

    python_exe = sys.executable
    print(f"[MCP Client] Using Python: {python_exe}")

    server_configs = {
        "leaf_seg": {
            "command": python_exe,
            "args": [str(leaf_path)],
            "transport": "stdio",
        },
        "lesion_seg": {
            "command": python_exe,
            "args": [str(lesion_path)],
            "transport": "stdio",
        },
    }

    try:
        _mcp_client = MultiServerMCPClient(server_configs)
        all_tools = await _mcp_client.get_tools()
        _mcp_tools = {tool.name: tool for tool in all_tools}
        _initialized = True
        print(f"[MCP Client] Loaded tools: {list(_mcp_tools.keys())}")
    except Exception as e:
        print(f"[MCP Client] 启动 MCP 服务器失败: {e}，将使用本地工具")
        _initialized = True
        return {}

    return _mcp_tools

def get_mcp_tool(name: str) -> Optional[Any]:
    return _mcp_tools.get(name)

async def close_mcp_client():
    global _mcp_client
    if _mcp_client:
        await _mcp_client.__aexit__(None, None, None)
        _mcp_client = None