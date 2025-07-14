#!/usr/bin/env python3
"""
MCP Client with Optional Server Connection
A generic client that can work standalone or with MCP servers, integrated with OpenAI GPT-4o Mini
"""

import json
import asyncio
import subprocess
import sys
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    import openai
    import requests
except ImportError:
    print("Please install required libraries: pip install openai requests")
    sys.exit(1)


@dataclass
class MCPMessage:
    """Represents an MCP message"""
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class MCPClient:
    """MCP client with optional server connection and OpenAI integration"""
    
    def __init__(self, openai_api_key: Optional[str] = None, bosch_api_key: Optional[str] = None):
        self.process = None
        self.message_id = 0
        self.connected = False
        self.server_mode = False
        self.tools_cache = []
        self.resources_cache = []
        
        # Bosch OpenAI endpoint configuration
        self.bosch_endpoint = "https://aoai-farm.bosch-temp.com/api/openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18/chat/completions?api-version=2024-08-01-preview"
        
        # Initialize OpenAI client for Bosch endpoint
        bosch_key = bosch_api_key or os.getenv('BOSCH_API_KEY')
        openai_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if bosch_key:
            # Use Bosch OpenAI endpoint
            self.openai_client = openai.OpenAI(
                api_key=bosch_key,
                base_url="https://aoai-farm.bosch-temp.com/api/openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18"
            )
            self.api_key = bosch_key
            self.use_bosch_endpoint = True
            print("✓ Bosch OpenAI client initialized")
        elif openai_key:
            # Use standard OpenAI endpoint
            self.openai_client = openai.OpenAI(api_key=openai_key)
            self.api_key = openai_key
            self.use_bosch_endpoint = False
            print("✓ Standard OpenAI client initialized")
        else:
            print("⚠️  No API key found. Set BOSCH_API_KEY or OPENAI_API_KEY environment variable.")
            print("   Chat functionality will be limited.")
            self.openai_client = None
            self.use_bosch_endpoint = False
        
    def _next_id(self) -> int:
        """Generate next message ID"""
        self.message_id += 1
        return self.message_id
    
    async def connect(self, command: List[str]) -> bool:
        """Connect to MCP server via subprocess"""
        try:
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Initialize connection
            init_msg = MCPMessage(
                id=self._next_id(),
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {"listChanged": True},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "mcp-openai-client",
                        "version": "1.0.0"
                    }
                }
            )
            
            response = await self._send_message(init_msg)
            if response and not response.error:
                self.connected = True
                self.server_mode = True
                print("✓ Connected to MCP server")
                
                # Cache available tools and resources
                await self._refresh_cache()
                return True
            else:
                print(f"✗ Failed to initialize: {response.error if response else 'No response'}")
                return False
                
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def start_standalone(self):
        """Start in standalone mode without MCP server"""
        self.server_mode = False
        self.connected = True
        print("✓ Started in standalone mode (no MCP server)")
        print("  Available: OpenAI chat functionality")
    
    async def _send_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Send message to server and get response"""
        if not self.process or not self.server_mode:
            return None
            
        try:
            # Send message
            msg_dict = {
                "jsonrpc": message.jsonrpc,
                "id": message.id,
                "method": message.method,
                "params": message.params
            }
            
            msg_json = json.dumps(msg_dict) + "\n"
            self.process.stdin.write(msg_json.encode())
            await self.process.stdin.drain()
            
            # Read response
            response_line = await self.process.stdout.readline()
            if not response_line:
                return None
                
            response_data = json.loads(response_line.decode().strip())
            
            return MCPMessage(
                jsonrpc=response_data.get("jsonrpc", "2.0"),
                id=response_data.get("id"),
                result=response_data.get("result"),
                error=response_data.get("error")
            )
            
        except Exception as e:
            print(f"✗ Message send failed: {e}")
            return None
    
    async def _refresh_cache(self):
        """Refresh tools and resources cache"""
        if self.server_mode:
            self.tools_cache = await self.list_tools() or []
            self.resources_cache = await self.list_resources() or []
        else:
            self.tools_cache = []
            self.resources_cache = []
    
    async def list_resources(self) -> Optional[List[Dict[str, Any]]]:
        """List available resources"""
        if not self.server_mode:
            return []
            
        msg = MCPMessage(
            id=self._next_id(),
            method="resources/list",
            params={}
        )
        
        response = await self._send_message(msg)
        if response and response.result:
            return response.result.get("resources", [])
        return None
    
    async def list_tools(self) -> Optional[List[Dict[str, Any]]]:
        """List available tools"""
        if not self.server_mode:
            return []
            
        msg = MCPMessage(
            id=self._next_id(),
            method="tools/list",
            params={}
        )
        
        response = await self._send_message(msg)
        if response and response.result:
            return response.result.get("tools", [])
        return None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Optional[Any]:
        """Call a tool"""
        if not self.server_mode:
            return {"error": "No MCP server connected. Tools not available in standalone mode."}
            
        msg = MCPMessage(
            id=self._next_id(),
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments or {}
            }
        )
        
        response = await self._send_message(msg)
        if response and response.result:
            return response.result
        elif response and response.error:
            print(f"✗ Tool call failed: {response.error}")
        return None
    
    async def read_resource(self, uri: str) -> Optional[Any]:
        """Read a resource"""
        if not self.server_mode:
            return {"error": "No MCP server connected. Resources not available in standalone mode."}
            
        msg = MCPMessage(
            id=self._next_id(),
            method="resources/read",
            params={"uri": uri}
        )
        
        response = await self._send_message(msg)
        if response and response.result:
            return response.result
        elif response and response.error:
            print(f"✗ Resource read failed: {response.error}")
        return None
    
    def _tools_to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI function format"""
        if not self.server_mode:
            return []
            
        openai_tools = []
        
        for tool in self.tools_cache:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {}
                    })
                }
            }
            openai_tools.append(openai_tool)
        
        return openai_tools
    
    async def chat_with_openai(self, user_message: str, use_tools: bool = True) -> str:
        """Chat with OpenAI using available MCP tools"""
        if not self.openai_client:
            return "❌ OpenAI client not available. Please set BOSCH_API_KEY or OPENAI_API_KEY."
        
        system_message = "You are a helpful assistant."
        if self.server_mode and use_tools:
            system_message += " You have access to various tools and resources through MCP (Model Context Protocol)."
        else:
            system_message += " You are running in standalone mode without external tools."
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        tools = self._tools_to_openai_format() if (use_tools and self.server_mode) else []
        
        try:
            if self.use_bosch_endpoint:
                # Use custom requests for Bosch endpoint with specific headers
                response = await self._call_bosch_endpoint(messages, tools)
            else:
                # Use standard OpenAI client
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None
                )
            
            if self.use_bosch_endpoint:
                message = response["choices"][0]["message"]
            else:
                message = response.choices[0].message
            
            # Handle tool calls
            tool_calls = message.get("tool_calls") if self.use_bosch_endpoint else getattr(message, "tool_calls", None)
            
            if tool_calls and self.server_mode:
                for tool_call in tool_calls:
                    if self.use_bosch_endpoint:
                        tool_name = tool_call["function"]["name"]
                        arguments = json.loads(tool_call["function"]["arguments"])
                        tool_call_id = tool_call["id"]
                    else:
                        tool_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        tool_call_id = tool_call.id
                    
                    print(f"🔧 Calling tool: {tool_name} with args: {arguments}")
                    
                    # Call the MCP tool
                    result = await self.call_tool(tool_name, arguments)
                    
                    if result:
                        # Add tool response to conversation
                        if self.use_bosch_endpoint:
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tool_call]
                            })
                        else:
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tool_call.model_dump()]
                            })
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(result)
                        })
                
                # Get final response with tool results
                if self.use_bosch_endpoint:
                    final_response = await self._call_bosch_endpoint(messages, [])
                    return final_response["choices"][0]["message"]["content"]
                else:
                    final_response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages
                    )
                    return final_response.choices[0].message.content
            
            if self.use_bosch_endpoint:
                return message["content"]
            else:
                return message.content
            
        except Exception as e:
            return f"❌ OpenAI API error: {e}"
    
    async def _call_bosch_endpoint(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """Call Bosch OpenAI endpoint with custom headers"""
        headers = {
            "Content-Type": "application/json",
            "genaiplatform-farm-subscription-key": self.api_key
        }
        
        payload = {
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.7
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        # Use asyncio to run the synchronous request
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(self.bosch_endpoint, json=payload, headers=headers)
        )
        
        if response.status_code != 200:
            raise Exception(f"Bosch API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    async def disconnect(self):
        """Disconnect from server"""
        if self.process and self.server_mode:
            self.process.terminate()
            await self.process.wait()
            print("✓ Disconnected from MCP server")
        elif not self.server_mode:
            print("✓ Exiting standalone mode")
        
        self.connected = False
        self.server_mode = False


async def interactive_session(client: MCPClient):
    """Interactive session with optional MCP server"""
    mode = "with MCP Server" if client.server_mode else "Standalone"
    print(f"\n=== MCP Client {mode} + OpenAI GPT-4o Mini ===")
    
    if client.server_mode:
        print("Commands: resources, tools, call <tool_name>, read <uri>, chat <message>, help, quit")
    else:
        print("Commands: chat <message>, help, quit")
        print("Note: MCP tools and resources are not available in standalone mode")
    
    while True:
        try:
            command = input("\nmcp> ").strip()
            
            if not command:
                continue
                
            if command == "quit":
                break
                
            elif command == "help":
                print("Available commands:")
                if client.server_mode:
                    print("  resources         - List available resources")
                    print("  tools             - List available tools")
                    print("  call <tool>       - Call a tool directly")
                    print("  read <uri>        - Read a resource")
                    print("  refresh           - Refresh tools and resources cache")
                print("  chat <message>    - Chat with OpenAI" + (" using MCP tools" if client.server_mode else ""))
                print("  status            - Show connection status")
                print("  help              - Show this help")
                print("  quit              - Exit the session")
                
            elif command == "status":
                if client.server_mode:
                    print(f"✓ Connected to MCP server ({len(client.tools_cache)} tools, {len(client.resources_cache)} resources)")
                else:
                    print("✓ Standalone mode (no MCP server)")
                if client.openai_client:
                    endpoint = "Bosch OpenAI endpoint" if client.use_bosch_endpoint else "Standard OpenAI"
                    print(f"✓ OpenAI client available ({endpoint})")
                else:
                    print("✗ OpenAI client not available")
                
            elif command == "refresh":
                if client.server_mode:
                    await client._refresh_cache()
                    print("✓ Cache refreshed")
                else:
                    print("⚠️  No MCP server connected - nothing to refresh")
                
            elif command == "resources":
                if not client.server_mode:
                    print("⚠️  No MCP server connected - resources not available")
                    continue
                    
                resources = client.resources_cache
                if resources:
                    print(f"\n📁 Found {len(resources)} resources:")
                    for i, resource in enumerate(resources, 1):
                        print(f"  {i}. {resource.get('name', 'Unknown')} ({resource.get('uri', 'No URI')})")
                        if resource.get('description'):
                            print(f"     {resource['description']}")
                else:
                    print("No resources found")
                    
            elif command == "tools":
                if not client.server_mode:
                    print("⚠️  No MCP server connected - tools not available")
                    continue
                    
                tools = client.tools_cache
                if tools:
                    print(f"\n🔧 Found {len(tools)} tools:")
                    for i, tool in enumerate(tools, 1):
                        print(f"  {i}. {tool.get('name', 'Unknown')}")
                        if tool.get('description'):
                            print(f"     {tool['description']}")
                else:
                    print("No tools found")
                    
            elif command.startswith("call "):
                if not client.server_mode:
                    print("⚠️  No MCP server connected - tools not available")
                    continue
                    
                tool_name = command[5:].strip()
                if tool_name:
                    print(f"Calling tool: {tool_name}")
                    result = await client.call_tool(tool_name)
                    if result:
                        print(f"Result: {json.dumps(result, indent=2)}")
                else:
                    print("Usage: call <tool_name>")
                    
            elif command.startswith("read "):
                if not client.server_mode:
                    print("⚠️  No MCP server connected - resources not available")
                    continue
                    
                uri = command[5:].strip()
                if uri:
                    print(f"Reading resource: {uri}")
                    result = await client.read_resource(uri)
                    if result:
                        print(f"Content: {json.dumps(result, indent=2)}")
                else:
                    print("Usage: read <uri>")
                    
            elif command.startswith("chat "):
                message = command[5:].strip()
                if message:
                    print("🤖 Thinking...")
                    response = await client.chat_with_openai(message)
                    print(f"\n🤖 GPT-4o Mini: {response}")
                else:
                    print("Usage: chat <your message>")
                    
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {e}")


async def main():
    """Main function"""
    client = MCPClient()
    
    # Check if server command is provided
    if len(sys.argv) >= 2:
        # Server mode
        server_command = sys.argv[1:]
        print(f"Connecting to MCP server: {' '.join(server_command)}")
        
        if await client.connect(server_command):
            try:
                await interactive_session(client)
            finally:
                await client.disconnect()
        else:
            print("Failed to connect to MCP server")
            print("Starting in standalone mode instead...")
            client.start_standalone()
            try:
                await interactive_session(client)
            finally:
                await client.disconnect()
    else:
        # Standalone mode
        print("No MCP server specified - starting in standalone mode")
        print("Usage: python mcp_client.py [server_command] [args...]")
        print("Example: python mcp_client.py python server.py")
        print("\nStarting standalone mode with OpenAI chat only...")
        
        client.start_standalone()
        try:
            await interactive_session(client)
        finally:
            await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())