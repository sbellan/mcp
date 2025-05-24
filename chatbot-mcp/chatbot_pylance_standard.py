from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import MessageParam, ToolUnionParam, Message, ToolParam
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import Any, Iterable, List, Optional, cast
import asyncio
import nest_asyncio

nest_asyncio.apply()

load_dotenv()

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.anthropic = Anthropic()
        self.available_tools: Iterable[ToolParam] = []

    async def process_query(self, query):
        messages : Iterable[MessageParam] = [{'role':'user', 'content':query}]
        response: Message = self.anthropic.messages.create(max_tokens = 2024,
                                      model = 'claude-3-7-sonnet-20250219', 
                                      tools = self.available_tools, # tools exposed to the LLM
                                      messages = messages)
        process_query = True
        while process_query:
            assistant_content = []
            for content in response.content:
                if content.type =='text':
                    print(content.text)
                    assistant_content.append(content)
                    if(len(response.content) == 1):
                        process_query= False
                elif content.type == 'tool_use':
                    assistant_content.append(content)
                    messages.append({'role':'assistant', 'content':assistant_content})
                    tool_id = content.id
                    tool_args  = content.input
                    tool_name = content.name
    
                    print(f"Calling tool {tool_name} with args {tool_args}")
                    
                    # Call a tool
                    #result = execute_tool(tool_name, tool_args): not anymore needed
                    # tool invocation through the client session
                    if (session := self.session):
                        result = await self.session.call_tool(tool_name, arguments=tool_args) # type: ignore
                        messages.append({"role": "user", 
                                          "content": [
                                              {
                                                  "type": "tool_result",
                                                  "tool_use_id":tool_id,
                                                  "content": result.content
                                            }
                                        ]
                                    }   )
                        
                    response = self.anthropic.messages.create(max_tokens = 2024,
                                      model = 'claude-3-7-sonnet-20250219', 
                                      tools = self.available_tools,
                                      messages = messages) 
                    
                    if(len(response.content) == 1 and response.content[0].type == "text"):
                        print(response.content[0].text)
                        process_query= False

    
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="uv",  # Executable
            args=["run", "mcp_project/research_server.py"],  # Optional command line arguments
            env=None,  # Optional environment variables
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()
    
                # List available tools
                response: types.ListToolsResult = await session.list_tools()
                
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])
                
                self.available_tools = cast(Iterable[ToolParam],[{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in response.tools])
    
                await self.chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()
  

if __name__ == "__main__":
    asyncio.run(main())