from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
#from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
import requests

from connection import model
from agents import Agent, ModelSettings, Runner, function_tool

#_________________________________Tool_______________________________

@function_tool
def sdk_tool():
    """Tool to fetch OpenAI Agents SDK documentation."""
    try:
        doc = requests.get("https://openai.github.io/openai-agents-python/")
        doc.raise_for_status()
        print("Fetched documentation:", doc.text[:500])  # Log first 500 chars
        return doc.text
    except Exception as e:
        print("Error in sdk_tool:", str(e))
        return f"Error fetching documentation: {str(e)}"

#_________________________________Agent_______________________________

sdk_agent = Agent(
    name="OpenAI Agent SDK",
    instructions="""You are an OpenAI SDK agent.
    Your job is to answer user queries related to concepts, code,
    and everything related to OpenAI Agents SDK by using the documentation available in the tool.""",
    model=model,
    tools=[sdk_tool],
    model_settings=ModelSettings(tool_choice="auto"),  # Changed to "auto"
)

runner = Runner()

#_________________________________Fastapi App_______________________________

app = FastAPI(title="OpenAI SDK Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "I'm alive and ready!"}

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):

    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    messages = [{"role": "user", "content": text}]

    try:
        result = await runner.run(sdk_agent, input=messages)
        print(result.final_output)
        reply = None

        if hasattr(result, "output"):
            if isinstance(result.final_output, str):
                reply = result.final_output
            elif isinstance(result.final_output, list) and len(result.final_output) > 0:
                reply = result.final_output[-1].get("content", None)
        elif hasattr(result, "final_output"):
            reply = result.final_output

        if not reply:
            reply = "⚠️ No meaningful response from SDK Agent."

        return {"reply": reply}

    except Exception as e:
        print("Error in chat_endpoint:", str(e))
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

#_____________________________Main Entry File____________________________

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(app, host="0.0.0.0", port=7860)