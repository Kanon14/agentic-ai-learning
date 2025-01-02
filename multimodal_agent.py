from praisonaiagents import Agent, Task, PraisonAIAgents
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Create Vision Analysis Agent
vision_agent = Agent(
    name="VisionAnalyst",
    role="Computer Vision Specialist",
    goal="Analyze images and videos to extract meaningful information",
    backstory="""You are an expert in computer vision and image analysis.
    You excel at describing images, detecting objects, and understanding visual content.
    """,
    llm="gpt-4o-mini",
    self_reflect=False
)

# 1. Task with Image URL
task1 = Task(
    name="analyze_landmark", 
    description="Describe this famous landmark and its architectural features.",
    expected_output="Detailed description of the landmark's architecture and significance",
    agent=vision_agent,
    images=["https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Yasaka-dori_early_morning_with_street_lanterns_and_the_Tower_of_Yasaka_%28Hokan-ji_Temple%29%2C_Kyoto%2C_Japan.jpg/640px-Yasaka-dori_early_morning_with_street_lanterns_and_the_Tower_of_Yasaka_%28Hokan-ji_Temple%29%2C_Kyoto%2C_Japan.jpg"]
)

# 2. Task with Local Image File
task2 = Task(
    name="analyze_local_image",
    description="What objects can you see in this image? Describe their arrangement.",
    expected_output="Detailed description of objeccts and their spatial relationships",
    agent=vision_agent,
    images=["./test_sample/kagami_mochi.png"]
)

# 3. Task with Video File
task3 = Task(
    name="analyze_video",
    description="""Watch this video and provide:
    1. A summary of the main events
    2. Key objects and people visible
    3. Any text or important information shown
    4. The overall context and setting.""",
    expected_output="Comprehensive analysis of the video content",
    agent=vision_agent,
    videos=["./test_sample/sample_04.mp4"]
)

agents = PraisonAIAgents(
    agents=[vision_agent],
    tasks=[task1, task2, task3],
    process="sequential",
    verbose=1
)

agents.start()